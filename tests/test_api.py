import asyncio
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from frametag.api import Analyzer, APIError, Cancelled, _retry_delay, diagnose_connection, list_models
from frametag.model_catalog import DEFAULT_MODELS, model_id, model_label, restore_model


GOOD = {
    "gender": "남성",
    "clothing": [{"color": "검정", "type": "반팔"}, {"color": "파랑", "type": "청바지"}],
    "objects": [{"name": "가방", "location": "손"}],
}


def reply(content=None, status=200, headers=None):
    if content is None:
        content = json.dumps(GOOD, ensure_ascii=False)
    return httpx.Response(status, headers=headers,
                          json={"choices": [{"message": {"content": content}}]})


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.photo = Path(self.temp.name) / "photo.png"
        Image.new("RGB", (2048, 1024), "blue").save(self.photo)

    def analyzer(self, handler, provider="NVIDIA"):
        return Analyzer(provider, "test-vision-model", "test-placeholder",
                        transport=httpx.MockTransport(handler))

    def test_resizes_jpeg_without_changing_original(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return reply()

        before = self.photo.read_bytes()
        result = self.analyzer(handler).analyze(self.photo, threading.Event())
        self.assertEqual(result["gender"], "남성")
        self.assertEqual(before, self.photo.read_bytes())
        uri = requests[0]["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
            self.assertEqual(image.size, (1024, 512))
            self.assertEqual(image.mode, "RGB")
        self.assertIn("청바지", requests[0]["messages"][0]["content"])

    def test_exif_rotation_and_no_exif_in_upload(self):
        self.photo = self.photo.with_suffix(".jpg")
        image = Image.new("RGB", (2048, 1024), "blue")
        exif = Image.Exif()
        exif[274] = 6
        image.save(self.photo, exif=exif)

        def handler(request):
            payload = json.loads(request.content)
            uri = payload["messages"][1]["content"][1]["image_url"]["url"]
            with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as uploaded:
                self.assertEqual(uploaded.size, (512, 1024))
                self.assertFalse(uploaded.getexif())
            return reply()

        self.analyzer(handler).analyze(self.photo, threading.Event())

    def test_gif_uses_first_frame(self):
        self.photo = self.photo.with_suffix(".gif")
        first = Image.new("RGB", (20, 20), "red")
        first.save(self.photo, save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")])

        def handler(request):
            payload = json.loads(request.content)
            uri = payload["messages"][1]["content"][1]["image_url"]["url"]
            with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as uploaded:
                red, _, blue = uploaded.getpixel((10, 10))
                self.assertGreater(red, blue)
            return reply()

        self.analyzer(handler).analyze(self.photo, threading.Event())

    def test_fences_and_missing_optional_arrays(self):
        result = self.analyzer(lambda _: reply('판정 결과:\n```json\n{"gender":"불명"}\n```')).analyze(
            self.photo, threading.Event())
        self.assertEqual(result["gender"], "불명")
        self.assertFalse(result["clothing"])
        self.assertFalse(result["objects"])

    def test_format_repair_contains_previous_response_and_is_bounded(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return reply("not JSON") if len(requests) < 3 else reply()

        result = self.analyzer(handler).analyze(self.photo, threading.Event())
        self.assertEqual(result["gender"], "남성")
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1]["messages"][2], {"role": "assistant", "content": "not JSON"})
        self.assertEqual(len(requests[2]["messages"]), 6)

    def test_unrelated_json_is_not_empty_success(self):
        calls = []

        def handler(request):
            calls.append(request)
            return reply('{"unexpected": "text"}')

        with self.assertRaises(APIError) as caught:
            self.analyzer(handler).analyze(self.photo, threading.Event())
        self.assertFalse(caught.exception.fatal)
        self.assertEqual(len(calls), 3)

    def test_transient_status_retried_then_success(self):
        calls = []

        def handler(request):
            calls.append(request)
            return reply(status=503, headers={"Retry-After": "0"}) if len(calls) == 1 else reply()

        self.analyzer(handler).analyze(self.photo, threading.Event())
        self.assertEqual(len(calls), 2)

    def test_transport_errors_retry_four_times(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ConnectError("unavailable", request=request)

        with patch("frametag.api._retry_delay", return_value=0):
            with self.assertRaises(APIError) as caught:
                self.analyzer(handler).analyze(self.photo, threading.Event())
        self.assertFalse(caught.exception.fatal)
        self.assertEqual(len(calls), 4)

    def test_auth_payment_and_model_errors_stop_immediately(self):
        for status in (401, 402, 403, 404, 410):
            with self.subTest(status=status):
                calls = []

                def handler(request):
                    calls.append(request)
                    return reply(status=status)

                with self.assertRaises(APIError) as caught:
                    self.analyzer(handler).analyze(self.photo, threading.Event())
                self.assertTrue(caught.exception.fatal)
                self.assertEqual(len(calls), 1)

    def test_exhausted_rate_limit_is_fatal(self):
        calls = []

        def handler(request):
            calls.append(request)
            return reply(status=429, headers={"Retry-After": "0"})

        with self.assertRaises(APIError) as caught:
            self.analyzer(handler).analyze(self.photo, threading.Event())
        self.assertTrue(caught.exception.fatal)
        self.assertEqual(len(calls), 4)

    def test_cancellation_interrupts_request_without_late_result(self):
        started = threading.Event()
        ended = threading.Event()
        cancel = threading.Event()
        results = []

        async def handler(request):
            started.set()
            try:
                await asyncio.sleep(10)
                return reply()
            finally:
                ended.set()

        analyzer = self.analyzer(handler)

        def run():
            try:
                results.append(analyzer.analyze(self.photo, cancel))
            except Exception as error:
                results.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(started.wait(2))
        began = time.monotonic()
        cancel.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - began, 1)
        self.assertTrue(ended.is_set())
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Cancelled)

    def test_cancellation_interrupts_retry_delay(self):
        requested = threading.Event()
        cancel = threading.Event()
        results = []

        def handler(request):
            requested.set()
            return reply(status=503, headers={"Retry-After": "1000"})

        def run():
            try:
                self.analyzer(handler).analyze(self.photo, cancel)
            except Exception as error:
                results.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(requested.wait(2))
        cancel.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(results[0], Cancelled)

    def test_pre_cancelled_does_not_read_image_or_send_request(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            self.analyzer(lambda _: self.fail("No request expected")).analyze(
                Path("missing.png"), cancel)

    def test_corrupt_and_decompression_bomb_images_do_not_send(self):
        self.photo.write_bytes(b"not an image")
        with self.assertRaises(APIError):
            self.analyzer(lambda _: self.fail("No request expected")).analyze(
                self.photo, threading.Event())
        Image.new("RGB", (15, 10)).save(self.photo)
        with patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(APIError):
                self.analyzer(lambda _: self.fail("No request expected")).analyze(
                    self.photo, threading.Event())

    def test_retry_after_is_bounded(self):
        self.assertEqual(_retry_delay(httpx.Response(429, headers={"Retry-After": "1000"}), 0), 8)
        self.assertEqual(_retry_delay(httpx.Response(429, headers={"Retry-After": "-1"}), 0), 0)

    def test_openai_payload_uses_public_endpoint_and_json_response(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
            payload = json.loads(request.content)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertIn("max_completion_tokens", payload)
            return reply()

        self.analyzer(handler, "OpenAI").analyze(self.photo, threading.Event())

    def test_model_discovery_sorted_unique(self):
        def handler(request):
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url), "https://api.openai.com/v1/models")
            return httpx.Response(200, json={"data": [
                {"id": "z"}, {"id": "a"}, {"id": "a"}, {"id": None},
            ]})

        self.assertEqual(list_models("OpenAI", "test-placeholder",
                                     transport=httpx.MockTransport(handler)), ["a", "z"])

    def test_nvidia_discovery_only_offers_supported_photo_candidates(self):
        available = ["nvidia/nemotron-nano-12b-v2-vl", "meta/llama-3.3-70b-instruct",
                     "qwen/qwen3.5-397b-a17b", "nvidia/nemotron-3-nano-omni-30b-a3b",
                     "qwen/qwen3.5-122b-a10b", "qwen/qwen3.5-122b-a10b"]
        def handler(request):
            self.assertEqual(str(request.url), "https://integrate.api.nvidia.com/v1/models")
            return httpx.Response(200, json={"data": [{"id": name} for name in available]})
        self.assertEqual(list_models("NVIDIA", "test-placeholder", transport=httpx.MockTransport(handler)),
                         ["qwen/qwen3.5-122b-a10b", "qwen/qwen3.5-397b-a17b",
                          "nvidia/nemotron-3-nano-omni-30b-a3b"])

    def test_each_recommended_nvidia_model_uses_its_vision_request_options(self):
        for identifier in DEFAULT_MODELS["NVIDIA"]:
            with self.subTest(model=identifier):
                def handler(request):
                    payload = json.loads(request.content)
                    self.assertEqual(payload["model"], identifier)
                    self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")
                    self.assertEqual(payload["max_tokens"], 1600)
                    if identifier.startswith("qwen/"):
                        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                    else:
                        self.assertIn("reasoning_effort", payload)
                        self.assertIsNone(payload["reasoning_effort"])
                    return reply()
                result = Analyzer("NVIDIA", identifier, "test-placeholder", transport=httpx.MockTransport(handler)).analyze(
                    self.photo, threading.Event())
                self.assertEqual(result["gender"], "남성")

    def test_legacy_saved_selection_migrates_but_explicit_model_is_not_blocked(self):
        legacy = "nvidia/nemotron-nano-12b-v2-vl"
        requests = []
        def handler(request):
            requests.append(json.loads(request.content)["model"])
            return reply()
        analyzer = Analyzer("NVIDIA", legacy, "test-placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(analyzer.analyze(self.photo, threading.Event())["gender"], "남성")
        self.assertEqual(requests, [legacy])
        self.assertEqual(restore_model("NVIDIA", legacy), DEFAULT_MODELS["NVIDIA"][0])
        for identifier in DEFAULT_MODELS["NVIDIA"]:
            self.assertEqual(model_id("NVIDIA", model_label("NVIDIA", identifier)), identifier)
        self.assertEqual(restore_model("OpenAI", "custom-vision-id"), "custom-vision-id")

    def test_error_retains_server_reason_and_request_id_without_echoed_secrets(self):
        key = "nvapi-dummy-private-value"
        other_key = "sk-other-dummy-secret"
        uri = "data:image/jpeg;base64," + "A" * 200
        def handler(request):
            return httpx.Response(410, json={"error": {"message":
                f"Account route disabled. key={key}; other={other_key}; image={uri}"},
                "request": {"Authorization": key, "messages": "private photo data"}},
                headers={"x-request-id": "trace-test"})
        with self.assertRaises(APIError) as caught:
            Analyzer("NVIDIA", "test-model", key, transport=httpx.MockTransport(handler)).analyze(
                self.photo, threading.Event())
        message = str(caught.exception)
        self.assertTrue(caught.exception.fatal)
        self.assertEqual(caught.exception.status_code, 410)
        self.assertIn("Account route disabled", message)
        self.assertIn("POST https://integrate.api.nvidia.com/v1/chat/completions", message)
        self.assertIn("trace-test", message)
        self.assertIn("단정할 수 없습니다", message)
        for private in (key, other_key, uri, "private photo data", "A" * 100):
            self.assertNotIn(private, message)

    def test_plain_text_and_html_errors_are_bounded_and_redacted(self):
        key = "dummy-value-without-standard-prefix"
        for content in (f"Gateway gone Bearer {key} data:image/jpeg;base64,AAAA " + "x" * 3000,
                        f"<html><script>secret script</script><body>Gateway gone {key}</body></html>"):
            with self.subTest(content=content[:30]):
                def handler(request):
                    return httpx.Response(410, text=content)
                with self.assertRaises(APIError) as caught:
                    Analyzer("NVIDIA", "test-model", key, transport=httpx.MockTransport(handler)).analyze(
                        self.photo, threading.Event())
                message = str(caught.exception)
                self.assertIn("Gateway gone", message)
                self.assertNotIn(key, message)
                self.assertNotIn("secret script", message)
                self.assertNotIn("data:image", message)
                self.assertLess(len(message), 1500)

    def test_wrapped_base64_error_is_hidden_in_copyable_diagnostic(self):
        wrapped = "\n".join(["A" * 76, "B" * 76, "C" * 76])
        def handler(request):
            return httpx.Response(410, json={"error": {
                "message": 'Image rejected: "data:image/jpeg;base64,\n' + wrapped + '" route disabled'}})
        report = diagnose_connection("NVIDIA", "test-model", "test-placeholder",
                                     transport=httpx.MockTransport(handler))
        self.assertIn("Image rejected", report)
        self.assertIn("route disabled", report)
        self.assertIn("이미지 데이터 숨김", report)
        for chunk in ("A" * 76, "B" * 76, "C" * 76, "data:image"):
            self.assertNotIn(chunk, report)

    def test_diagnostic_compares_catalog_and_inference_without_photo_data(self):
        key = "nvapi-dummy-diagnostic-key"
        model = DEFAULT_MODELS["NVIDIA"][0]
        requests = []
        def handler(request):
            requests.append(request)
            self.assertEqual(request.headers["Authorization"], "Bearer " + key)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model}]})
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], model)
            self.assertEqual(payload["max_tokens"], 32)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertIsInstance(payload["messages"][0]["content"], str)
            self.assertNotIn("image_url", request.content.decode())
            self.assertNotIn(str(self.photo), request.content.decode())
            return httpx.Response(410, json={"error": {"message": "Inference route not available"}})
        report = diagnose_connection("NVIDIA", model, key, transport=httpx.MockTransport(handler))
        self.assertEqual([r.method for r in requests], ["GET", "POST"])
        self.assertIn("HTTP 200", report)
        self.assertIn("목록에 있습니다", report)
        self.assertIn("410", report)
        self.assertIn("Inference route not available", report)
        self.assertNotIn(key, report)

    def test_diagnostic_checks_inference_even_when_catalog_is_gone(self):
        for inference_status in (200, 410):
            with self.subTest(inference_status=inference_status):
                calls = []
                def handler(request):
                    calls.append(request.method)
                    if request.method == "GET":
                        return httpx.Response(410, text="Catalog unavailable")
                    return reply(status=inference_status)
                report = diagnose_connection("NVIDIA", "custom-model", "test-placeholder",
                                             transport=httpx.MockTransport(handler))
                self.assertEqual(calls, ["GET", "POST"])
                self.assertIn("Catalog unavailable", report)
                if inference_status == 200:
                    self.assertIn("텍스트 응답을 받았습니다", report)
                    self.assertIn("별도 확인", report)
                else:
                    self.assertEqual(report.count("(410)"), 2)

    def test_diagnostic_does_not_infer_after_auth_or_quota_failure(self):
        for status in (401, 402, 403, 429):
            with self.subTest(status=status):
                requests = []
                def handler(request):
                    requests.append(request.method)
                    return httpx.Response(status, json={"error": "Account restricted"},
                                          headers={"Retry-After": "0"})
                report = diagnose_connection("NVIDIA", "test-model", "test-placeholder",
                                             transport=httpx.MockTransport(handler))
                self.assertEqual(set(requests), {"GET"})
                self.assertIn("생략", report)
                self.assertIn("Account restricted", report)

    def test_diagnostic_handles_malformed_catalog_but_tests_inference(self):
        def handler(request):
            return httpx.Response(200, json={"data": None}) if request.method == "GET" else reply()
        report = diagnose_connection("NVIDIA", "test-model", "test-placeholder",
                                     transport=httpx.MockTransport(handler))
        self.assertIn("모델 목록 형식", report)
        self.assertIn("텍스트 응답을 받았습니다", report)

    def test_cancellation_interrupts_diagnostic_and_skips_second_route(self):
        entered = threading.Event()
        cancel = threading.Event()
        calls = []
        async def handler(request):
            calls.append(request.method)
            entered.set()
            await asyncio.sleep(10)
            return httpx.Response(200, json={"data": []})
        def cancel_soon():
            entered.wait(1)
            cancel.set()
        thread = threading.Thread(target=cancel_soon, daemon=True)
        thread.start()
        start = time.monotonic()
        with self.assertRaises(Cancelled):
            diagnose_connection("NVIDIA", "test-model", "test-placeholder", cancel,
                                transport=httpx.MockTransport(handler))
        thread.join(1)
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual(calls, ["GET"])


if __name__ == "__main__":
    unittest.main()
