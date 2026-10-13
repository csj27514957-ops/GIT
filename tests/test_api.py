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

from frametag.api import (Analyzer, APIError, Cancelled, _request, _retry_delay,
                          diagnose_connection, discover_photo_models, list_models)
from frametag.model_catalog import NVIDIA_DISCOVERABLE, model_id, model_label, restore_model


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


def challenge_reply(request):
    """The fake model must inspect actual pixels rather than echoing prompt JSON."""
    payload = json.loads(request.content)
    content = payload["messages"][0]["content"]
    uri = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
    palette = {"red": (255, 0, 0), "green": (0, 255, 0),
               "blue": (0, 0, 255), "yellow": (255, 255, 0)}
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
        colors = [image.getpixel((64, 64)), image.getpixel((192, 64))]
    names = [min(palette, key=lambda name: sum((a - b) ** 2
                 for a, b in zip(color, palette[name]))) for color in colors]
    return reply(json.dumps(dict(zip(("left", "right"), names))))


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
                         ["nvidia/nemotron-nano-12b-v2-vl", "qwen/qwen3.5-397b-a17b",
                          "nvidia/nemotron-3-nano-omni-30b-a3b"])

    def test_each_recommended_nvidia_model_uses_its_vision_request_options(self):
        for identifier in NVIDIA_DISCOVERABLE:
            with self.subTest(model=identifier):
                def handler(request):
                    payload = json.loads(request.content)
                    self.assertEqual(payload["model"], identifier)
                    content = next(message["content"] for message in payload["messages"]
                                   if message["role"] == "user")
                    self.assertTrue(any(part["type"] == "image_url" for part in content))
                    self.assertEqual(payload["max_tokens"], 1600)
                    if identifier == "qwen/qwen3.5-397b-a17b":
                        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                    elif identifier == "mistralai/mistral-small-4-119b-2603":
                        self.assertIn("reasoning_effort", payload)
                        self.assertIsNone(payload["reasoning_effort"])
                    elif identifier == "moonshotai/kimi-k2.6":
                        self.assertEqual(payload["chat_template_kwargs"], {"thinking": False})
                    elif (identifier == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" or
                          identifier.startswith("google/gemma-4-")):
                        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                    if identifier == "nvidia/nemotron-nano-12b-v2-vl":
                        self.assertTrue(next(part["text"] for part in content
                                             if part["type"] == "text").startswith("/no_think"))
                    return reply()
                result = Analyzer("NVIDIA", identifier, "test-placeholder", transport=httpx.MockTransport(handler)).analyze(
                    self.photo, threading.Event())
                self.assertEqual(result["gender"], "남성")

    def test_confirmed_retired_selection_clears_but_other_explicit_model_is_preserved(self):
        legacy = "nvidia/nemotron-nano-12b-v2-vl"
        requests = []
        def handler(request):
            requests.append(json.loads(request.content)["model"])
            return reply()
        analyzer = Analyzer("NVIDIA", legacy, "test-placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(analyzer.analyze(self.photo, threading.Event())["gender"], "남성")
        self.assertEqual(requests, [legacy])
        self.assertEqual(restore_model("NVIDIA", legacy), legacy)
        self.assertEqual(restore_model("NVIDIA", "qwen/qwen3.5-122b-a10b"), "")
        for identifier in NVIDIA_DISCOVERABLE:
            self.assertEqual(model_id("NVIDIA", model_label("NVIDIA", identifier)), identifier)
        self.assertEqual(restore_model("OpenAI", "custom-vision-id"), "custom-vision-id")

    def test_confirmed_builtin_retired_model_never_reads_or_uploads_a_photo(self):
        model = "qwen/qwen3.5-122b-a10b"
        analyzer = Analyzer("NVIDIA", model, "placeholder", transport=httpx.MockTransport(
            lambda request: self.fail("A confirmed retired model must not send a request")))
        with patch("frametag.api._image_url", side_effect=AssertionError("User photo read")):
            with self.assertRaises(APIError) as caught:
                analyzer.analyze(Path("missing-user-photo.jpg"), threading.Event())
        self.assertTrue(caught.exception.fatal)
        self.assertEqual(caught.exception.model_retired, model)
        self.assertIsNone(caught.exception.status_code)

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
        model = "qwen/qwen3.5-397b-a17b"
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

    def test_explicit_model_retirement_requires_matching_requested_model(self):
        model = "vendor/test-vision"
        descriptions = [
            (f"The model '{model}' has reached its end of life on 2026-07-20 and is no longer available.", model),
            (f"The model '{model}' is no longer available.", model),
            ("The model 'different-model' has reached its end of life and is no longer available.", ""),
            (f"The model '{model}' is active. The model 'another-model' has reached its end of life.", ""),
            (f"Route gone for {model}; contact the account owner.", ""),
        ]
        for description, retired in descriptions:
            with self.subTest(description=description):
                def handler(request):
                    return httpx.Response(410, json={"error": {"message": description}})
                with self.assertRaises(APIError) as caught:
                    Analyzer("NVIDIA", model, "placeholder", transport=httpx.MockTransport(handler)).analyze(
                        self.photo, threading.Event())
                self.assertEqual(caught.exception.model_retired, retired)
                self.assertIn(description, str(caught.exception))

    def test_discovery_offers_only_actually_verified_images_and_never_opens_user_files(self):
        available = ["meta/llama-3.3-70b-instruct", "vendor/new-vision-v1",
                     "nvidia/nemotron-nano-12b-v2-vl"]
        requests, progress = [], []
        def handler(request):
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": name} for name in available]})
            payload = json.loads(request.content)
            self.assertEqual(payload["max_tokens"], 128)
            self.assertNotIn(str(self.photo), request.content.decode())
            self.assertNotIn("photo.png", request.content.decode())
            return challenge_reply(request)
        with patch("frametag.api._image_url", side_effect=AssertionError("User image read")):
            result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler),
                                           progress=lambda *values: progress.append(values))
        self.assertEqual(result.models, ["nvidia/nemotron-nano-12b-v2-vl", "vendor/new-vision-v1"])
        self.assertEqual(result.catalog, sorted(available))
        self.assertFalse(result.retired)
        self.assertEqual(progress, [(1, 2, result.models[0]), (2, 2, result.models[1])])
        self.assertEqual(len(requests), 3)

    def test_successful_text_response_without_correct_image_colors_is_not_verified(self):
        model = "vendor/new-vision-v1"
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model}]})
            return reply('{"left":"blue","right":"red"}')
        with patch("frametag.api.secrets.choice", side_effect=["red", "green"]):
            result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, [])
        self.assertEqual(len(result.notes), 1)
        self.assertIn("색상과 위치", result.notes[0])

    def test_discovery_skips_confirmed_and_saved_retired_ids_without_probing(self):
        model = "vendor/remaining-vision"
        retired = "qwen/qwen3.5-122b-a10b"
        excluded = "vendor/saved-retired-vision"
        probed = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": identifier}
                    for identifier in (model, retired, excluded)]})
            probed.append(json.loads(request.content)["model"])
            return challenge_reply(request)
        result = discover_photo_models("NVIDIA", "placeholder", preferred=retired, excluded=(excluded,),
                                       transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, [model])
        self.assertEqual(probed, [model])

    def test_discovery_handles_individual_retirement_and_unsupported_images(self):
        models = ["vendor/a-vision", "vendor/b-vision", "vendor/c-vision", "vendor/d-vision"]
        requests = []
        def handler(request):
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in models]})
            model = json.loads(request.content)["model"]
            if model == models[0]:
                return httpx.Response(410, json={"error": {"message":
                    f"The model '{model}' has reached its end of life and is no longer available."}})
            if model == models[1]:
                return httpx.Response(400, json={"error": "Image input is not supported"})
            if model == models[2]:
                return httpx.Response(404, json={"error": "Model not found"})
            return challenge_reply(request)
        result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, [models[3]])
        self.assertEqual(result.retired, [models[0]])
        self.assertEqual(len(result.notes), 3)
        self.assertEqual(len(requests), 5)

    def test_auth_and_quota_abort_discovery_without_trying_other_models(self):
        models = ["vendor/a-vision", "vendor/b-vision"]
        for status in (401, 402, 403, 429):
            with self.subTest(status=status):
                calls = []
                def handler(request):
                    calls.append(request.method)
                    if request.method == "GET":
                        return httpx.Response(200, json={"data": [{"id": model} for model in models]})
                    return httpx.Response(status, json={"error": "Account route unavailable"})
                with self.assertRaises(APIError) as caught:
                    discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
                self.assertEqual(calls, ["GET", "POST"])
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.model_retired, "")
                self.assertEqual(caught.exception.retired_models, [])

    def test_later_global_discovery_error_preserves_previously_confirmed_retirement(self):
        models = ["vendor/a-vision", "vendor/b-vision"]
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in models]})
            model = json.loads(request.content)["model"]
            if model == models[0]:
                return httpx.Response(410, json={"error": f"The model '{model}' is no longer available."})
            return httpx.Response(403, json={"error": "Account restricted"})
        with self.assertRaises(APIError) as caught:
            discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(caught.exception.retired_models, [models[0]])

    def test_discovery_limit_and_preferred_catalog_model(self):
        models = [f"vendor/test-{index:02d}-vision" for index in range(12)]
        preferred = "vendor/custom-vision"
        probed = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in [*models, preferred]]})
            probed.append(json.loads(request.content)["model"])
            return reply("not a color response")
        result = discover_photo_models("NVIDIA", "placeholder", preferred=preferred,
                                       transport=httpx.MockTransport(handler))
        self.assertEqual(probed, [preferred, *models[:7]])
        self.assertEqual(result.models, [])
        self.assertEqual(len(result.catalog), 13)

    def test_discovery_stops_after_three_verified_models(self):
        models = [f"vendor/test-{index:02d}-vision" for index in range(8)]
        probed = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in models]})
            probed.append(json.loads(request.content)["model"])
            return challenge_reply(request)
        result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, models[:3])
        self.assertEqual(probed, models[:3])

    def test_discovery_network_failure_is_not_retried(self):
        calls = []
        def handler(request):
            calls.append(request.method)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "vendor/test-vision"}]})
            raise httpx.ConnectError("unavailable", request=request)
        with self.assertRaises(APIError):
            discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(calls, ["GET", "POST"])

    def test_model_http_rejections_preserve_verified_models_and_continue(self):
        models = ["vendor/a-vision", "vendor/b-vision", "vendor/c-vision"]
        for status in (400, 404, 410, 422, 500, 503):
            with self.subTest(status=status):
                probed = []
                def handler(request):
                    if request.method == "GET":
                        return httpx.Response(200, json={"data": [{"id": model} for model in models]})
                    model = json.loads(request.content)["model"]
                    probed.append(model)
                    if model == models[1]:
                        return httpx.Response(status, json={"error": "Candidate unavailable"})
                    return challenge_reply(request)
                result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
                self.assertEqual(result.models, [models[0], models[2]])
                self.assertEqual(probed, models)
                self.assertEqual(result.retired, [])
                self.assertEqual(len(result.notes), 1)
                self.assertIn("Candidate unavailable", result.notes[0])

    def test_all_model_http_errors_return_unverified_notes_without_retiring_generic_410(self):
        models = ["vendor/a-vision", "vendor/b-vision"]
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in models]})
            return httpx.Response(410, json={"error": "Route unavailable"})
        result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, [])
        self.assertEqual(result.retired, [])
        self.assertEqual(len(result.notes), 2)

    def test_transport_failure_returns_prior_successes_and_stops_remaining_probes(self):
        models = ["vendor/a-vision", "vendor/b-vision", "vendor/c-vision"]
        probed = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in models]})
            model = json.loads(request.content)["model"]
            probed.append(model)
            if model == models[1]:
                raise httpx.ConnectError("unavailable", request=request)
            return challenge_reply(request)
        result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, [models[0]])
        self.assertEqual(probed, models[:2])
        self.assertEqual(result.retired, [])
        self.assertEqual(len(result.notes), 1)
        self.assertIn("중단", result.notes[0])

    def test_account_failure_after_success_still_aborts_discovery(self):
        models = ["vendor/a-vision", "vendor/b-vision"]
        for status in (401, 402, 403, 429):
            with self.subTest(status=status):
                def handler(request):
                    if request.method == "GET":
                        return httpx.Response(200, json={"data": [{"id": model} for model in models]})
                    if json.loads(request.content)["model"] == models[0]:
                        return challenge_reply(request)
                    return httpx.Response(status, json={"error": "Account restricted"})
                with self.assertRaises(APIError) as caught:
                    discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
                self.assertEqual(caught.exception.status_code, status)

    def test_preferred_embedding_or_incompatible_image_model_cannot_bypass_filter(self):
        valid = "vendor/new-vision"
        excluded_models = ["nvidia/nv-embed-vl", "vendor/vision-embedding-v1",
                           "microsoft/phi-3-vision-128k-instruct", "vendor/plain-text"]
        for preferred in excluded_models:
            with self.subTest(preferred=preferred):
                probed = []
                def handler(request):
                    if request.method == "GET":
                        return httpx.Response(200, json={"data": [{"id": model}
                            for model in [valid, *excluded_models]]})
                    probed.append(json.loads(request.content)["model"])
                    return challenge_reply(request)
                result = discover_photo_models("NVIDIA", "placeholder", preferred=preferred,
                                               transport=httpx.MockTransport(handler))
                self.assertEqual(result.models, [valid])
                self.assertEqual(probed, [valid])

    def test_llama_vision_fixed_routes_are_shared_by_analysis_discovery_and_diagnostic(self):
        for size in ("11b", "90b"):
            with self.subTest(size=size):
                model = f"meta/llama-3.2-{size}-vision-instruct"
                endpoint = f"https://ai.api.nvidia.com/v1/gr/{model}/chat/completions"
                calls = []
                def handler(request):
                    calls.append(str(request.url))
                    if request.method == "GET":
                        self.assertEqual(str(request.url), "https://integrate.api.nvidia.com/v1/models")
                        return httpx.Response(200, json={"data": [{"id": model}]})
                    self.assertEqual(str(request.url), endpoint)
                    payload = json.loads(request.content)
                    self.assertEqual(payload["model"], model)
                    if payload["max_tokens"] in (128, 1600):
                        user = next(message for message in payload["messages"] if message["role"] == "user")
                        self.assertEqual([part["type"] for part in user["content"]], ["image_url", "text"])
                        self.assertEqual([message["role"] for message in payload["messages"]], ["user"])
                    if payload["max_tokens"] == 128:
                        return challenge_reply(request)
                    if payload["max_tokens"] == 1600:
                        self.assertIn("주된 촬영 인물", payload["messages"][0]["content"][1]["text"])
                    return reply()
                transport = httpx.MockTransport(handler)
                result = Analyzer("NVIDIA", model, "placeholder", transport=transport).analyze(
                    self.photo, threading.Event())
                self.assertEqual(result["gender"], "남성")
                discovery = discover_photo_models("NVIDIA", "placeholder", transport=transport)
                self.assertEqual(discovery.models, [model])
                report = diagnose_connection("NVIDIA", model, "placeholder", transport=transport)
                self.assertIn("2. 분석 경로: POST " + endpoint, report)
                self.assertIn("텍스트 응답을 받았습니다", report)
                self.assertEqual(calls.count(endpoint), 3)

    def test_model_string_cannot_set_request_destination(self):
        model = "https://untrusted.invalid/steal-key"
        def handler(request):
            self.assertEqual(str(request.url), "https://integrate.api.nvidia.com/v1/chat/completions")
            return reply()
        Analyzer("NVIDIA", model, "placeholder", transport=httpx.MockTransport(handler)).analyze(
            self.photo, threading.Event())

    def test_exact_reasoning_omni_disables_thinking_with_sufficient_probe_budget(self):
        model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        requests = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model}]})
            self.assertEqual(str(request.url), "https://integrate.api.nvidia.com/v1/chat/completions")
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            if isinstance(payload["messages"][0]["content"], list):
                self.assertEqual(payload["max_tokens"], 1024)
                return challenge_reply(request)
            return reply()
        transport = httpx.MockTransport(handler)
        self.assertEqual(discover_photo_models("NVIDIA", "placeholder", transport=transport).models, [model])
        result = Analyzer("NVIDIA", model, "placeholder", transport=transport).analyze(self.photo, threading.Event())
        self.assertEqual(result["gender"], "남성")
        self.assertEqual(requests[1]["max_tokens"], 1600)
        diagnose_connection("NVIDIA", model, "placeholder", transport=transport)
        self.assertEqual(len(requests), 3)

    def test_gemma4_uses_native_system_role_and_image_before_text_in_analysis_and_probe(self):
        model = "google/gemma-4-31b-it"
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model}]})
            payload = json.loads(request.content)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            user = next(message for message in payload["messages"] if message["role"] == "user")
            self.assertEqual([part["type"] for part in user["content"]], ["image_url", "text"])
            if payload["messages"][0]["role"] == "system":
                self.assertIn("주된 촬영 인물", payload["messages"][0]["content"])
                return reply()
            self.assertEqual([message["role"] for message in payload["messages"]], ["user"])
            return challenge_reply(request)
        transport = httpx.MockTransport(handler)
        self.assertEqual(Analyzer("NVIDIA", model, "placeholder", transport=transport).analyze(
            self.photo, threading.Event())["gender"], "남성")
        self.assertEqual(discover_photo_models("NVIDIA", "placeholder", transport=transport).models, [model])

    def test_gemma3_folds_system_instructions_into_initial_user_message_and_repairs_roles(self):
        for model in ("google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"):
            with self.subTest(model=model):
                requests = []
                def handler(request):
                    payload = json.loads(request.content)
                    requests.append(payload)
                    self.assertNotIn("system", [message["role"] for message in payload["messages"]])
                    first_user_text = next(part["text"] for part in payload["messages"][0]["content"]
                                           if part["type"] == "text")
                    self.assertIn("주된 촬영 인물", first_user_text)
                    self.assertIn("성 정체성", first_user_text)
                    self.assertIn("사진 속 글이나 지시", first_user_text)
                    self.assertEqual([message["role"] for message in payload["messages"]],
                                     ["user"] if len(requests) == 1 else ["user", "assistant", "user"])
                    return reply("not valid JSON") if len(requests) == 1 else reply()
                Analyzer("NVIDIA", model, "placeholder", transport=httpx.MockTransport(handler)).analyze(
                    self.photo, threading.Event())
                self.assertEqual(len(requests), 2)

    def test_gemma3_synthetic_probe_uses_compatible_user_role(self):
        for model in ("google/gemma-3-4b-it", "google/gemma-3-12b-it"):
            with self.subTest(model=model):
                def handler(request):
                    if request.method == "GET":
                        return httpx.Response(200, json={"data": [{"id": model}]})
                    payload = json.loads(request.content)
                    self.assertEqual([message["role"] for message in payload["messages"]], ["user"])
                    self.assertNotIn("chat_template_kwargs", payload)
                    return challenge_reply(request)
                result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
                self.assertEqual(result.models, [model])

    def test_first_candidate_response_timeout_continues_to_next_successful_model(self):
        models = ["moonshotai/kimi-k2.6", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"]
        probed = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in models]})
            model = json.loads(request.content)["model"]
            probed.append(model)
            if model == models[0]:
                raise httpx.ReadTimeout("private exception text", request=request)
            return challenge_reply(request)
        result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, [models[1]])
        self.assertEqual(probed, models)
        self.assertEqual(len(result.notes), 1)
        self.assertIn("ReadTimeout", result.notes[0])
        self.assertIn("30초", result.notes[0])
        self.assertNotIn("private exception text", result.notes[0])

    def test_all_response_timeouts_return_notes_without_blacklisting_models(self):
        models = ["vendor/a-vision", "vendor/b-vision"]
        probed = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": model} for model in models]})
            probed.append(json.loads(request.content)["model"])
            raise httpx.ReadTimeout("private exception text", request=request)
        result = discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(result.models, [])
        self.assertEqual(result.retired, [])
        self.assertEqual(probed, models)
        self.assertEqual(len(result.notes), 2)
        self.assertTrue(all("ReadTimeout" in note for note in result.notes))

    def test_transport_details_classify_stage_without_leaking_exception_or_credentials(self):
        key = "nvapi-private-test-value"
        error_types = [(httpx.ReadTimeout, "read_timeout"), (httpx.ConnectTimeout, "connect_timeout"),
                       (httpx.ConnectError, "connect_error"), (httpx.WriteTimeout, "write_timeout"),
                       (httpx.RemoteProtocolError, "transport_error")]
        for error_class, expected_kind in error_types:
            with self.subTest(error_class=error_class.__name__):
                def handler(request):
                    raise error_class(f"private path photo.png Bearer {key}", request=request)
                with self.assertRaises(APIError) as caught:
                    _request("POST", "https://integrate.api.nvidia.com/v1/chat/completions", key,
                             threading.Event(), {"model": "vendor/test-vision"},
                             transport=httpx.MockTransport(handler), attempts=1, timeout_seconds=30)
                self.assertEqual(caught.exception.transport_kind, expected_kind)
                message = str(caught.exception)
                self.assertIn("POST https://integrate.api.nvidia.com/v1/chat/completions", message)
                self.assertIn(error_class.__name__, message)
                self.assertNotIn(key, message)
                self.assertNotIn("photo.png", message)
                self.assertNotIn("private path", message)

    def test_connection_and_write_errors_without_prior_success_stop_discovery(self):
        models = ["vendor/a-vision", "vendor/b-vision"]
        for error_class in (httpx.ConnectError, httpx.ConnectTimeout, httpx.WriteTimeout,
                            httpx.RemoteProtocolError):
            with self.subTest(error_class=error_class.__name__):
                probed = []
                def handler(request):
                    if request.method == "GET":
                        return httpx.Response(200, json={"data": [{"id": model} for model in models]})
                    probed.append(json.loads(request.content)["model"])
                    raise error_class("unavailable", request=request)
                with self.assertRaises(APIError):
                    discover_photo_models("NVIDIA", "placeholder", transport=httpx.MockTransport(handler))
                self.assertEqual(probed, models[:1])

    def test_cancellation_takes_precedence_over_simultaneous_response_timeout(self):
        cancel = threading.Event()
        def handler(request):
            cancel.set()
            raise httpx.ReadTimeout("deadline", request=request)
        with self.assertRaises(Cancelled):
            _request("POST", "https://integrate.api.nvidia.com/v1/chat/completions", "placeholder",
                     cancel, transport=httpx.MockTransport(handler), attempts=1)

    def test_cancellation_interrupts_model_image_probe(self):
        started, ended, cancel = threading.Event(), threading.Event(), threading.Event()
        results, calls = [], []
        async def handler(request):
            calls.append(request.method)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "vendor/test-vision"}]})
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                ended.set()
        def run():
            try:
                results.append(discover_photo_models("NVIDIA", "placeholder", cancel=cancel,
                                                     transport=httpx.MockTransport(handler)))
            except Exception as error:
                results.append(error)
        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(started.wait(1))
        cancel.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(ended.is_set())
        self.assertIsInstance(results[0], Cancelled)
        self.assertEqual(calls, ["GET", "POST"])

    def test_single_request_has_total_timeout_and_cancels_transport(self):
        ended = threading.Event()
        async def handler(request):
            try:
                await asyncio.sleep(10)
            finally:
                ended.set()
        started = time.monotonic()
        with self.assertRaises(APIError) as caught:
            _request("GET", "https://test.invalid/models", "placeholder", threading.Event(),
                     transport=httpx.MockTransport(handler), attempts=1, timeout_seconds=0.03)
        self.assertEqual(caught.exception.transport_kind, "read_timeout")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(ended.is_set())

    def test_catalog_only_diagnostic_preserves_raw_ids_without_inference(self):
        models = ["vendor/text-only", "vendor/new-vision"]
        calls = []
        def handler(request):
            calls.append(request.method)
            return httpx.Response(200, json={"data": [{"id": model} for model in models]})
        report = diagnose_connection("NVIDIA", "", "placeholder", transport=httpx.MockTransport(handler))
        self.assertEqual(calls, ["GET"])
        self.assertIn("서버 모델 수: 2", report)
        self.assertIn("vendor/text-only", report)
        self.assertIn("vendor/new-vision", report)
        self.assertIn("미선택", report)
        self.assertIn("생략", report)


    def test_overload_and_degraded_model_are_reported_as_server_busy(self):
        overloaded = {"error": "ResourceExhausted: Worker local total request limit reached (16/16)"}
        degraded = {"error": "Function id 'x': DEGRADED function cannot be invoked"}
        cases = [(lambda request: httpx.Response(503, json=overloaded), True, 4),
                 (lambda request: httpx.Response(400, json=degraded), True, 1),
                 (lambda request: httpx.Response(400, json={"error": "bad image"}), False, 1)]
        for respond, busy, attempts in cases:
            with self.subTest(busy=busy, attempts=attempts):
                calls = []

                def handler(request, respond=respond):
                    calls.append(request)
                    return respond(request)

                with patch("frametag.api._retry_delay", return_value=0):
                    with self.assertRaises(APIError) as caught:
                        self.analyzer(handler).analyze(self.photo, threading.Event())
                self.assertEqual(caught.exception.server_busy, busy)
                self.assertFalse(caught.exception.fatal)
                self.assertEqual(len(calls), attempts)

    def test_prompt_records_worn_extras_as_objects_without_location(self):
        from frametag.api import PROMPT
        self.assertIn("모자, 목도리", PROMPT)
        self.assertIn("clothing이 아니라 objects", PROMPT)
        self.assertIn("위치(손, 바닥 등)를 쓰지 말고", PROMPT)
        self.assertNotIn('"location"', PROMPT)

if __name__ == "__main__":
    unittest.main()
