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

from frametag.api import Analyzer, APIError, Cancelled, _retry_delay, list_models


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
        for status in (401, 402, 403, 404):
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
            self.assertEqual(str(request.url), "https://integrate.api.nvidia.com/v1/models")
            return httpx.Response(200, json={"data": [
                {"id": "z"}, {"id": "a"}, {"id": "a"}, {"id": None},
            ]})

        self.assertEqual(list_models("nvidia", "test-placeholder",
                                     transport=httpx.MockTransport(handler)), ["a", "z"])


if __name__ == "__main__":
    unittest.main()
