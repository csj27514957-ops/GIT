import unittest

from frametag.model_catalog import model_id, model_label, photo_models, restore_model


# Actual NVIDIA catalog supplied by the user on 2026-10-11. Includes every
# image-looking entry and representative text, embedding and safety models.
ACTUAL_CATALOG = [
    "adept/fuyu-8b", "google/deplot", "google/diffusiongemma-26b-a4b-it",
    "google/gemma-3-12b-it", "google/gemma-3-4b-it", "google/gemma-4-31b-it",
    "meta/llama-3.2-11b-vision-instruct", "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-guard-4-12b", "microsoft/kosmos-2",
    "microsoft/phi-3-vision-128k-instruct", "moonshotai/kimi-k2.6", "moonshotai/kimi-k3",
    "nvidia/ai-synthetic-video-detector", "nvidia/cosmos-reason2-8b",
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
    "nvidia/llama-nemotron-embed-vl-1b-v2", "nvidia/nemotron-3-embed-1b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3.5-content-safety", "nvidia/nemotron-parse-2.0",
    "nvidia/neva-22b", "nvidia/nvclip", "nvidia/vila", "openai/gpt-oss-20b",
]


class ModelCatalogTests(unittest.TestCase):
    def test_actual_catalog_offers_supported_generation_candidates(self):
        self.assertEqual(photo_models("NVIDIA", ACTUAL_CATALOG), [
            "moonshotai/kimi-k2.6",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "google/gemma-4-31b-it", "google/gemma-3-12b-it", "google/gemma-3-4b-it",
            "meta/llama-3.2-11b-vision-instruct", "meta/llama-3.2-90b-vision-instruct",
        ])

    def test_vision_names_do_not_admit_non_generation_models(self):
        self.assertEqual(photo_models("NVIDIA", [
            "vendor/image-vision-embed", "vendor/vision-retriever", "vendor/vl-reranker",
            "vendor/vision-guard", "vendor/omni-safety", "vendor/vision-parse",
            "vendor/vision-detector", "nvidia/llama-nemotron-embed-vl-1b-v2",
        ]), [])

    def test_new_vision_candidates_still_require_server_membership(self):
        self.assertEqual(photo_models("NVIDIA", ["vendor/new-vl-2", "vendor/new-text"]),
                         ["vendor/new-vl-2"])
        self.assertEqual(photo_models("NVIDIA", []), [])

    def test_exact_omni_selection_roundtrips_and_retired_model_is_cleared(self):
        identifier = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        self.assertEqual(model_id("NVIDIA", model_label("NVIDIA", identifier)), identifier)
        self.assertEqual(restore_model("NVIDIA", "qwen/qwen3.5-122b-a10b"), "")


if __name__ == "__main__":
    unittest.main()
