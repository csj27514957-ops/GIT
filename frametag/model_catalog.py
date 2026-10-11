"""Photo-model presets; account discovery remains authoritative for access.

Vision types and request options were checked against the NVIDIA-maintained
LangChain integration registry on 2026-10-11:
https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/langchain_nvidia_ai_endpoints/_statics.py
Exact hosted IDs were checked against the user's NVIDIA /models response.
Native image support and Omni's reasoning settings are documented by the makers:
https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
https://huggingface.co/google/gemma-4-31B-it
"""

import re

NVIDIA_PRESETS = {
    "moonshotai/kimi-k2.6": "Kimi K2.6",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": "Nemotron 3 Nano Omni 30B",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "google/gemma-3-12b-it": "Gemma 3 12B",
    "google/gemma-3-4b-it": "Gemma 3 4B",
    "meta/llama-3.2-11b-vision-instruct": "Llama 3.2 Vision 11B",
    "meta/llama-3.2-90b-vision-instruct": "Llama 3.2 Vision 90B",
    "stepfun-ai/step-3.7-flash": "Step 3.7 Flash",
    "mistralai/ministral-3-14b-instruct-2512": "Ministral 3 14B",
    "mistralai/ministral-14b-instruct-2512": "Ministral 14B",
    "meta/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick",
    "nvidia/nemotron-nano-12b-v2-vl": "Nemotron Nano 12B V2 VL",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": "Nemotron Nano VL 8B",
    "mistralai/mistral-large-3-675b-instruct-2512": "Mistral Large 3",
    "qwen/qwen3.5-397b-a17b": "Qwen 3.5 397B A17B",
    "mistralai/mistral-small-4-119b-2603": "Mistral Small 4 119B",
}
# These are discovery candidates, never a claim of current hosted availability.
NVIDIA_DISCOVERABLE = NVIDIA_PRESETS
# Explicit NVIDIA error supplied by the user on 2026-10-11, EOL 2026-07-20 UTC.
RETIRED_NVIDIA_MODELS = {"qwen/qwen3.5-122b-a10b"}
# Phi-3 Vision uses NVIDIA's separate nv-vlm payload, which this adapter does
# not implement. A vision name alone must not route it as OpenAI-style chat.
UNSUPPORTED_NVIDIA_MODELS = {"microsoft/phi-3-vision-128k-instruct"}
NVIDIA_UNVERIFIED_LABEL = "사진 모델 확인 필요"
MODEL_LABELS = {
    "NVIDIA": NVIDIA_DISCOVERABLE,
    "OpenAI": {"gpt-4.1-mini": "GPT-4.1 mini · 사진 분석"},
}
DEFAULT_MODELS = {"NVIDIA": [], "OpenAI": ["gpt-4.1-mini"]}


def model_label(provider: str, model_id: str) -> str:
    if provider == "NVIDIA" and not model_id:
        return NVIDIA_UNVERIFIED_LABEL
    return MODEL_LABELS.get(provider, {}).get(model_id, model_id)


def model_id(provider: str, selection: str) -> str:
    selection = selection.strip()
    if provider == "NVIDIA" and selection == NVIDIA_UNVERIFIED_LABEL:
        return ""
    return next((identifier for identifier, label in MODEL_LABELS.get(provider, {}).items()
                 if label == selection), selection)


def restore_model(provider: str, saved: str) -> str:
    identifier = model_id(provider, saved) if isinstance(saved, str) else ""
    if not identifier or (provider == "NVIDIA" and identifier in RETIRED_NVIDIA_MODELS):
        return next(iter(DEFAULT_MODELS[provider]), "")
    return identifier


def photo_models(provider: str, available: list[str]) -> list[str]:
    """Find candidate IDs; live image verification decides which can be offered."""
    if provider == "NVIDIA":
        identifiers = {
            identifier for identifier in available
            if identifier not in RETIRED_NVIDIA_MODELS | UNSUPPORTED_NVIDIA_MODELS
            and not any(word in identifier.casefold() for word in
                        ("embed", "retriev", "rerank", "guard", "safety", "parse", "detector", "nvclip"))
        }
        known = [identifier for identifier in NVIDIA_DISCOVERABLE if identifier in identifiers]
        # Versioned vision IDs absent from this static registry remain candidates
        # only; the API's synthetic image check decides whether to offer them.
        additional = sorted(identifier for identifier in identifiers - set(known)
                            if re.search(r"(?:^|[/_-])(?:vl\d*|vision|omni)(?:[-_/.]|$)", identifier, re.I))
        return known + additional
    return available
