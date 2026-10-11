"""Photo-model presets; account discovery remains authoritative for access.

Vision types and request options were checked against the NVIDIA-maintained
LangChain integration registry on 2026-10-11:
https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/langchain_nvidia_ai_endpoints/_statics.py
Nemotron Omni's image support is documented in NVIDIA's developer repository:
https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/omni3/inference.md
"""

NVIDIA_PRESETS = {
    "qwen/qwen3.5-122b-a10b": "Qwen 3.5 122B A10B · 기본 추천",
    "qwen/qwen3.5-397b-a17b": "Qwen 3.5 397B A17B · 큰 모델",
    "mistralai/mistral-small-4-119b-2603": "Mistral Small 4 119B · 대안",
}
# Show Omni only when this exact hosted ID is returned by the account catalog.
NVIDIA_DISCOVERABLE = {
    **NVIDIA_PRESETS,
    "nvidia/nemotron-3-nano-omni-30b-a3b": "Nemotron 3 Nano Omni 30B · 계정 목록 확인",
}
MODEL_LABELS = {
    "NVIDIA": NVIDIA_DISCOVERABLE,
    "OpenAI": {"gpt-4.1-mini": "GPT-4.1 mini · 사진 분석"},
}
DEFAULT_MODELS = {"NVIDIA": list(NVIDIA_PRESETS), "OpenAI": ["gpt-4.1-mini"]}
LEGACY_NVIDIA_MODELS = {"nvidia/nemotron-nano-12b-v2-vl"}


def model_label(provider: str, model_id: str) -> str:
    return MODEL_LABELS.get(provider, {}).get(model_id, model_id)


def model_id(provider: str, selection: str) -> str:
    selection = selection.strip()
    return next((identifier for identifier, label in MODEL_LABELS.get(provider, {}).items()
                 if label == selection), selection)


def restore_model(provider: str, saved: str) -> str:
    identifier = model_id(provider, saved) if isinstance(saved, str) else ""
    if not identifier or (provider == "NVIDIA" and identifier in LEGACY_NVIDIA_MODELS):
        return DEFAULT_MODELS[provider][0]
    return identifier


def photo_models(provider: str, available: list[str]) -> list[str]:
    """Never treat a text-only NVIDIA model as an image-analysis choice."""
    if provider == "NVIDIA":
        identifiers = set(available)
        return [identifier for identifier in NVIDIA_DISCOVERABLE if identifier in identifiers]
    return available
