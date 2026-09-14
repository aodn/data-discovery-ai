import os
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.config.constants import (
    KEYWORD_FOLDER,
    KEYWORD_LABEL_FILE,
    STATUS_DOWN,
    STATUS_STARTING,
    STATUS_UP,
)


def _component(status: str, detail: str | None = None) -> Dict[str, Any]:
    return {"status": status, "detail": detail}


def check_keyword_resources() -> Dict[str, Any]:
    """
    Check the pretrained keyword classification model and its label file exist on disk.
    """
    config = ConfigUtil.get_config()
    base_path = config.base_dir / "resources"

    keyword_model = config.get_keyword_classification_config().pretrained_model
    keyword_model_path = (base_path / KEYWORD_FOLDER / keyword_model).with_suffix(
        ".keras"
    )
    keyword_label_path = base_path / KEYWORD_FOLDER / KEYWORD_LABEL_FILE

    missing = []
    if not keyword_model_path.exists():
        missing.append(f"Keyword model resource not found at {keyword_model_path}")
    if not keyword_label_path.exists():
        missing.append(f"Keyword label resource not found at {keyword_label_path}")
    if missing:
        return _component(STATUS_DOWN, "; ".join(missing))
    return _component(STATUS_UP)


async def check_llm() -> Dict[str, Any]:
    """
    Check the LLM backend is usable:
      - OpenAI API key is set in non-development environments.
      - Ollama server is running in development.
    """
    load_dotenv()
    env = os.getenv("PROFILE", "development")
    if env != "development":
        if not os.getenv("OPENAI_API_KEY"):
            return _component(STATUS_DOWN, "OpenAI API key not set")
        return _component(STATUS_UP)

    ollama_base_url = "http://localhost:11434"
    try:
        async with httpx.AsyncClient() as client:
            await client.get(f"{ollama_base_url}/models", timeout=2)
    except httpx.RequestError:
        return _component(STATUS_DOWN, "Ollama server not running")
    return _component(STATUS_UP)


def check_models(app: FastAPI) -> Dict[str, Any]:
    """
    Check the Hugging Face models loaded in the background by the server lifespan.
    Defaults to DOWN if the lifespan has not run (e.g. TestClient without context manager).
    """
    status = getattr(app.state, "model_status", STATUS_DOWN)
    return _component(status, getattr(app.state, "model_error", None))


async def collect_components(app: FastAPI) -> Dict[str, Dict[str, Any]]:
    return {
        "keyword_resources": check_keyword_resources(),
        "llm": await check_llm(),
        "models": check_models(app),
    }


def aggregate_status(components: Dict[str, Dict[str, Any]]) -> str:
    """
    UP if every component is UP, STARTING if the only non-UP components are still starting, DOWN otherwise.
    """
    statuses = [c["status"] for c in components.values()]
    if all(s == STATUS_UP for s in statuses):
        return STATUS_UP
    if all(s in (STATUS_UP, STATUS_STARTING) for s in statuses):
        return STATUS_STARTING
    return STATUS_DOWN
