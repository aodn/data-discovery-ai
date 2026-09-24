import asyncio
import json
import os
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI

from data_discovery_ai.config.config import ConfigUtil
from data_discovery_ai.config.constants import (
    HEALTH_FILE,
    KEYWORD_FOLDER,
    KEYWORD_LABEL_FILE,
    STATUS_DOWN,
    STATUS_STARTING,
    STATUS_UP,
)

import structlog


logger = structlog.get_logger(__name__)
_health_file_lock = asyncio.Lock()


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


def check_elasticsearch(app: FastAPI) -> Dict[str, Any]:
    """
    Check the Elasticsearch index set up in the background by the server lifespan.
    Defaults to DOWN if the lifespan has not run.
    """
    status = getattr(app.state, "es_status", STATUS_DOWN)
    return _component(status, getattr(app.state, "es_error", None))


def collect_startup_components(app: FastAPI) -> Dict[str, Dict[str, Any]]:
    """
    Components that only change when a startup task finishes (or never at runtime), so a snapshot of them cannot
    go stale between health file writes.
    """
    return {
        "keyword_resources": check_keyword_resources(),
        "models": check_models(app),
        "elasticsearch": check_elasticsearch(app),
    }


async def collect_components(app: FastAPI) -> Dict[str, Dict[str, Any]]:
    """All components, including the live LLM check."""
    startup = collect_startup_components(app)
    return {
        "keyword_resources": startup["keyword_resources"],
        "llm": await check_llm(),
        "models": startup["models"],
        "elasticsearch": startup["elasticsearch"],
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


async def build_health_payload(
    app: FastAPI, include_live: bool = True
) -> Dict[str, Any]:
    """
    Build the health response. The FastAPI route includes live checks; the Nginx status file passes
    include_live=False so it only holds startup state, as in data-access-service.
    """
    try:
        if include_live:
            components = await collect_components(app)
        else:
            components = collect_startup_components(app)
        status = aggregate_status(components)
    except Exception as e:
        components = {"health_check": {"status": STATUS_DOWN, "detail": str(e)}}
        status = STATUS_DOWN

    return {"status_code": 200, "status": status, "components": components}


async def write_health_file(app: FastAPI) -> None:
    """
    Atomically publish the startup health payload for Nginx without blocking startup. Live checks (LLM) are left out
    because the file is only rewritten when a startup task finishes, so they would go stale.
    """
    temp_file = f"{HEALTH_FILE}.tmp"
    try:
        async with _health_file_lock:
            payload = await build_health_payload(app, include_live=False)
            os.makedirs(os.path.dirname(HEALTH_FILE), exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as file:
                json.dump(payload, file, separators=(",", ":"))
            os.replace(temp_file, HEALTH_FILE)
    except Exception as e:
        logger.error("Failed to write health file", error=str(e))
        try:
            os.remove(temp_file)
        except OSError:
            pass


def remove_health_file() -> None:
    """Remove the health file so Nginx reports 404 after app shutdown."""
    try:
        os.remove(HEALTH_FILE)
    except OSError:
        pass
