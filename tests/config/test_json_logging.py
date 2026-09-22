# unit test for config.py's JSON logging setup (issue 9257)
"""
Regression coverage for the foreign_pre_chain bug: before the fix, JSON
encoding only happened inside structlog.processors.JSONRenderer(), which
only structlog-originated events reached. The handler actually attached to
logging.root was a plain logging.Formatter("%(message)s") pass-through, so
any plain logging.getLogger(...) call (uvicorn, httpx, urllib3, etc.) printed
as raw un-JSON'd text on edge/staging/production. Each case here runs in a
fresh subprocess: structlog.configure() and logging.root are global,
cache_logger_on_first_use=True caches loggers process-wide, and ConfigUtil
subclasses install different root handlers, so state must not leak between
profile cases.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(profile: str, snippet: str) -> str:
    """Run snippet in a fresh interpreter with PROFILE=<profile>, return stderr."""
    env = {**os.environ, "PROFILE": profile}
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def _lines(output: str) -> list:
    return [line for line in output.splitlines() if line.strip()]


JSON_PROFILES = ["edge", "staging", "production"]


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_structlog_call_produces_valid_json(profile):
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger().setLevel(logging.INFO)\n"
        "import structlog\n"
        "structlog.get_logger('test.logger').info('hello from structlog')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "hello from structlog"
    assert payload["level"].lower() == "info"
    assert payload["service"] == "data-discovery-ai"
    assert "instant" in payload
    assert "loggerName" in payload
    assert "threadId" in payload
    assert payload["endOfBatch"] is False


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_foreign_stdlib_call_produces_valid_json(profile):
    """The regression test: a plain stdlib logging call (not via structlog)
    must also come out as JSON, not raw text, on edge/staging/production."""
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger().setLevel(logging.INFO)\n"
        "logging.getLogger('uvicorn.error').info('hello from stdlib logging')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1
    payload = json.loads(lines[0])  # raises json.JSONDecodeError on plain text
    assert payload["message"] == "hello from stdlib logging"
    assert payload["loggerName"] == "uvicorn.error"
    assert payload["service"] == "data-discovery-ai"


def test_development_profile_stays_plain_text():
    # development installs no explicit root handler (pre-existing, out of
    # scope here); Python's logging.lastResort fallback only fires at
    # WARNING+, so use that rather than accepting empty output at INFO.
    snippet = (
        "import structlog\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "structlog.get_logger('test.logger').warning('hello from dev')\n"
    )
    lines = _lines(_run("development", snippet))

    assert len(lines) == 1
    assert "hello from dev" in lines[0]
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_stdlib_exception_emits_one_json_object_with_traceback(profile):
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logger = logging.getLogger('test.logger')\n"
        "try:\n"
        "    raise ValueError('boom')\n"
        "except ValueError:\n"
        "    logger.exception('stdlib failure')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1  # multiline traceback stays one physical output line
    payload = json.loads(lines[0])
    assert payload["message"] == "stdlib failure"
    assert "ValueError: boom" in payload["exception"]
    assert "Traceback" in payload["exception"]


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_structlog_exception_emits_one_json_object_with_traceback(profile):
    snippet = (
        "import structlog\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logger = structlog.get_logger('test.logger')\n"
        "try:\n"
        "    raise ValueError('boom')\n"
        "except ValueError:\n"
        "    logger.exception('structlog failure')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "structlog failure"
    assert "ValueError: boom" in payload["exception"]
    assert "Traceback" in payload["exception"]


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_multiline_message_stays_one_physical_output_line(profile):
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger('test.logger').warning('line1\\nline2')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "line1\nline2"


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_log_config_path_is_none_so_uvicorn_logging_is_not_bypassed(profile):
    """On JSON profiles uvicorn.run(log_config=...) must get None so uvicorn
    does not install its own non-propagating text handlers on uvicorn.error/
    uvicorn.access - they must fall through to the JSON root handler instead."""
    snippet = (
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "config = ConfigUtil.get_config()\n"
        "print(config.log_config_path)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env={**os.environ, "PROFILE": profile},
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None"


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_reinitializing_config_does_not_duplicate_output(profile):
    """Guards against accumulating a second handler on logging.root if
    ConfigUtil.get_config() (and therefore _init_json_logging) runs twice."""
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger('uvicorn.error').warning('once only')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1
