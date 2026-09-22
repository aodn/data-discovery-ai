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
    uvicorn.access - they must fall through to the JSON root handler instead.

    This only covers the `python -m data_discovery_ai.server` startup path.
    docker-compose.yml is local-dev only, never used by CI/CD. The actually
    deployed path is the ECS `app_container_command` set per environment in
    aodn/appdeploy (tg/{edge,staging,production,dr-production}/
    data-discovery-ai/ecs/variables.yaml, identical in all four): `uvicorn
    --reload --log-config=log_config.yaml data_discovery_ai.server:app` -
    that applies log_config.yaml unconditionally regardless of this value.
    See the test_yaml_uvicorn_loggers_* cases below, which cover that path
    directly."""
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


LOG_CONFIG_PATH = REPO_ROOT / "log_config.yaml"

def _dictconfig_snippet(*log_calls: str) -> str:
    return (
        "import logging.config, yaml\n"
        f"with open({str(LOG_CONFIG_PATH)!r}) as f:\n"
        "    config = yaml.safe_load(f)\n"
        "logging.config.dictConfig(config)\n" + "\n".join(log_calls) + "\n"
    )


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_yaml_uvicorn_loggers_emit_json_despite_no_propagate(profile):
    """The regression test: uvicorn.error/uvicorn.access have propagate: no
    and their own handlers straight from log_config.yaml - they must still
    come out as JSON, not the plain text log_config.yaml used to hardcode.
    default (uvicorn.error) -> stderr, access (uvicorn.access) -> stdout."""
    snippet = _dictconfig_snippet(
        "logging.getLogger('uvicorn.error').info('Application startup complete.')",
        "logging.getLogger('uvicorn.access').info('some access line')",
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env={**os.environ, "PROFILE": profile},
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    err_lines = _lines(result.stderr)
    out_lines = _lines(result.stdout)

    assert len(err_lines) == 1
    assert len(out_lines) == 1
    error_payload = json.loads(err_lines[0])
    access_payload = json.loads(out_lines[0])
    assert error_payload["loggerName"] == "uvicorn.error"
    assert error_payload["message"] == "Application startup complete."
    assert access_payload["loggerName"] == "uvicorn.access"
    assert access_payload["message"] == "some access line"
    for payload in (error_payload, access_payload):
        assert payload["service"] == "data-discovery-ai"
        assert "instant" in payload
        assert "threadId" in payload


def test_yaml_uvicorn_loggers_stay_plain_text_on_development():
    snippet = _dictconfig_snippet(
        "logging.getLogger('uvicorn.error').warning('Will watch for changes')"
    )
    lines = _lines(_run("development", snippet))

    assert len(lines) == 1
    assert "Will watch for changes" in lines[0]
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_yaml_and_root_handler_json_have_the_same_shape(profile):
    """build_formatter (YAML path) and _init_json_logging (root path) must
    render identically - they share the same SHARED_PROCESSORS chain."""
    snippet = _dictconfig_snippet(
        "logging.getLogger('uvicorn.error').info('from uvicorn logger')",
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()  # re-runs _init_json_logging on root\n"
        "logging.getLogger('app.module').info('from root logger')",
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 2
    uvicorn_payload, root_payload = (json.loads(line) for line in lines)
    assert set(uvicorn_payload.keys()) == set(root_payload.keys())
