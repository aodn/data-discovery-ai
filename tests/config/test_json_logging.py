# unit test for config.py's JSON logging setup (issues 9257, 9305)
"""
Every log line on edge/staging/production must be one JSON object in the
schema shared with es-indexer/ogcapi-java (instant/level/loggerName/message/
service/threadId, thrown on exceptions) - whether it comes from this
package, a third-party library logging through the root logger, or uvicorn's
own non-propagating loggers configured from log_config.yaml. Each case runs
in a fresh subprocess: logging.root is global and ConfigUtil subclasses
install different root handlers, so state must not leak between profile
cases.
"""

import ast
import json
import os
import re
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


CORE_FIELDS = {"instant", "level", "loggerName", "message", "service", "threadId"}
INSTANT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_package_logger_produces_java_aligned_json(profile):
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger().setLevel(logging.INFO)\n"
        "logging.getLogger('data_discovery_ai.agents.x').info('hello %s', 'world')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert set(payload) == CORE_FIELDS
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["loggerName"] == "data_discovery_ai.agents.x"
    assert payload["service"] == "data-discovery-ai"
    assert INSTANT.match(payload["instant"])
    assert isinstance(payload["threadId"], int)


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_foreign_stdlib_call_produces_valid_json(profile):
    """The regression test: a plain stdlib logging call (not via structlog)
    must also come out as JSON, not raw text, on edge/staging/production."""
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger().setLevel(logging.INFO)\n"
        # not 'uvicorn.error'/httpx/httpcore/urllib3: set_logging_level() pins
        # those to WARNING on purpose, which would swallow this INFO call
        "logging.getLogger('some_foreign_library').info('hello from stdlib logging')\n"
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 1
    payload = json.loads(lines[0])  # raises json.JSONDecodeError on plain text
    assert payload["message"] == "hello from stdlib logging"
    assert payload["loggerName"] == "some_foreign_library"
    assert payload["service"] == "data-discovery-ai"


def test_development_profile_stays_plain_text():
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger('test.logger').info('hello from dev')\n"
    )
    lines = _lines(_run("development", snippet))

    assert len(lines) == 1
    assert re.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - test\.logger - INFO - hello from dev$",
        lines[0],
    )
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
    assert payload["thrown"]["name"] == "ValueError"
    assert payload["thrown"]["message"] == "boom"
    assert payload["thrown"]["extendedStackTrace"].startswith("Traceback")
    assert "ValueError: boom" in payload["thrown"]["extendedStackTrace"]


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
        assert set(payload) == CORE_FIELDS


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
    render identically - both use JsonLogFormatter."""
    # warning, not info: ProdConfig's root level is WARNING (see #9310's
    # set_logging_level() fix), so an info call would be dropped on production
    snippet = _dictconfig_snippet(
        "logging.getLogger('uvicorn.error').warning('from uvicorn logger')",
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()  # re-runs _init_json_logging on root\n"
        "logging.getLogger('app.module').warning('from root logger')",
    )
    lines = _lines(_run(profile, snippet))

    assert len(lines) == 2
    uvicorn_payload, root_payload = (json.loads(line) for line in lines)
    assert set(uvicorn_payload.keys()) == set(root_payload.keys())


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_extra_fields_are_top_level_json(profile):
    """Call sites that used structlog keyword fields now pass extra={...}."""
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger('x').error('Failed', extra={'status': 503, 'url': 'u'})\n"
    )
    payload = json.loads(_lines(_run(profile, snippet))[0])
    assert payload["status"] == 503
    assert payload["url"] == "u"


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_yaml_handlers_and_root_carry_bound_request_id(profile):
    snippet = _dictconfig_snippet(
        "from data_discovery_ai.utils.log_context import bind_log_context",
        "with bind_log_context(request_id='req-1'):",
        "    logging.getLogger('uvicorn.access').warning('access line')",
        "    logging.getLogger('app.module').warning('root line')",
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env={**os.environ, "PROFILE": profile},
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    (access,) = (json.loads(line) for line in _lines(result.stdout))
    (root,) = (json.loads(line) for line in _lines(result.stderr))
    assert access["request_id"] == root["request_id"] == "req-1"


def test_get_config_with_explicit_profile_ignores_env():
    """get_config(EnvType.EDGE) must log JSON even when PROFILE says
    development - the profile comes from the config class, not the env."""
    snippet = (
        "import logging\n"
        "from data_discovery_ai.config.config import ConfigUtil, EnvType\n"
        "ConfigUtil.get_config(EnvType.EDGE)\n"
        "logging.getLogger('x').warning('explicit edge')\n"
    )
    payload = json.loads(_lines(_run("development", snippet))[0])
    assert payload["message"] == "explicit edge"


STDLIB_LOGGER_KWARGS = {"exc_info", "extra", "stack_info", "stacklevel"}


def test_no_structlog_style_keyword_fields_in_logger_calls():
    """stdlib Logger methods raise TypeError on arbitrary keyword arguments,
    so a leftover structlog-style logger.error('msg', status=...) call would
    crash - typically on an error path. Use extra={...} instead."""
    offenders = []
    for path in (REPO_ROOT / "data_discovery_ai").rglob("*.py"):
        source = path.read_text()
        assert "structlog" not in source, f"{path} still references structlog"
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr
                in {"debug", "info", "warning", "error", "exception", "critical"}
                and "log" in ast.unparse(node.func.value).lower()
            ):
                continue
            bad = {k.arg for k in node.keywords} - STDLIB_LOGGER_KWARGS
            if bad:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {bad}")
    assert not offenders, offenders


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_repeated_get_config_keeps_other_root_handlers(profile):
    """get_config() runs on every request; it must not rebuild root's
    handlers each time (dropping e.g. pytest's caplog handler)."""
    snippet = (
        "import logging, sys\n"
        "from data_discovery_ai.config.config import ConfigUtil\n"
        "ConfigUtil.get_config()\n"
        "extra = logging.StreamHandler(sys.stdout)\n"
        "logging.getLogger().addHandler(extra)\n"
        "ConfigUtil.get_config()\n"
        "logging.getLogger('x').warning('still here')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env={**os.environ, "PROFILE": profile},
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result.stdout) == ["still here"]
    assert json.loads(_lines(result.stderr)[0])["message"] == "still here"


TRANSFORMERS_SNIPPET = (
    "import logging\n"
    "from data_discovery_ai.config.config import ConfigUtil\n"
    "ConfigUtil.get_config()\n"
    "ConfigUtil.get_config()  # runs per request; must stay idempotent\n"
    "from transformers.utils import logging as hf_logging\n"
    "hf_logging.get_logger('transformers.modeling_tf_pytorch_utils')"
    ".warning('Some weights of the PyTorch model were not used\\n- This IS expected')\n"
)


@pytest.mark.parametrize("profile", JSON_PROFILES)
def test_transformers_logs_come_out_as_json_once(profile):
    """transformers installs its own stderr text handler on the 'transformers'
    logger with propagate=False; on JSON profiles its records must go through
    root's JSON handler instead, exactly once."""
    lines = _lines(_run(profile, TRANSFORMERS_SNIPPET))

    assert len(lines) == 1, lines
    payload = json.loads(lines[0])
    assert payload["loggerName"] == "transformers.modeling_tf_pytorch_utils"
    assert payload["message"].startswith("Some weights of the PyTorch model")


def test_transformers_logs_keep_their_own_output_on_development():
    lines = _lines(_run("development", TRANSFORMERS_SNIPPET))

    assert lines[0] == "Some weights of the PyTorch model were not used"
