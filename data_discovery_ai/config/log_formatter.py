import json
import logging
import os
from datetime import datetime, timezone

# config.py imports this module lazily (inside set_logging_level), so this
# top-level import of EnvType is not circular.
from data_discovery_ai.config.config import EnvType

JSON_LOG_PROFILES = (EnvType.EDGE, EnvType.STAGING, EnvType.PRODUCTION)
SERVICE_NAME = "data-discovery-ai"

TEXT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
TEXT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Standard LogRecord attributes. Anything else on a record came from a caller's
# extra={...} or from ContextFilter (request_id/job_id) and is emitted as a
# top-level JSON field.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        # uvicorn attaches an ANSI-coloured copy of some messages via extra=
        "color_message",
    }
)


class JsonLogFormatter(logging.Formatter):
    """Field names match what es-indexer/ogcapi-java already emit via log4j2's
    JsonTemplateLayout (instant/level/loggerName/message/service/threadId, plus
    thrown on exceptions) so CloudWatch queries work across services. A real
    es-indexer line for reference:
        {"instant":"2025-06-06T00:01:44.529Z","level":"INFO",
         "loggerName":"au.org.aodn.esindexer.BaseTestClass",
         "message":"Triggered indexer successfully","endOfBatch":false,
         "threadId":1,"threadPriority":5,"service":"es-indexer"}
    endOfBatch/threadPriority are Log4j2/JVM internals and are not copied.

    Kept identical to data-access-service's
    data_access_service/utils/log_formatter.py apart from SERVICE_NAME and
    use_json_logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "instant": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "loggerName": record.name,
            "message": record.getMessage(),
            "service": SERVICE_NAME,
            "threadId": record.thread,
        }

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["thrown"] = {
                "name": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "extendedStackTrace": self.formatException(record.exc_info),
            }
        elif record.exc_text:
            payload["thrown"] = {"extendedStackTrace": record.exc_text}

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and key not in payload:
                payload[key] = value

        # default=str so a stray non-string value cannot kill the log line.
        return json.dumps(payload, default=str)


def use_json_logs(profile: EnvType = None) -> bool:
    """True when the active profile should log JSON rather than text."""
    if profile is None:
        profile = EnvType(os.getenv("PROFILE", EnvType.DEV))
    return profile in JSON_LOG_PROFILES


def build_formatter(
    fmt: str = None, datefmt: str = None, style: str = "%"
) -> logging.Formatter:
    """Formatter for the active profile. fmt/datefmt/style only apply to the
    text profiles; ignored for JSON."""
    if use_json_logs():
        return JsonLogFormatter()
    return logging.Formatter(fmt or TEXT_LOG_FORMAT, datefmt, style)
