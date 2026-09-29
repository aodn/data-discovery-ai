import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone

from data_discovery_ai.config.config import EnvType

JSON_LOG_PROFILES = (EnvType.EDGE, EnvType.STAGING, EnvType.PRODUCTION)
SERVICE_NAME = "data-discovery-ai"

TEXT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
TEXT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Standard LogRecord attributes; any other attribute (extra={...}, request_id)
# becomes a top-level JSON field.
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
        "color_message",  # uvicorn's ANSI-coloured copy
    }
)


class JsonLogFormatter(logging.Formatter):
    """JSON schema shared with es-indexer/ogcapi-java/data-access-service:
    instant/level/loggerName/message/service/threadId, plus thrown on
    exceptions."""

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

        return json.dumps(payload, default=str)


def use_json_logs(profile: EnvType = None) -> bool:
    """True when the active profile should log JSON rather than text."""
    if profile is None:
        profile = EnvType(os.getenv("PROFILE", EnvType.DEV))
    return profile in JSON_LOG_PROFILES


def build_formatter(
    fmt: str = None, datefmt: str = None, style: str = "%"
) -> logging.Formatter:
    """Formatter for the active profile; fmt/datefmt/style are text-only."""
    if use_json_logs():
        return JsonLogFormatter()
    return logging.Formatter(fmt or TEXT_LOG_FORMAT, datefmt, style)


def install_exception_hooks() -> None:
    """Log uncaught main-thread and thread exceptions as one JSON record
    instead of a raw traceback."""
    logger = logging.getLogger("uncaught")

    def excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        try:
            logger.critical(
                "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
            )
        except Exception:
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    def thread_excepthook(args):
        if args.exc_type is SystemExit:  # ignored by the default hook too
            return
        try:
            logger.error(
                "Uncaught exception in thread %s",
                args.thread.name if args.thread else "<unknown>",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        except Exception:
            threading.__excepthook__(args)

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook
