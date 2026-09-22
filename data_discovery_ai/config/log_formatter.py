import logging
import os
import threading

import structlog

from data_discovery_ai.config.config import EnvType

JSON_LOG_PROFILES = (EnvType.EDGE, EnvType.STAGING, EnvType.PRODUCTION)


def _add_service_name(logger, method_name, event_dict):
    event_dict["service"] = "data-discovery-ai"
    return event_dict


def _rename_timestamp(logger, method_name, event_dict):
    if "timestamp" in event_dict:
        event_dict["instant"] = event_dict.pop("timestamp")
    return event_dict


def _rename_logger_name(logger, method_name, event_dict):
    if "logger" in event_dict:
        event_dict["loggerName"] = event_dict.pop("logger")
    return event_dict


def _add_thread_info(logger, method_name, event_dict):
    thread = threading.current_thread()
    event_dict["threadId"] = thread.ident
    event_dict["threadPriority"] = 5  # use default priority
    return event_dict


def _add_end_of_batch(logger, method_name, event_dict):
    event_dict["endOfBatch"] = False
    return event_dict


# Shared by both structlog-native events and "foreign" (plain stdlib logging)
# records, so every log line ends up with the same fields before being
# JSON-rendered - used both as ConfigUtil._init_json_logging's root handler
# foreign_pre_chain and, via build_formatter below, by any handler configured
# straight from log_config.yaml (e.g. uvicorn's own non-propagating
# uvicorn.error/uvicorn.access handlers, which never reach the root handler).
SHARED_PROCESSORS = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    _rename_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    _rename_timestamp,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.EventRenamer("message"),
    _add_end_of_batch,
    _add_thread_info,
    _add_service_name,
]


def use_json_logs(profile: EnvType = None) -> bool:
    """True when the active profile should log JSON rather than text."""
    if profile is None:
        profile = EnvType(os.getenv("PROFILE", EnvType.DEV))
    return profile in JSON_LOG_PROFILES


def build_formatter(
    fmt: str = None, datefmt: str = None, style: str = "%"
) -> logging.Formatter:
    """Formatter for the active profile, for use as a logging.config
    dictConfig formatter factory (see log_config.yaml). JSON profiles get a
    structlog ProcessorFormatter using the same SHARED_PROCESSORS chain as
    ConfigUtil._init_json_logging, so a handler wired straight from YAML -
    such as uvicorn's own uvicorn.error/uvicorn.access handlers, which have
    propagate: no and so never reach the root handler - still renders
    identically-shaped JSON. fmt/datefmt/style only apply to the text
    profile; ignored for JSON.
    """
    if use_json_logs():
        return structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=SHARED_PROCESSORS,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    return logging.Formatter(fmt, datefmt, style)
