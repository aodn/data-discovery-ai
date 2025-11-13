import structlog
import logging

# Logging config
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)


# add key-value mapping
def add_service_name(logger, method_name, event_dict):
    event_dict["service"] = "data-discovery-ai"
    return event_dict


def rename_timestamp(logger, method_name, event_dict):
    if "timestamp" in event_dict:
        event_dict["instant"] = event_dict.pop("timestamp")
    return event_dict


def rename_logger_name(logger, method_name, event_dict):
    if "logger" in event_dict:
        event_dict["loggerName"] = event_dict.pop("logger")
    return event_dict


def add_thread_info(logger, method_name, event_dict):
    """Add thread ID and priority information"""
    import threading

    thread = threading.current_thread()
    event_dict["threadId"] = thread.ident
    event_dict["threadPriority"] = 5  # use default priority
    return event_dict


def add_logger_name(logger, method_name, event_dict):
    event_dict["loggerName"] = logger.name if hasattr(logger, "name") else __name__
    return event_dict


def add_end_of_batch(logger, method_name, event_dict):
    event_dict["endOfBatch"] = False
    return event_dict


# logging config to output in json format, example format:
# {
#   "instant":"2025-06-06T00:01:44.529Z",
#   "level":"INFO",
#   "loggerName":"au.org.aodn.esindexer.BaseTestClass",
#   "message":"Triggered indexer successfully",
#   "endOfBatch":false,
#   "threadId":1,
#   "threadPriority":5,
#   "service":"es-indexer"
# }
structlog.configure(
    processors=[
        # instant field (timestamp use UTC timezone)
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        rename_timestamp,
        # level field
        structlog.stdlib.add_log_level,
        # loggerName field
        structlog.stdlib.add_logger_name,
        rename_logger_name,
        # message field
        structlog.processors.EventRenamer("message"),
        # endOfBatch field
        add_end_of_batch,
        # threadId and threadPriority fields
        add_thread_info,
        # service field
        add_service_name,
        # in JSON format
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)
