import logging
import uuid

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from data_discovery_ai.utils.log_context import bind_log_context

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a request_id to every log line of a request."""

    async def dispatch(self, request: Request, call_next):
        with bind_log_context(request_id=str(uuid.uuid4())):
            try:
                return await call_next(request)
            except Exception:
                # log while request_id is bound; avoids uvicorn's duplicate
                logger.exception(
                    "Unhandled error processing %s %s",
                    request.method,
                    request.url.path,
                )
                return PlainTextResponse("Internal Server Error", status_code=500)


def configure_request_context_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)
