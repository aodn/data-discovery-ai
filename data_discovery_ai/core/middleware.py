import uuid

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from data_discovery_ai.utils.log_context import bind_log_context


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a fresh request_id for the whole request, so every log line it
    produces (route, SSE body, supervisor worker thread, background tasks)
    carries the same value."""

    async def dispatch(self, request: Request, call_next):
        with bind_log_context(request_id=str(uuid.uuid4())):
            return await call_next(request)


def configure_request_context_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)
