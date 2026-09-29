import contextvars
import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Iterator, Mapping

# Propagates to asyncio tasks and asyncio.to_thread; plain threads need
# contextvars.copy_context().run.
_context: contextvars.ContextVar[Mapping[str, object]] = contextvars.ContextVar(
    "log_context", default=MappingProxyType({})
)


@contextmanager
def bind_log_context(**fields) -> Iterator[None]:
    """Bind fields for the block; nested binds merge."""
    token = _context.set(MappingProxyType({**_context.get(), **fields}))
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> Mapping[str, object]:
    return _context.get()


class ContextFilter(logging.Filter):
    """Copies bound fields onto each record. Attach to handlers, not loggers,
    so propagated records are covered."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _context.get().items():
            setattr(record, key, value)
        return True


def install_context_filter(handler: logging.Handler) -> None:
    if not any(isinstance(f, ContextFilter) for f in handler.filters):
        handler.addFilter(ContextFilter())
