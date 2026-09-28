import contextvars
import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Iterator, Mapping

# Fields bound here (request_id per HTTP request, job_id per Batch job) are
# added to every log record emitted in the same context, including from
# asyncio tasks and threads started via asyncio.to_thread/copy_context().
# Plain threading.Thread / run_in_executor workers start with an empty context
# and must be handed contextvars.copy_context().run explicitly.
_context: contextvars.ContextVar[Mapping[str, object]] = contextvars.ContextVar(
    "log_context", default=MappingProxyType({})
)


@contextmanager
def bind_log_context(**fields) -> Iterator[None]:
    """Add fields to the log context for the duration of the block. Nested
    binds merge; the previous context is restored on exit."""
    token = _context.set(MappingProxyType({**_context.get(), **fields}))
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> Mapping[str, object]:
    """Read-only view of the fields currently bound."""
    return _context.get()


class ContextFilter(logging.Filter):
    """Copies the bound fields onto each record; JsonLogFormatter emits them as
    top-level fields. Attach to handlers (not loggers) so records propagated
    from child loggers are covered too."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _context.get().items():
            setattr(record, key, value)
        return True


def install_context_filter(handler: logging.Handler) -> None:
    """Attach ContextFilter to handler unless it already has one."""
    if not any(isinstance(f, ContextFilter) for f in handler.filters):
        handler.addFilter(ContextFilter())
