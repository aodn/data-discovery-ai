"""Shared helpers for building tenacity retry decorators.

Retry decorators are defined with ``tenacity.retry(...)`` next to the code they
protect. Each call must still have its own timeout because a retry decorator
cannot interrupt a call that is already in progress.
"""

import structlog

logger = structlog.get_logger(__name__)

# HTTP statuses that are safe to retry.
RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def log_retry(retry_state) -> None:
    """``before_sleep`` callback: log the failed attempt and the next wait."""
    fn = retry_state.fn
    logger.warning(
        "Retrying after transient failure",
        function=getattr(fn, "__name__", repr(fn)),
        attempt=retry_state.attempt_number,
        next_wait_seconds=round(retry_state.next_action.sleep, 2),
        error=repr(retry_state.outcome.exception()),
    )
