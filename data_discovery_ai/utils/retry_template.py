"""Reusable retry policy with exponential backoff and jitter."""

from dataclasses import dataclass
import logging
from typing import Callable

import structlog
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_before_delay,
    wait_exponential_jitter,
)

logger = structlog.get_logger(__name__)

# HTTP statuses that are safe to retry.
RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """Retry matching exceptions with bounded exponential backoff.

    ``max_attempts`` includes the first call. ``max_elapsed`` prevents a wait
    that would begin the next attempt at or beyond the elapsed-time budget.
    Individual calls must still have their own timeout because a retry policy
    cannot interrupt a call that is already in progress.
    """

    name: str
    retry_on: tuple[type[BaseException], ...] = ()
    retry_if: Callable[[BaseException], bool] | None = None
    max_attempts: int = 10
    initial: float = 1.0
    max_wait: float = 30.0
    max_elapsed: float | None = None

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, self.retry_on) or (
            self.retry_if is not None and self.retry_if(exc)
        )

    def __call__(self, func):
        stop = stop_after_attempt(self.max_attempts)
        if self.max_elapsed is not None:
            stop |= stop_before_delay(self.max_elapsed)

        # Works as a decorator for both sync and async functions
        return retry(
            retry=retry_if_exception(self.is_retryable),
            stop=stop,
            wait=wait_exponential_jitter(initial=self.initial, max=self.max_wait),
            # Built-in hook: logs "Retrying <func> in <n> seconds as it raised <exc>."
            before_sleep=before_sleep_log(logger, logging.WARNING),
            # Raise the original exception instead of tenacity.RetryError
            reraise=True,
        )(func)
