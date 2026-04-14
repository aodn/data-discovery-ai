class LLMClientError(Exception):
    """4xx client errors - bad request, auth failure, etc. No point retrying."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"LLM client error {status_code}: {message}")


class LLMServerError(Exception):
    """5xx server errors - service unavailable, timeout, etc."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"LLM server error {status_code}: {message}")
