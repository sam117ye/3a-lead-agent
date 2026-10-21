"""JSON-over-HTTPS with retries, shared by the Exa, OpenAI and Anthropic clients."""
import logging
import time

import requests

from lead_agent.errors import ApiError

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}  # busy or temporary errors


def request_json(method: str, url: str, headers: dict, body: dict | None = None, timeout: int = 300) -> dict:
    """Send a JSON request and return the JSON answer.

    Network errors and busy/temporary errors are retried with growing waits (6s, 12s, 24s, 48s).
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.request(method, url, headers=headers, json=body, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as error:
            status_code, problem = None, str(error)
        else:
            if response.ok:
                return response.json()
            status_code, problem = response.status_code, f"HTTP {response.status_code}: {response.text[:500]}"

        retryable = status_code is None or status_code in RETRY_STATUS_CODES
        if not retryable or attempt == MAX_ATTEMPTS:
            raise ApiError(f"{method} {url} failed: {problem}", status_code)
        wait_seconds = 3 * 2**attempt
        log.warning("%s %s failed (%s). Retrying in %ss.", method, url, problem[:120], wait_seconds)
        time.sleep(wait_seconds)
