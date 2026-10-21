"""Small helpers shared by all stages."""
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

log = logging.getLogger(__name__)


def now_iso() -> str:
    """Current UTC time, e.g. "2026-10-20T12:30:00Z"."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(text: str) -> str:
    """Turn a name into a file-safe id: "Nicol's péniches" -> "nicols-peniches"."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    ascii_text = ascii_text.lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")


def domain_of(url: str) -> str:
    """Company domain used to spot duplicates: "https://www.Najad.se/en" -> "najad.se"."""
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    host = urlparse(url).netloc.lower().split(":")[0]
    return host.removeprefix("www.")


def as_text_list(value) -> list[str]:
    """Clean a value that should be a list of texts (Exa may return None or a single string)."""
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in value or [] if item and str(item).strip()]


def run_parallel(func, items: list, max_workers: int) -> list:
    """Call func(item) for every item using threads.

    A failing item is logged and gives None, so one bad company never stops a whole batch.
    """

    def safe_call(item):
        try:
            return func(item)
        except Exception as error:
            name = item.get("name") if isinstance(item, dict) else item
            log.error("Failed for %s: %s", name, error)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(safe_call, items))
