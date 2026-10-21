"""Ask an LLM for a JSON answer. The provider (OpenAI or Anthropic) is chosen in .env."""
import json
import logging

from lead_agent import settings
from lead_agent.clients.http import request_json
from lead_agent.errors import LLMError

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
ANTHROPIC_MAX_TOKENS = 16000


def ask_json(prompt: str) -> dict:
    """Send the prompt and return the JSON object the model answers with."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        answer = _ask(prompt)
        try:
            return parse_json(answer)
        except ValueError:
            log.warning("The LLM answer was not valid JSON (attempt %d of %d).", attempt, MAX_ATTEMPTS)
    raise LLMError("The LLM did not return valid JSON. Try again, or choose another model in .env.")


def model_name() -> str:
    return settings.ANTHROPIC_MODEL if settings.LLM_PROVIDER == "anthropic" else settings.OPENAI_MODEL


def parse_json(text: str) -> dict:
    """Parse the JSON object in an answer, ignoring ```json fences or text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("No JSON object in the answer.")
    return json.loads(text[start : end + 1])


def _ask(prompt: str) -> str:
    if settings.LLM_PROVIDER == "anthropic":
        return _ask_anthropic(prompt)
    return _ask_openai(prompt)


def _ask_openai(prompt: str) -> str:
    answer = request_json(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.require('OPENAI_API_KEY')}"},
        body={"model": settings.OPENAI_MODEL, "messages": [{"role": "user", "content": prompt}]},
    )
    return answer["choices"][0]["message"]["content"] or ""


def _ask_anthropic(prompt: str) -> str:
    headers = {"x-api-key": settings.require("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}
    if settings.ANTHROPIC_WORKSPACE_ID:
        headers["anthropic-workspace-id"] = settings.ANTHROPIC_WORKSPACE_ID
    answer = request_json(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        body={
            "model": settings.ANTHROPIC_MODEL,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return "".join(block.get("text", "") for block in answer["content"] if block["type"] == "text")
