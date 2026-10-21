"""Prompt templates live in prompts/*.txt so they can be edited without touching code.

Placeholders use $name syntax (Python string.Template). Every template can also use
$company_context, the 3A company description from prompts/3a_context.txt.
Lists of texts are inserted as bullet lists; other lists and dicts as JSON.
"""
import json
from string import Template

from lead_agent import settings


def render(template_name: str, **values) -> str:
    values.setdefault("company_context", _read("3a_context"))
    texts = {key: _as_text(value) for key, value in values.items()}
    return Template(_read(template_name)).substitute(texts)


def _read(template_name: str) -> str:
    return (settings.PROMPTS_DIR / f"{template_name}.txt").read_text(encoding="utf-8").strip()


def _as_text(value) -> str:
    if isinstance(value, str):
        return value or "(none)"
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "\n".join(f"- {item}" for item in value) if value else "(none)"
    return json.dumps(value, indent=2, ensure_ascii=False)
