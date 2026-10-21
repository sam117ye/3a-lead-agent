"""`python -m lead_agent check`: test the API keys with cheap calls before a long run."""
from lead_agent import settings
from lead_agent.clients import embeddings, exa, llm
from lead_agent.errors import LLMError


def check_connections() -> list[dict]:
    """One result per service: {"name", "ok", "detail"}."""
    return [
        _check("LLM", _check_llm),
        _check("Similarity", _check_embeddings),
        _check("Exa", _check_exa),
    ]


def _check(name: str, check) -> dict:
    try:
        return {"name": name, "ok": True, "detail": check()}
    except Exception as error:  # report every problem instead of stopping at the first
        return {"name": name, "ok": False, "detail": str(error)}


def _check_llm() -> str:
    answer = llm.ask_json('Reply with this JSON object and nothing else: {"ok": true}')
    if answer.get("ok") is not True:
        raise LLMError(f"Unexpected answer: {answer}")
    return f"{settings.LLM_PROVIDER} model {llm.model_name()} answered."


def _check_embeddings() -> str:
    if not settings.OPENAI_API_KEY:
        return "No OpenAI key, so ranking uses a simple word-count similarity."
    vector = embeddings.embed(["sandwich composite core"])[0]
    return f"OpenAI {settings.EMBEDDING_MODEL} answered ({len(vector)} dimensions)."


def _check_exa() -> str:
    exa.list_runs(limit=1)  # listing runs is free
    return "The Exa Agent API accepted the key."
