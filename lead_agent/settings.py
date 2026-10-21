"""All configuration in one place.

API keys and model choices are read from the .env file in the project folder.
Everything else is a plain constant: edit it here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from lead_agent.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Folders -----------------------------------------------------------------
INPUTS_DIR = PROJECT_ROOT / "inputs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"

SEED_COMPANIES_FILE = INPUTS_DIR / "seed_companies.csv"
NORMALIZATION_RULES_FILE = INPUTS_DIR / "normalization_rules.json"

# --- API keys (from .env) ------------------------------------------------------
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Only needed for Anthropic keys that are not tied to one workspace.
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")

# --- LLM: normalizer leftovers, product fit, profile text, learning -------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6.1-sol")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
LLM_PARALLEL_CALLS = 4

# Similarity for ranking uses OpenAI embeddings. Without an OpenAI key a simple
# word-count similarity is used instead, so the pipeline still runs.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Exa Agent runs --------------------------------------------------------------
# effort: minimal | low | medium | high | xhigh | auto | ultra
# Exa applies max_cost_dollars only to "auto" and "ultra", and max_duration_seconds
# only to "ultra". Ultra runs usually take about 30 minutes.
EXA_RUNS = {
    "taxonomy": {"effort": "ultra", "max_cost_dollars": 10, "max_duration_seconds": 3600},
    "seed_profile": {"effort": "ultra", "max_cost_dollars": 10, "max_duration_seconds": 1800},
    "discovery": {"effort": "ultra", "max_cost_dollars": 10, "max_duration_seconds": 3600},
}
EXA_PARALLEL_RUNS = 5  # seed companies researched at the same time
EXA_POLL_SECONDS = 15

# --- Discovery and ranking --------------------------------------------------------
PROSPECTS_PER_RUN = 10

# score = need_overlap + stack_overlap + profile_similarity + exa_confidence
#         - dislike_similarity        (each part is between 0 and 1, times its weight)
SCORE_WEIGHTS = {
    "need_overlap": 1.0,
    "stack_overlap": 1.0,
    "profile_similarity": 1.0,
    "exa_confidence": 1.0,
    "dislike_similarity": 1.0,
}
EXA_CONFIDENCE_SCORES = {"high": 1.0, "medium": 0.6, "low": 0.3}


def require(name: str) -> str:
    """Return a setting such as "EXA_API_KEY", or explain how to set it when it is empty."""
    value = globals()[name]
    if not value:
        raise ConfigError(f"{name} is not set. Add it to the .env file in the project folder.")
    return value
