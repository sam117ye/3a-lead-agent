"""Reading and writing the pipeline's files.

Every stage saves JSON under output/, following the plan's artifact map:

    output/data/taxonomy_v1.json                              Stage 1
    output/profiles/{seed_id}/manufacturer_profile.json       Stage 2
    output/data/taxonomy_v1_normalized.json                   Stage 3
    output/profiles/{seed_id}/normalized.json                 Stage 3
    output/profiles/{seed_id}/product_fit.json                Stage 4
    output/data/ideal_client_profile_v1.json                  Stage 5 (v2, v3, ... by Stage 9)
    output/runs/{run_id}/prospects.json                       Stage 6
    output/leads/{lead_id}.json                               Stage 7
    output/decisions/{lead_id}.json                           Stage 8
    output/data/disliked_examples.json                        Stage 9
    output/exa_pending/{task}.json      Exa runs still in progress (lets an interrupted run resume)
"""
import csv
import json
import re
from pathlib import Path

from lead_agent import settings
from lead_agent.errors import MissingInputError
from lead_agent.utils import slugify


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_up_to_date(output: Path, *sources: Path) -> bool:
    """True when the output file exists and is newer than all of its sources (like `make`)."""
    if not output.exists():
        return False
    return all(output.stat().st_mtime >= source.stat().st_mtime for source in sources if source.exists())


def write_json(path: Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --- Inputs ------------------------------------------------------------------------
def read_seeds() -> list[dict]:
    """Seed companies from inputs/seed_companies.csv (columns: company, website)."""
    with open(settings.SEED_COMPANIES_FILE, newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if (row.get("company") or "").strip()]
    return [
        {"seed_id": slugify(row["company"]), "name": row["company"].strip(), "website": row["website"].strip()}
        for row in rows
    ]


# --- Versioned files in output/data (taxonomy, ideal client profile) -----------------
def data_path(file_name: str) -> Path:
    return settings.OUTPUT_DIR / "data" / file_name


def versioned_path(kind: str, version: int, suffix: str = "") -> Path:
    """versioned_path("taxonomy", 1, "_normalized") -> output/data/taxonomy_v1_normalized.json"""
    return data_path(f"{kind}_v{version}{suffix}.json")


def latest_version(kind: str) -> int:
    """Highest saved version number of a file kind, or 0 when there is none yet."""
    pattern = re.compile(rf"{kind}_v(\d+)\.json")
    versions = [
        int(match.group(1))
        for path in (settings.OUTPUT_DIR / "data").glob(f"{kind}_v*.json")
        if (match := pattern.fullmatch(path.name))
    ]
    return max(versions, default=0)


def load_latest(kind: str, suffix: str = "") -> dict:
    """Load the newest version of a file kind, e.g. load_latest("ideal_client_profile")."""
    version = latest_version(kind)
    path = versioned_path(kind, version, suffix)
    if version == 0 or not path.exists():
        raise MissingInputError(f"{path.name} not found. Run the earlier stages first (python -m lead_agent run).")
    return read_json(path)


# --- Per-seed files in output/profiles/{seed_id} ---------------------------------
def seed_file(seed_id: str, name: str) -> Path:
    """seed_file("najad", "normalized") -> output/profiles/najad/normalized.json"""
    return settings.OUTPUT_DIR / "profiles" / seed_id / f"{name}.json"


def load_seed_files(name: str) -> list[dict]:
    """The given file for every seed that has it, in seed-list order."""
    paths = [seed_file(seed["seed_id"], name) for seed in read_seeds()]
    documents = [read_json(path) for path in paths if path.exists()]
    if not documents:
        raise MissingInputError(f"No seed files named {name}.json yet. Run the earlier stages first.")
    return documents


# --- Discovery runs, leads and decisions -------------------------------------------
def prospects_path(run_id: str) -> Path:
    return settings.OUTPUT_DIR / "runs" / run_id / "prospects.json"


def lead_path(lead_id: str) -> Path:
    return settings.OUTPUT_DIR / "leads" / f"{lead_id}.json"


def decision_path(lead_id: str) -> Path:
    return settings.OUTPUT_DIR / "decisions" / f"{lead_id}.json"


def load_leads(status: str | None = None) -> list[dict]:
    leads = [read_json(path) for path in sorted((settings.OUTPUT_DIR / "leads").glob("*.json"))]
    return [lead for lead in leads if status is None or lead["status"] == status]


def load_decisions() -> list[dict]:
    """All review decisions, oldest first."""
    decisions = [read_json(path) for path in (settings.OUTPUT_DIR / "decisions").glob("*.json")]
    return sorted(decisions, key=lambda decision: decision["decided_at"])


def pending_exa_run_path(task_name: str) -> Path:
    return settings.OUTPUT_DIR / "exa_pending" / f"{task_name}.json"
