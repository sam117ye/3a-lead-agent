"""Stage 8 — Review logic behind the web app: list leads, record decisions, start a search.

Kept free of web code so it is easy to test and reuse.
A decision is saved to output/decisions/{lead_id}.json together with a snapshot of the lead.
"""
import logging
import re
import threading
from collections import Counter

from lead_agent import pipeline, settings, storage
from lead_agent.stages import stage9_learning
from lead_agent.stages.stage5_ideal_profile import PROFILE_KIND
from lead_agent.utils import now_iso

log = logging.getLogger(__name__)

STATUS_AFTER = {"accept": "accepted", "reject": "rejected"}
LEAD_ID_PATTERN = re.compile(r"[a-z0-9-]+")


class LeadNotFound(LookupError):
    pass


def get_leads(status: str | None = None) -> list[dict]:
    """Leads, best score first, with the decision for reviewed ones.

    status: pending_review, accepted or rejected (None for all).
    """
    decisions = {decision["lead_id"]: decision for decision in storage.load_decisions()}
    leads = storage.load_leads(status)
    for lead in leads:
        decision = decisions.get(lead["lead_id"])
        if decision:
            lead["decision"] = {key: decision[key] for key in ("decision", "comment", "decided_at")}
    return sorted(leads, key=lambda lead: lead["score"], reverse=True)


def record_decision(lead_id: str, decision: str, comment: str = "") -> dict:
    """Save an accept/reject decision and update the lead's status."""
    if decision not in STATUS_AFTER:
        raise ValueError(f"decision must be 'accept' or 'reject', not {decision!r}")
    path = storage.lead_path(lead_id)
    if not LEAD_ID_PATTERN.fullmatch(lead_id) or not path.exists():
        raise LeadNotFound(lead_id)

    lead = storage.read_json(path)
    lead["status"] = STATUS_AFTER[decision]
    record = {
        "lead_id": lead_id,
        "decision": decision,
        "comment": (comment or "").strip(),
        "decided_at": now_iso(),
        "lead_snapshot": lead,
    }
    storage.write_json(storage.decision_path(lead_id), record)
    storage.write_json(path, lead)
    return record


def get_summary() -> dict:
    """Counts per status, the profile version, unlearned decisions and the score weights."""
    counts = Counter(lead["status"] for lead in storage.load_leads())
    summary = {
        "counts": {status: counts[status] for status in ("pending_review", "accepted", "rejected")},
        "profile_version": storage.latest_version(PROFILE_KIND),
        "new_decisions": 0,
        "score_weights": settings.SCORE_WEIGHTS,
    }
    if summary["profile_version"]:
        profile = storage.load_latest(PROFILE_KIND)
        summary["new_decisions"] = len(stage9_learning.unlearned_decisions(profile))
    return summary


# --- Discovery in the background (an Exa run can take many minutes) ---------------------
_discovery = {"running": False, "started_at": None, "finished_at": None, "new_leads": None, "error": None}
_discovery_lock = threading.Lock()


def start_discovery() -> dict:
    """Start Stages 6-7 in a background thread, unless a search is already running."""
    with _discovery_lock:
        if not _discovery["running"]:
            _discovery.update(running=True, started_at=now_iso(), finished_at=None, new_leads=None, error=None)
            threading.Thread(target=_discover_in_background, daemon=True).start()
    return discovery_status()


def discovery_status() -> dict:
    return dict(_discovery)


def _discover_in_background() -> None:
    try:
        _discovery["new_leads"] = len(pipeline.discover())
    except Exception as error:
        log.exception("Discovery failed")
        _discovery["error"] = str(error)
    finally:
        _discovery.update(running=False, finished_at=now_iso())
