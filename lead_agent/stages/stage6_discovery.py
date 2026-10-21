"""Stage 6 — Discover prospects (Exa Agent list-build).

Finds new companies that match the latest Ideal Client Profile and saves them to
output/runs/{run_id}/prospects.json. Exa only finds candidates: Stage 7 assigns products and scores.
"""
import logging
import re
from datetime import datetime

from lead_agent import prompts, settings, storage
from lead_agent.clients import exa
from lead_agent.utils import as_text_list, domain_of, now_iso

log = logging.getLogger(__name__)

MATCH_CONFIDENCES = ["high", "medium", "low"]


def new_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def prospect_schema(count: int) -> dict:
    signals = {"type": "array", "maxItems": 10, "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "prospects": {
                "type": "array",
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "website": {"type": "string", "description": "Official company website."},
                        "country": {"type": "string"},
                        "segment": {"type": "string", "description": "What they build, e.g. 'sailing catamaran OEM'."},
                        "selection_rationale": {"type": "string", "description": "Why it matches, citing evidence."},
                        "material_signals": {**signals, "description": "Evidence of materials and processes used."},
                        "need_signals": {**signals, "description": "Evidence of engineering or production needs."},
                        "match_confidence": {"type": "string", "enum": MATCH_CONFIDENCES},
                    },
                    "required": [
                        "name",
                        "website",
                        "country",
                        "segment",
                        "selection_rationale",
                        "material_signals",
                        "need_signals",
                        "match_confidence",
                    ],
                },
            }
        },
        "required": ["prospects"],
    }


def discover_prospects(profile: dict, run_id: str) -> dict:
    """Ask Exa for new companies that fit the profile, skipping known ones."""
    exclusion = exclusion_list(profile)
    query = prompts.render(
        "stage6_discover",
        count=settings.PROSPECTS_PER_RUN,
        brief=profile["exa_search_brief"],
        company_types=profile["company_types"],
        must_have=profile["must_have"],
        must_not=profile["must_not"],
        derived_rules=[f"{rule['type']}: {rule['rule']}" for rule in profile["derived_rules"]],
        positive_examples=profile["positive_examples"],
    )
    log.info("Stage 6: searching for up to %d new prospects (%d companies excluded)...", settings.PROSPECTS_PER_RUN, len(exclusion))
    run = exa.run_agent(
        "discovery", query, prospect_schema(settings.PROSPECTS_PER_RUN), settings.EXA_RUNS["discovery"], exclusion
    )

    grounding = exa.grounding(run)
    found = exa.structured_output(run).get("prospects") or []
    prospects = [_clean_prospect(prospect, index, grounding) for index, prospect in enumerate(found) if prospect]
    result = {
        "run_id": run_id,
        "created_at": now_iso(),
        "profile_version": profile["version"],
        **exa.run_metadata(run),
        "excluded_count": len(exclusion),
        "prospects": [prospect for prospect in prospects if prospect["name"]],
        "grounding": grounding,
    }
    storage.write_json(storage.prospects_path(run_id), result)
    log.info("Stage 6: Exa returned %d prospects (run %s).", len(result["prospects"]), run_id)
    return result


def exclusion_list(profile: dict) -> list[dict]:
    """Seeds, reviewed leads and leads already in the queue: Exa should not return them again."""
    entries = profile["exclusions"] + [{"name": lead["name"], "website": lead["website"]} for lead in storage.load_leads()]
    unique = {}
    for entry in entries:
        key = domain_of(entry.get("website")) or entry["name"].lower()
        unique.setdefault(key, {"name": entry["name"], "website": entry.get("website") or ""})
    return list(unique.values())


def _clean_prospect(prospect: dict, index: int, grounding: list[dict]) -> dict:
    confidence = str(prospect.get("match_confidence") or "low").lower()
    return {
        "name": str(prospect.get("name") or "").strip(),
        "website": str(prospect.get("website") or "").strip(),
        "country": str(prospect.get("country") or "").strip(),
        "segment": str(prospect.get("segment") or "").strip(),
        "selection_rationale": str(prospect.get("selection_rationale") or "").strip(),
        "material_signals": as_text_list(prospect.get("material_signals")),
        "need_signals": as_text_list(prospect.get("need_signals")),
        "match_confidence": confidence if confidence in MATCH_CONFIDENCES else "low",
        "citations": _citations_for(index, grounding),
    }


def _citations_for(index: int, grounding: list[dict]) -> list[dict]:
    """Citations of prospect number `index`.

    Grounding fields point into the output, e.g. "prospects[3].website"; the first number
    in the field is the prospect's position in the list.
    """
    citations = []
    for item in grounding:
        numbers = re.findall(r"\d+", str(item.get("field") or ""))
        if not numbers or int(numbers[0]) != index:
            continue
        for citation in item.get("citations") or []:
            url = citation.get("url")
            if url and all(existing["url"] != url for existing in citations):
                citations.append({"url": url, "title": citation.get("title") or ""})
    return citations
