"""Stage 9 — Active learning.

Feeds human review decisions back into the Ideal Client Profile, and so into the next search:
- the LLM turns decisions and comments into rules ("exclude: ...", "require: ...") and rewrites
  must_have / must_not / exa_search_brief  ->  output/data/ideal_client_profile_v{N+1}.json
- reviewed leads join the exclusions; accepted leads become positive examples
- rejected leads and their comments are saved to output/data/disliked_examples.json, which
  Stage 7 uses to push down prospects that look like them
"""
import logging

from lead_agent import prompts, storage
from lead_agent.clients import llm
from lead_agent.stages.stage5_ideal_profile import PROFILE_KIND, example_line
from lead_agent.stages.stage7_ranking import DISLIKED_EXAMPLES_FILE, company_text
from lead_agent.utils import as_text_list, now_iso

log = logging.getLogger(__name__)

RULE_TYPES = ("exclude", "require")
PROFILE_FIELDS_TO_UPDATE = ("company_types", "must_have", "must_not", "derived_rules", "exa_search_brief")


def learn() -> dict:
    """Create the next profile version from new review decisions. Returns a short summary."""
    profile = storage.load_latest(PROFILE_KIND)
    save_disliked_examples(storage.load_decisions())
    decisions = unlearned_decisions(profile)
    if not decisions:
        log.info("Stage 9: no new decisions to learn from.")
        return {"updated": False, "version": profile["version"], "change_summary": "No new decisions to learn from."}

    log.info("Stage 9: learning from %d new decisions...", len(decisions))
    answer = llm.ask_json(
        prompts.render(
            "stage9_learn",
            profile={field: profile[field] for field in PROFILE_FIELDS_TO_UPDATE},
            decisions=[_decision_for_prompt(decision) for decision in decisions],
        )
    )
    updated = apply_feedback(profile, decisions, answer)
    storage.write_json(storage.versioned_path(PROFILE_KIND, updated["version"]), updated)
    log.info("Stage 9: saved profile v%d. %s", updated["version"], updated["change_summary"])
    return {
        "updated": True,
        "version": updated["version"],
        "change_summary": updated["change_summary"],
        "new_rules": updated["derived_rules"][len(profile["derived_rules"]) :],
    }


def unlearned_decisions(profile: dict) -> list[dict]:
    """Review decisions this profile version has not learned from yet."""
    learned = set(profile["learned_from_decisions"])
    return [decision for decision in storage.load_decisions() if decision["lead_id"] not in learned]


def apply_feedback(profile: dict, decisions: list[dict], answer: dict) -> dict:
    """Profile vN+1 = vN + the LLM's rewrite + exclusions and examples from the decisions."""
    new_rules = [
        {
            "type": rule["type"],
            "rule": str(rule["rule"]).strip(),
            "lead_ids": as_text_list(rule.get("lead_ids")),
            "created_at": now_iso(),
        }
        for rule in answer.get("derived_rules") or []
        if isinstance(rule, dict) and rule.get("type") in RULE_TYPES and rule.get("rule")
    ]
    accepted_leads = [decision["lead_snapshot"] for decision in decisions if decision["decision"] == "accept"]
    return {
        **profile,
        "version": profile["version"] + 1,
        "parent_version": profile["version"],
        "created_at": now_iso(),
        "must_have": _new_list_or_keep(answer.get("must_have"), profile["must_have"]),
        "must_not": _new_list_or_keep(answer.get("must_not"), profile["must_not"]),
        "exa_search_brief": str(answer.get("exa_search_brief") or profile["exa_search_brief"]),
        "derived_rules": profile["derived_rules"] + new_rules,
        "exclusions": profile["exclusions"] + [_exclusion(decision) for decision in decisions],
        "positive_examples": profile["positive_examples"] + [_lead_example(lead) for lead in accepted_leads],
        "learned_from_decisions": profile["learned_from_decisions"] + [d["lead_id"] for d in decisions],
        "change_summary": str(answer.get("change_summary") or ""),
    }


def save_disliked_examples(decisions: list[dict]) -> None:
    """Rejected leads plus the reviewer's comment, for the Stage 7 dislike penalty."""
    examples = [
        {
            "lead_id": decision["lead_id"],
            "name": decision["lead_snapshot"]["name"],
            "comment": decision["comment"],
            "text": f"{company_text(decision['lead_snapshot'])}. Rejected: {decision['comment'] or 'not a fit'}",
        }
        for decision in decisions
        if decision["decision"] == "reject"
    ]
    storage.write_json(storage.data_path(DISLIKED_EXAMPLES_FILE), examples)


# --- Helpers ---------------------------------------------------------------------------
def _decision_for_prompt(decision: dict) -> dict:
    lead = decision["lead_snapshot"]
    return {
        "lead_id": decision["lead_id"],
        "decision": decision["decision"],
        "comment": decision["comment"] or "(no comment)",
        "company": {
            "name": lead["name"],
            "website": lead["website"],
            "country": lead["country"],
            "segment": lead["segment"],
            "selection_rationale": lead["selection_rationale"],
            "tokens": lead["tokens"],
            "top_fits": [f"{fit['product_id']} ({fit['fit_type']})" for fit in lead["fits"][:3]],
        },
    }


def _exclusion(decision: dict) -> dict:
    lead = decision["lead_snapshot"]
    reason = "accepted" if decision["decision"] == "accept" else f"rejected: {decision['comment'] or 'no comment'}"
    return {"name": lead["name"], "website": lead["website"], "reason": reason}


def _lead_example(lead: dict) -> str:
    return example_line(lead["name"], lead["website"], lead["segment"], lead["tokens"]["material"] + lead["tokens"]["process"])


def _new_list_or_keep(value, current: list[str]) -> list[str]:
    """The LLM's new list when it gave one, otherwise the current list."""
    return as_text_list(value) if isinstance(value, list) else current
