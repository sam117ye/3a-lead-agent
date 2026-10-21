"""Stage 2 — Seed manufacturing profiles (Exa Agent).

One company-level research chart per seed in inputs/seed_companies.csv, saved to
output/profiles/{seed_id}/manufacturer_profile.json.

Empty lists mean "not disclosed", not "absent". Seeds may be confirmed buyers or aspirational targets.
"""
import logging

from lead_agent import prompts, settings, storage
from lead_agent.clients import exa
from lead_agent.utils import as_text_list, now_iso, run_parallel

log = logging.getLogger(__name__)

PROFILE_FILE = "manufacturer_profile"
STACK_PARTS = ["cores", "reinforcements", "resins", "veils", "infusion_media"]
LIST_FIELDS = ["end_products", "manufacturing_processes", "production_constraints", "performance_mandates"]


def _text_list(description: str) -> dict:
    return {"type": "array", "maxItems": 15, "items": {"type": "string"}, "description": description}


PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "manufacturer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "website": {"type": "string"},
                "location": {"type": "string", "description": "Headquarters and main shipyards."},
            },
            "required": ["name", "website", "location"],
        },
        "end_products": _text_list("Vessels or components they build: model ranges, boat types, sizes."),
        "manufacturing_processes": _text_list("How they build, e.g. vacuum infusion, hand lay-up, RTM, prepreg."),
        "material_stack": {
            "type": "object",
            "properties": {
                "cores": _text_list("Sandwich core materials (foam, balsa, honeycomb...), with brands if known."),
                "reinforcements": _text_list("Glass, carbon, basalt or natural fibres."),
                "resins": _text_list("Polyester, vinylester, epoxy, bio-based resins."),
                "veils": _text_list("Surfacing veils and finishing mats."),
                "infusion_media": _text_list("Flow media and infusion core mats."),
            },
            "required": STACK_PARTS,
        },
        "production_constraints": _text_list("Limits on production: labour, cycle time, capacity, cost, rules."),
        "performance_mandates": _text_list("Targets they commit to: weight, stiffness, range, sustainability."),
        "core_engineering_challenge": {
            "type": "string",
            "description": "The main engineering challenge where sandwich composites could help (1-2 sentences).",
        },
    },
    "required": ["manufacturer", *LIST_FIELDS, "material_stack", "core_engineering_challenge"],
}


def build_seed_profiles(force: bool = False) -> list[dict]:
    """Research every seed company that has no profile yet (or all of them with force)."""
    seeds = storage.read_seeds()
    todo = [seed for seed in seeds if force or not storage.seed_file(seed["seed_id"], PROFILE_FILE).exists()]
    log.info("Stage 2: %d of %d seed companies to research.", len(todo), len(seeds))
    run_parallel(profile_seed, todo, settings.EXA_PARALLEL_RUNS)

    missing = [seed["name"] for seed in seeds if not storage.seed_file(seed["seed_id"], PROFILE_FILE).exists()]
    if missing:
        log.warning("Stage 2: no profile yet for %s. Run the command again to retry.", ", ".join(missing))
    return storage.load_seed_files(PROFILE_FILE)


def profile_seed(seed: dict) -> dict:
    """Research one seed company with Exa and save its manufacturing profile."""
    query = prompts.render("stage2_seed_profile", company=seed["name"], website=seed["website"])
    run = exa.run_agent(f"seed_{seed['seed_id']}", query, PROFILE_SCHEMA, settings.EXA_RUNS["seed_profile"])
    profile = {
        "seed_id": seed["seed_id"],
        **_complete_profile(exa.structured_output(run), seed),
        "grounding": exa.grounding(run),
        **exa.run_metadata(run),
        "created_at": now_iso(),
    }
    storage.write_json(storage.seed_file(seed["seed_id"], PROFILE_FILE), profile)
    log.info("Stage 2: profiled %s", seed["name"])
    return profile


def _complete_profile(data: dict, seed: dict) -> dict:
    """Give every field a value (empty when undisclosed) so later stages can rely on the shape."""
    manufacturer = data.get("manufacturer") or {}
    stack = data.get("material_stack") or {}
    return {
        "manufacturer": {
            "name": manufacturer.get("name") or seed["name"],
            "website": manufacturer.get("website") or seed["website"],
            "location": manufacturer.get("location") or "",
        },
        **{field: as_text_list(data.get(field)) for field in LIST_FIELDS},
        "material_stack": {part: as_text_list(stack.get(part)) for part in STACK_PARTS},
        "core_engineering_challenge": (data.get("core_engineering_challenge") or "").strip(),
    }
