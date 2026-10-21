"""Stage 5 — Ideal Client Profile (merge logic + LLM text).

Compresses all seeds into one reusable "wanted poster" for discovery:
output/data/ideal_client_profile_v1.json. Stage 9 writes v2, v3, ... from review feedback.

Where each field comes from:
    company_types                      what the seeds build (LLM, from Stage 2 end products)
    material/process/need_fingerprint  Stage 3 tokens, with the share of seeds showing each one
    brand_priority                     union of the Stage 4 fits across seeds
    must_have, centroid_text,
    exa_search_brief                   LLM summary of the above
    must_not, derived_rules            start empty, grow from Stage 9 feedback
    exclusions                         the seeds (Stage 9 adds reviewed leads)
    positive_examples                  one line per seed (Stage 9 adds accepted leads)
"""
import logging
from collections import Counter, defaultdict

from lead_agent import prompts, storage
from lead_agent.clients import llm
from lead_agent.utils import as_text_list, now_iso

log = logging.getLogger(__name__)

PROFILE_KIND = "ideal_client_profile"
FINGERPRINT_GROUPS = ("material", "process", "need")


def build_ideal_profile(force: bool = False) -> dict:
    """Merge all seeds into an Ideal Client Profile, unless the latest version is up to date.

    A new version is built when the seed list or the seeds' product fits changed. It starts
    learning afresh: the next Stage 9 run learns from all review decisions again.
    """
    current_version = storage.latest_version(PROFILE_KIND)
    seeds = storage.load_seed_files("normalized")
    if current_version and not force and _is_up_to_date(current_version, seeds):
        log.info("Stage 5: ideal client profile v%d is up to date.", current_version)
        return storage.load_latest(PROFILE_KIND)

    fits_by_seed = {document["seed_id"]: document["fits"] for document in storage.load_seed_files("product_fit")}
    fingerprints = merge_fingerprints([seed["fingerprint"] for seed in seeds])
    brand_priority = rank_brands(fits_by_seed)
    text = llm.ask_json(
        prompts.render(
            "stage5_ideal_profile",
            seeds=[_seed_summary(seed, fits_by_seed.get(seed["seed_id"], [])) for seed in seeds],
            fingerprints=fingerprints,
            brand_priority=brand_priority,
        )
    )

    profile = {
        "version": current_version + 1,
        "created_at": now_iso(),
        "based_on_seeds": [seed["seed_id"] for seed in seeds],
        "company_types": as_text_list(text.get("company_types")),
        "must_have": as_text_list(text.get("must_have")),
        "must_not": [],
        "material_fingerprint": fingerprints["material"],
        "process_fingerprint": fingerprints["process"],
        "need_fingerprint": fingerprints["need"],
        "brand_priority": brand_priority,
        "centroid_text": str(text.get("centroid_text") or ""),
        "exa_search_brief": str(text.get("exa_search_brief") or ""),
        "positive_examples": [_seed_example(seed) for seed in seeds],
        "exclusions": [{"name": s["name"], "website": s["website"], "reason": "seed"} for s in storage.read_seeds()],
        "derived_rules": [],
        "learned_from_decisions": [],
    }
    path = storage.write_json(storage.versioned_path(PROFILE_KIND, profile["version"]), profile)
    log.info("Stage 5: saved %s", path.name)
    return profile


def merge_fingerprints(fingerprints: list[dict]) -> dict[str, dict[str, float]]:
    """Per group: token -> share of companies that show it (0..1), most common first."""
    merged = {}
    for group in FINGERPRINT_GROUPS:
        counts = Counter(token for fingerprint in fingerprints for token in fingerprint[group])
        merged[group] = {token: round(count / len(fingerprints), 2) for token, count in counts.most_common()}
    return merged


def rank_brands(fits_by_seed: dict[str, list[dict]]) -> list[dict]:
    """Brands ordered by how many seeds they fit, each with its most suggested products."""
    seeds_per_brand = defaultdict(set)
    products_per_brand = defaultdict(Counter)
    for seed_id, fits in fits_by_seed.items():
        for fit in fits:
            seeds_per_brand[fit["brand"]].add(seed_id)
            products_per_brand[fit["brand"]][fit["product_id"]] += 1
    ranked = sorted(seeds_per_brand, key=lambda brand: (-len(seeds_per_brand[brand]), brand))
    return [
        {
            "brand": brand,
            "seed_count": len(seeds_per_brand[brand]),
            "top_products": [product for product, _ in products_per_brand[brand].most_common(3)],
        }
        for brand in ranked
    ]


def _is_up_to_date(version: int, seeds: list[dict]) -> bool:
    """Same seeds as the latest profile, and no product fit newer than it."""
    path = storage.versioned_path(PROFILE_KIND, version)
    same_seeds = storage.read_json(path)["based_on_seeds"] == [seed["seed_id"] for seed in seeds]
    fit_files = [storage.seed_file(seed["seed_id"], "product_fit") for seed in seeds]
    return same_seeds and storage.is_up_to_date(path, *fit_files)


def example_line(name: str, website: str, builds: str, tokens: list[str]) -> str:
    """A one-line positive example for the Exa search, e.g. for a seed or an accepted lead."""
    signals = ", ".join(tokens) or "none disclosed"
    return f"{name} ({website}): builds {builds or 'boats'}; signals: {signals}"


def _seed_example(seed: dict) -> str:
    tokens = seed["fingerprint"]["material"] + seed["fingerprint"]["process"]
    return example_line(seed["name"], seed["manufacturer"]["website"], ", ".join(seed["end_products"][:3]), tokens)


def _seed_summary(seed: dict, fits: list[dict]) -> dict:
    return {
        "name": seed["name"],
        "location": seed["manufacturer"]["location"],
        "end_products": seed["end_products"],
        "core_engineering_challenge": seed["core_engineering_challenge"]["raw"],
        "tokens": seed["fingerprint"],
        "product_fits": [f"{fit['product_id']} ({fit['fit_type']})" for fit in fits],
    }
