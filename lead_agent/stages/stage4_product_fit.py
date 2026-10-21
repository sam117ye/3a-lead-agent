"""Stage 4 — Seed ↔ product fit (LLM).

Attaches sellable 3A products to a company, with one of two fit types:
    current_stack  they already look like a core / veil / kit buyer
                   (e.g. PET/PVC cores + infusion -> AIREX / ENGICORE / LANTOR)
    need_bridge    no disclosed core stack yet, but their needs overlap
                   (e.g. modular builds + recyclability -> ENGICORE / PET story)

The same matcher (match_products) is reused for new prospects in Stage 7.
"""
import logging

from lead_agent import prompts, settings, storage
from lead_agent.clients import llm
from lead_agent.utils import now_iso, run_parallel

log = logging.getLogger(__name__)

FIT_TYPES = ("current_stack", "need_bridge")
MAX_FITS = 5
TEXT_PREVIEW_LENGTH = 300  # characters of raw product text shown to the LLM


def build_product_fits(force: bool = False) -> list[dict]:
    """Match every normalized seed to 3A products: output/profiles/{seed_id}/product_fit.json."""
    taxonomy = storage.load_latest("taxonomy", "_normalized")
    taxonomy_file = storage.versioned_path("taxonomy", taxonomy["version"], "_normalized")
    seeds = storage.load_seed_files("normalized")
    todo = [seed for seed in seeds if force or _seed_needs_update(seed["seed_id"], taxonomy_file)]
    results = run_parallel(lambda seed: _save_seed_fits(seed, taxonomy), todo, settings.LLM_PARALLEL_CALLS)
    log.info("Stage 4: matched %d of %d seeds to products.", sum(r is not None for r in results), len(todo))
    return storage.load_seed_files("product_fit")


def match_products(company: dict, taxonomy: dict) -> list[dict]:
    """Ask the LLM which 3A products fit a company. Answers naming unknown products are dropped."""
    products = [product for product in taxonomy["products"] if product["active"] != "no"]
    prompt = prompts.render(
        "stage4_product_fit",
        company=company,
        products=[_product_for_prompt(product) for product in products],
        max_fits=MAX_FITS,
    )
    by_id = {product["product_id"].lower(): product for product in products}

    fits = []
    for fit in llm.ask_json(prompt).get("fits", []):
        product = by_id.get(str(fit.get("product_id") or "").strip().lower())
        if product is None or fit.get("fit_type") not in FIT_TYPES:
            continue  # an invented product or an unknown fit type
        if any(existing["product_id"] == product["product_id"] for existing in fits):
            continue
        fits.append(
            {
                "product_id": product["product_id"],
                "brand": product["brand"],
                "fit_type": fit["fit_type"],
                "confidence": _as_confidence(fit.get("confidence")),
                "why": str(fit.get("why") or "").strip(),
            }
        )
    return sorted(fits, key=lambda fit: fit["confidence"], reverse=True)[:MAX_FITS]


def _seed_needs_update(seed_id: str, taxonomy_file) -> bool:
    """Match again when the seed's tokens or the taxonomy changed since the last time."""
    output = storage.seed_file(seed_id, "product_fit")
    return not storage.is_up_to_date(output, storage.seed_file(seed_id, "normalized"), taxonomy_file)


def _save_seed_fits(seed: dict, taxonomy: dict) -> dict:
    document = {
        "seed_id": seed["seed_id"],
        "name": seed["name"],
        "created_at": now_iso(),
        "fits": match_products(seed, taxonomy),
    }
    storage.write_json(storage.seed_file(seed["seed_id"], "product_fit"), document)
    return document


def _product_for_prompt(product: dict) -> dict:
    """A compact product description: trimmed raw text plus its tokens."""
    return {
        "product_id": product["product_id"],
        "brand": product["brand"],
        "material": _preview(product["material_composition"]["raw"]),
        "mechanics": _preview(product["structural_mechanics"]["raw"]),
        "processing": _preview(product["engineering_tolerances"]["raw"]),
        "tokens": product["fingerprint"],
    }


def _preview(text: str | None) -> str:
    text = (text or "").strip()
    return text if len(text) <= TEXT_PREVIEW_LENGTH else text[:TEXT_PREVIEW_LENGTH] + "..."


def _as_confidence(value) -> float:
    """A number between 0 and 1 (0.5 when the LLM gave something else)."""
    try:
        return round(min(max(float(value), 0.0), 1.0), 2)
    except (TypeError, ValueError):
        return 0.5
