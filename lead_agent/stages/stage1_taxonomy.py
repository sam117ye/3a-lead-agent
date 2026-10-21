"""Stage 1 — 3A product taxonomy (Exa Agent).

Builds the product dictionary from 3accorematerials.com and lantorcomposites.com.
Run rarely. Every new build is saved as the next version: output/data/taxonomy_vN.json.
"""
import logging

from lead_agent import prompts, settings, storage
from lead_agent.clients import exa
from lead_agent.utils import now_iso

log = logging.getLogger(__name__)

BRANDS = ["AIREX", "BALTEK", "LANTOR", "ENGICORE"]
RAW_TEXT_FIELDS = ["material_composition", "structural_mechanics", "engineering_tolerances"]

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Official product name, e.g. 'AIREX T92'."},
                    "brand": {"type": "string", "enum": BRANDS},
                    "active": {"type": "string", "enum": ["yes", "no", "unknown"], "description": "Currently offered?"},
                    "material_composition": {"type": "string", "description": "What it is made of (source wording)."},
                    "structural_mechanics": {"type": "string", "description": "Density, strength, stiffness... (source wording)."},
                    "engineering_tolerances": {"type": "string", "description": "Temperature limits, compatible resins and processes (source wording)."},
                },
                "required": ["product_id", "brand", "active", *RAW_TEXT_FIELDS],
            },
        }
    },
    "required": ["products"],
}


def build_taxonomy(force: bool = False) -> dict:
    """Research the 3A portfolio and save it as the next taxonomy version."""
    current_version = storage.latest_version("taxonomy")
    if current_version and not force:
        log.info("Stage 1: taxonomy v%d already exists (use --force for a new version).", current_version)
        return storage.read_json(storage.versioned_path("taxonomy", current_version))

    log.info("Stage 1: researching the 3A product portfolio with Exa...")
    run = exa.run_agent("taxonomy", prompts.render("stage1_taxonomy"), TAXONOMY_SCHEMA, settings.EXA_RUNS["taxonomy"])
    products = _clean_products(exa.structured_output(run).get("products"))

    version = current_version + 1
    taxonomy = {
        "version": version,
        "created_at": now_iso(),
        **exa.run_metadata(run),
        "products": products,
        "grounding": exa.grounding(run),
    }
    path = storage.write_json(storage.versioned_path("taxonomy", version), taxonomy)
    log.info("Stage 1: saved %d products to %s", len(products), path.name)
    return taxonomy


def _clean_products(products: list | None) -> list[dict]:
    """Drop empty or duplicate products and replace missing values (Exa may return nulls)."""
    cleaned, seen = [], set()
    for product in products or []:
        product_id = (product.get("product_id") or "").strip()
        if not product_id or product_id.lower() in seen:
            continue
        seen.add(product_id.lower())
        cleaned.append(
            {
                "product_id": product_id,
                "brand": (product.get("brand") or "").upper(),
                "active": product.get("active") or "unknown",
                **{field: (product.get(field) or "").strip() for field in RAW_TEXT_FIELDS},
            }
        )
    return cleaned
