"""Stage 7 — Match + rank.

Turns a discovery run into a scored review queue: output/leads/{lead_id}.json (status pending_review).

    score = need_overlap + stack_overlap + profile_similarity + exa_confidence - dislike_similarity
            (each part is between 0 and 1 and is multiplied by its weight in settings.SCORE_WEIGHTS)

    need_overlap        how much of the profile's need fingerprint the prospect shows, weighted by
                        how many seeds share each need
    stack_overlap       the same for materials and processes
    profile_similarity  text similarity between the prospect and the profile's centroid_text
    exa_confidence      Exa's match confidence: high 1.0, medium 0.6, low 0.3
    dislike_similarity  how much closer the prospect is to a rejected lead than to the ideal profile
                        (0 when it is not); rejected leads are collected by Stage 9
"""
import logging

from lead_agent import settings, storage
from lead_agent.clients.embeddings import cosine_similarity, embed
from lead_agent.errors import MissingInputError
from lead_agent.stages.stage3_normalizer import PROSPECT_ATTRIBUTES, fingerprint, normalize_documents
from lead_agent.stages.stage4_product_fit import match_products
from lead_agent.utils import domain_of, now_iso, run_parallel, slugify

log = logging.getLogger(__name__)

POSITIVE_PARTS = ["need_overlap", "stack_overlap", "profile_similarity", "exa_confidence"]
DISLIKED_EXAMPLES_FILE = "disliked_examples.json"


def rank_run(run_id: str) -> list[dict]:
    """Match, score and save the new prospects of a discovery run as leads. Returns them best first."""
    path = storage.prospects_path(run_id)
    if not path.exists():
        raise MissingInputError(f"No discovery run named {run_id} (expected {path}).")
    profile = storage.load_latest("ideal_client_profile")
    taxonomy = storage.load_latest("taxonomy", "_normalized")
    prospects = _only_new(storage.read_json(path)["prospects"])
    if not prospects:
        log.info("Stage 7: no new prospects to rank in %s.", run_id)
        return []

    normalized = normalize_documents(prospects, PROSPECT_ATTRIBUTES)
    companies = [_company(prospect, attributes) for prospect, attributes in zip(prospects, normalized)]
    fits = run_parallel(lambda company: match_products(company, taxonomy), companies, settings.LLM_PARALLEL_CALLS)
    similarities = _similarities(companies, profile)

    leads = [
        _make_lead(prospect, company, company_fits or [], similarity, profile, run_id)
        for prospect, company, company_fits, similarity in zip(prospects, companies, fits, similarities)
    ]
    for lead in leads:
        storage.write_json(storage.lead_path(lead["lead_id"]), lead)
    log.info("Stage 7: added %d leads to the review queue.", len(leads))
    return sorted(leads, key=lambda lead: lead["score"], reverse=True)


def weighted_overlap(tokens: list[str], weights: dict[str, float]) -> float:
    """Share of a fingerprint's total weight that the tokens cover (0..1)."""
    total = sum(weights.values())
    covered = sum(weight for token, weight in weights.items() if token in tokens)
    return round(covered / total, 3) if total else 0.0


def total_score(breakdown: dict) -> float:
    weights = settings.SCORE_WEIGHTS
    score = sum(weights[part] * breakdown[part] for part in POSITIVE_PARTS)
    return round(score - weights["dislike_similarity"] * breakdown["dislike_similarity"], 3)


def company_text(company: dict) -> str:
    """The text that represents a prospect or lead in similarity comparisons."""
    signals = company["material_signals"]["raw"] + company["need_signals"]["raw"]
    return ". ".join([company["name"], company["segment"], company["country"], company["selection_rationale"], *signals])


def load_disliked_examples() -> list[dict]:
    path = storage.data_path(DISLIKED_EXAMPLES_FILE)
    return storage.read_json(path) if path.exists() else []


# --- Helpers ------------------------------------------------------------------------------
def _only_new(prospects: list[dict]) -> list[dict]:
    """Drop prospects without a website and companies we already know (seeds and existing leads)."""
    known = {domain_of(seed["website"]) for seed in storage.read_seeds()}
    known |= {domain_of(lead["website"]) for lead in storage.load_leads()}
    new = []
    for prospect in prospects:
        domain = domain_of(prospect["website"])
        if domain and domain not in known:
            known.add(domain)
            new.append(prospect)
    if len(new) < len(prospects):
        log.info("Stage 7: skipped %d prospects that were already known or had no website.", len(prospects) - len(new))
    return new


def _company(prospect: dict, normalized: dict) -> dict:
    """What the product matcher sees: the prospect, its normalized signals and token fingerprint."""
    details = {key: value for key, value in prospect.items() if key != "citations"}
    return {**details, **normalized, "tokens": fingerprint(normalized)}


def _similarities(companies: list[dict], profile: dict) -> list[dict]:
    """profile_similarity and dislike_similarity for every company."""
    disliked = load_disliked_examples()
    texts = [profile["centroid_text"]] + [company_text(c) for c in companies] + [d["text"] for d in disliked]
    vectors = embed(texts)
    centroid = vectors[0]
    company_vectors = vectors[1 : 1 + len(companies)]
    disliked_vectors = vectors[1 + len(companies) :]

    results = []
    for vector in company_vectors:
        profile_similarity = cosine_similarity(vector, centroid)
        closest_similarity, closest_name = max(
            ((cosine_similarity(vector, d_vector), d["name"]) for d_vector, d in zip(disliked_vectors, disliked)),
            default=(0.0, None),
        )
        dislike_similarity = max(0.0, closest_similarity - profile_similarity)
        results.append(
            {
                "profile_similarity": profile_similarity,
                "dislike_similarity": dislike_similarity,
                "closest_rejected": closest_name if dislike_similarity > 0 else None,
            }
        )
    return results


def _make_lead(prospect: dict, company: dict, fits: list, similarity: dict, profile: dict, run_id: str) -> dict:
    tokens = company["tokens"]
    stack_weights = {**profile["material_fingerprint"], **profile["process_fingerprint"]}
    breakdown = {
        "need_overlap": weighted_overlap(tokens["need"], profile["need_fingerprint"]),
        "stack_overlap": weighted_overlap(tokens["material"] + tokens["process"], stack_weights),
        "profile_similarity": round(similarity["profile_similarity"], 3),
        "exa_confidence": settings.EXA_CONFIDENCE_SCORES.get(prospect["match_confidence"], 0.3),
        "dislike_similarity": round(similarity["dislike_similarity"], 3),
    }
    return {
        "lead_id": slugify(domain_of(prospect["website"])),
        "status": "pending_review",
        "name": prospect["name"],
        "website": prospect["website"],
        "country": prospect["country"],
        "segment": prospect["segment"],
        "selection_rationale": prospect["selection_rationale"],
        "material_signals": company["material_signals"],
        "need_signals": company["need_signals"],
        "match_confidence": prospect["match_confidence"],
        "citations": prospect["citations"],
        "tokens": tokens,
        "fits": fits,
        "score": total_score(breakdown),
        "score_breakdown": breakdown,
        "closest_rejected_lead": similarity["closest_rejected"],
        "run_id": run_id,
        "profile_version": profile["version"],
        "created_at": now_iso(),
    }
