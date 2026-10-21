"""Stage 3 — Attribute normalizer (rules first, LLM only for leftovers).

Turns long phrases from Exa (Stages 1, 2 and 6) into comparable tokens. For a "cores" attribute:
    "ArmaForm recycled PET foam, plus PVC foam kept where PET shear strength is not enough."
    -> ["PET", "PVC"]

How it works
1. Rules come from inputs/normalization_rules.json. A pattern matches whole words, ignoring
   case and allowing a plural "s"/"es". A pattern ending in "*" matches the start of a word
   ("recyclab*" -> "recyclable", "recyclability"). British spellings count as American ones.
2. Each attribute only uses the rule categories that make sense for it, so a list of core
   materials can only produce core tokens (a "VIP cabin" is never read as vacuum infusion).
3. Phrases that no rule matches ("leftovers") go to the LLM in one batch. It may only answer
   with tokens allowed for that phrase's attribute.

The result keeps the input's structure, with {"raw": ..., "normalized": [...]} per attribute.
The normalizer only simplifies attributes: it does not pick products or rewrite profiles.
"""
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from lead_agent import prompts, settings, storage
from lead_agent.clients import llm
from lead_agent.utils import as_text_list, now_iso

log = logging.getLogger(__name__)

# Canonical tokens that no rule produces (ALUMINUM, STEEL, TIMBER, BIO_BASED) count as materials.
OTHER_MATERIALS = "materials"
MATERIAL_CATEGORIES = ["cores", "reinforcements", "resins", "veils", OTHER_MATERIALS]

# The rule categories each attribute may produce tokens from.
ATTRIBUTE_CATEGORIES = {
    # Stage 1: taxonomy products
    "material_composition": MATERIAL_CATEGORIES,
    "structural_mechanics": ["needs"],
    "engineering_tolerances": ["processes", "needs"],
    # Stage 2: seed manufacturing profiles
    "manufacturing_processes": ["processes"],
    "material_stack.cores": ["cores", OTHER_MATERIALS],
    "material_stack.reinforcements": ["reinforcements", OTHER_MATERIALS],
    "material_stack.resins": ["resins", OTHER_MATERIALS],
    "material_stack.veils": ["veils"],
    "material_stack.infusion_media": ["veils"],
    "production_constraints": ["needs"],
    "performance_mandates": ["needs"],
    "core_engineering_challenge": ["needs"],
    # Stage 6: discovered prospects
    "material_signals": [*MATERIAL_CATEGORIES, "processes"],
    "need_signals": ["needs"],
}
TAXONOMY_ATTRIBUTES = ["material_composition", "structural_mechanics", "engineering_tolerances"]
SEED_ATTRIBUTES = [
    "manufacturing_processes",
    "material_stack.cores",
    "material_stack.reinforcements",
    "material_stack.resins",
    "material_stack.veils",
    "material_stack.infusion_media",
    "production_constraints",
    "performance_mandates",
    "core_engineering_challenge",
]
PROSPECT_ATTRIBUTES = ["material_signals", "need_signals"]

# A fingerprint groups a company's tokens by what they describe.
FINGERPRINT_GROUPS = {"material": MATERIAL_CATEGORIES, "process": ["processes"], "need": ["needs"]}

SPELLING_VARIANTS = {"fibre": "fiber", "mould": "mold", "labour": "labor", "lay-up": "layup"}
LEFTOVER_BATCH_SIZE = 40


# --- Stage entry points ---------------------------------------------------------------
def normalize_all(force: bool = False) -> None:
    normalize_taxonomy(force)
    normalize_seed_profiles(force)


def normalize_taxonomy(force: bool = False) -> dict:
    """Normalize the latest taxonomy into output/data/taxonomy_vN_normalized.json."""
    taxonomy = storage.load_latest("taxonomy")
    path = storage.versioned_path("taxonomy", taxonomy["version"], "_normalized")
    source = storage.versioned_path("taxonomy", taxonomy["version"])
    if not force and storage.is_up_to_date(path, source, settings.NORMALIZATION_RULES_FILE):
        log.info("Stage 3: %s is up to date.", path.name)
        return storage.read_json(path)

    normalized = normalize_documents(taxonomy["products"], TAXONOMY_ATTRIBUTES)
    products = [
        {
            "product_id": product["product_id"],
            "brand": product["brand"],
            "active": product["active"],
            **attributes,
            "fingerprint": fingerprint(attributes),
        }
        for product, attributes in zip(taxonomy["products"], normalized)
    ]
    normalized_taxonomy = {"version": taxonomy["version"], "created_at": now_iso(), "products": products}
    storage.write_json(path, normalized_taxonomy)
    log.info("Stage 3: normalized %d products into %s", len(products), path.name)
    return normalized_taxonomy


def normalize_seed_profiles(force: bool = False) -> list[dict]:
    """Normalize every seed profile into output/profiles/{seed_id}/normalized.json."""
    profiles = storage.load_seed_files("manufacturer_profile")
    todo = [profile for profile in profiles if force or _seed_needs_update(profile["seed_id"])]
    for profile, attributes in zip(todo, normalize_documents(todo, SEED_ATTRIBUTES)):
        document = {
            "seed_id": profile["seed_id"],
            "name": profile["manufacturer"]["name"],
            "manufacturer": profile["manufacturer"],
            "end_products": profile["end_products"],
            **attributes,
            "fingerprint": fingerprint(attributes),
        }
        storage.write_json(storage.seed_file(profile["seed_id"], "normalized"), document)
    log.info("Stage 3: normalized %d seed profiles.", len(todo))
    return storage.load_seed_files("normalized")


def _seed_needs_update(seed_id: str) -> bool:
    """Normalize again when the profile or the rules changed since the last time."""
    output = storage.seed_file(seed_id, "normalized")
    profile = storage.seed_file(seed_id, "manufacturer_profile")
    return not storage.is_up_to_date(output, profile, settings.NORMALIZATION_RULES_FILE)


# --- Normalizing ------------------------------------------------------------------
def normalize_documents(documents: list[dict], attributes: list[str]) -> list[dict]:
    """Normalize the given attributes of many documents at once.

    Returns one dict per document with the same nested shape, where each attribute becomes
    {"raw": original value, "normalized": [tokens]}. Leftovers of all documents go to the LLM together.
    """
    rulebook = load_rulebook()
    tokens = [{attribute: [] for attribute in attributes} for _ in documents]
    leftovers = []  # (document index, attribute, text)
    for index, document in enumerate(documents):
        for attribute in attributes:
            for text in as_text_list(_get(document, attribute)):
                found = rulebook.match(text, ATTRIBUTE_CATEGORIES[attribute])
                if found:
                    tokens[index][attribute] += found
                else:
                    leftovers.append((index, attribute, text))

    for (index, attribute, _), found in zip(leftovers, _map_leftovers_with_llm(leftovers, rulebook)):
        tokens[index][attribute] += found

    return [
        _nest(
            {
                attribute: {"raw": _get(document, attribute), "normalized": rulebook.sort(found)}
                for attribute, found in document_tokens.items()
            }
        )
        for document, document_tokens in zip(documents, tokens)
    ]


def fingerprint(normalized: dict) -> dict[str, list[str]]:
    """All tokens of a normalized document, grouped into material / process / need."""
    rulebook = load_rulebook()
    tokens = rulebook.sort(_all_tokens(normalized))
    return {
        group: [token for token in tokens if rulebook.category_of.get(token) in categories]
        for group, categories in FINGERPRINT_GROUPS.items()
    }


# --- The rule dictionary ------------------------------------------------------------
@dataclass
class Rule:
    category: str
    pattern: re.Pattern
    tokens: list[str]


class RuleBook:
    """The normalization dictionary: compiled rules plus the canonical token list."""

    def __init__(self, rules_file):
        data = storage.read_json(rules_file)
        self.ontology: list[str] = data["canonical_ontology"]
        self.rules = [
            Rule(rule["category"], _compile(pattern), rule["tokens"])
            for rule in data["rules"]
            for pattern in rule["patterns"]
        ]
        self.category_of = {token: rule.category for rule in self.rules for token in rule.tokens}
        for token in self.ontology:
            self.category_of.setdefault(token, OTHER_MATERIALS)

    def match(self, text: str, categories: list[str]) -> list[str]:
        """Tokens of every rule in the given categories whose pattern occurs in the text."""
        text = _american_spelling(text.lower())
        found = [
            token
            for rule in self.rules
            if rule.category in categories and rule.pattern.search(text)
            for token in rule.tokens
        ]
        return self.sort(found)

    def allowed_tokens(self, attribute: str) -> list[str]:
        categories = ATTRIBUTE_CATEGORIES[attribute]
        return [token for token in self.ontology if self.category_of[token] in categories]

    def sort(self, tokens: list[str]) -> list[str]:
        """Unique tokens in canonical-ontology order."""
        unique = set(tokens)
        return [token for token in self.ontology if token in unique] + sorted(unique - set(self.ontology))


@lru_cache(maxsize=1)
def load_rulebook() -> RuleBook:
    return RuleBook(settings.NORMALIZATION_RULES_FILE)


def _compile(pattern: str) -> re.Pattern:
    """Whole words (plural allowed), or the start of a word when the pattern ends with "*"."""
    pattern = _american_spelling(pattern.lower().strip())
    if pattern.endswith("*"):
        return re.compile(r"\b" + re.escape(pattern[:-1]))
    return re.compile(r"\b" + re.escape(pattern) + r"(?:s|es)?\b")


def _american_spelling(text: str) -> str:
    for british, american in SPELLING_VARIANTS.items():
        text = text.replace(british, american)
    return text


# --- LLM for leftovers ------------------------------------------------------------------
def _map_leftovers_with_llm(leftovers: list[tuple], rulebook: RuleBook) -> list[list[str]]:
    """Ask the LLM to tokenize phrases no rule matched. Returns one token list per leftover."""
    results = []
    for start in range(0, len(leftovers), LEFTOVER_BATCH_SIZE):
        batch = leftovers[start : start + LEFTOVER_BATCH_SIZE]
        attributes = sorted({attribute for _, attribute, _ in batch})
        prompt = prompts.render(
            "stage3_normalize_leftovers",
            allowed_tokens={attribute: rulebook.allowed_tokens(attribute) for attribute in attributes},
            phrases=[{"id": i, "attribute": attribute, "text": text} for i, (_, attribute, text) in enumerate(batch)],
        )
        answer = llm.ask_json(prompt)
        tokens_by_id = {str(item.get("id")): item.get("tokens") or [] for item in answer.get("results", [])}
        for i, (_, attribute, _) in enumerate(batch):
            allowed = rulebook.allowed_tokens(attribute)
            results.append([token for token in tokens_by_id.get(str(i), []) if token in allowed])
    if leftovers:
        log.info("Stage 3: %d phrase(s) matched no rule and were mapped by the LLM.", len(leftovers))
    return results


# --- Helpers ---------------------------------------------------------------------------
def _get(document: dict, attribute: str):
    """Read a dotted attribute: _get(doc, "material_stack.cores")."""
    value = document
    for key in attribute.split("."):
        value = (value or {}).get(key)
    return value


def _nest(flat: dict) -> dict:
    """{"material_stack.cores": x} -> {"material_stack": {"cores": x}}"""
    nested = {}
    for attribute, value in flat.items():
        *parents, last = attribute.split(".")
        target = nested
        for key in parents:
            target = target.setdefault(key, {})
        target[last] = value
    return nested


def _all_tokens(node) -> list[str]:
    if not isinstance(node, dict):
        return []
    if "normalized" in node:
        return list(node["normalized"])
    return [token for child in node.values() for token in _all_tokens(child)]
