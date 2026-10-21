"""Fake Exa runs and LLM answers, so the whole pipeline can be tested offline and for free."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lead_agent import settings
from lead_agent.clients import exa, llm

TAXONOMY_PRODUCTS = [
    {
        "product_id": "AIREX T92",
        "brand": "AIREX",
        "active": "yes",
        "material_composition": "Recycled PET foam core (polyethylene terephthalate).",
        "structural_mechanics": "Good shear strength and stiffness at low weight.",
        "engineering_tolerances": "Suitable for vacuum infusion and prepreg; thermoforming possible.",
    },
    {
        "product_id": "AIREX C70",
        "brand": "AIREX",
        "active": "yes",
        "material_composition": "Cross-linked PVC foam core.",
        "structural_mechanics": "High shear strength and resistance to slamming loads.",
        "engineering_tolerances": "Works with polyester, vinylester and epoxy resins in vacuum infusion.",
    },
    {
        "product_id": "BALTEK SBC",
        "brand": "BALTEK",
        "active": "yes",
        "material_composition": "End-grain balsa wood core.",
        "structural_mechanics": "Very high compressive strength and stiffness.",
        "engineering_tolerances": "Hand layup and resin infusion.",
    },
    {
        "product_id": "LANTOR Soric XF",
        "brand": "LANTOR",
        "active": "yes",
        "material_composition": "Polyester nonwoven infusion core mat with flow channels.",
        "structural_mechanics": "Adds laminate bulk with low resin uptake.",
        "engineering_tolerances": "Made for vacuum infusion and light RTM.",
    },
    {
        "product_id": "ENGICORE kits",
        "brand": "ENGICORE",
        "active": "yes",
        "material_composition": "Machined AIREX or BALTEK cores.",
        "structural_mechanics": "",
        "engineering_tolerances": "Pre-cut kits reduce labor and cycle time.",
    },
    {"product_id": "AIREX T92", "brand": "AIREX", "active": "yes"},  # a duplicate Exa might return
]

PROSPECTS = [
    {
        "name": "Catana Catamarans",
        "website": "https://www.catana.com",
        "country": "France",
        "segment": "sailing catamaran OEM",
        "selection_rationale": "Builds infused composite catamarans with foam cores.",
        "material_signals": ["vacuum infusion", "PVC foam core", "carbon fibre"],
        "need_signals": ["lightweight construction", "high stiffness"],
        "match_confidence": "high",
    },
    {
        "name": "Alu Marine",
        "website": "alumarine.example",
        "country": "Netherlands",
        "segment": "aluminium workboat builder",
        "selection_rationale": "Builds workboats.",
        "material_signals": ["aluminium hulls"],
        "need_signals": [],
        "match_confidence": "low",
    },
    {
        "name": "Windelo again",  # a seed: must be skipped
        "website": "https://windelo-catamaran.com/en/",
        "country": "France",
        "segment": "catamarans",
        "selection_rationale": "",
        "material_signals": [],
        "need_signals": [],
        "match_confidence": "high",
    },
    {
        "name": "No Website Boats",  # no website: must be skipped
        "website": "",
        "country": "Italy",
        "segment": "motor yachts",
        "selection_rationale": "",
        "material_signals": [],
        "need_signals": [],
        "match_confidence": "medium",
    },
    {
        "name": "Electric Day Boats",
        "website": "https://e-dayboats.example",
        "country": "Austria",
        "segment": "electric day boat builder",
        "selection_rationale": "Electric boats need every kilo saved for range.",
        "material_signals": ["epoxy resin", "wet layup"],
        "need_signals": ["weight reduction for battery range"],
        "match_confidence": "medium",
    },
]

DISCOVERY_GROUNDING = [
    {"field": "prospects[0].website", "citations": [{"url": "https://www.catana.com/en/", "title": "Catana"}], "confidence": "high"},
    {"field": "prospects[4].selection_rationale", "citations": [{"url": "https://news.example/e-boats"}], "confidence": "medium"},
]


def finished_run(run_id: str, structured: dict, grounding: list | None = None) -> dict:
    """A completed Exa Agent run, shaped like the real API response."""
    return {
        "id": run_id,
        "object": "agent_run",
        "status": "completed",
        "stopReason": "schema_satisfied",
        "output": {"text": "", "structured": structured, "grounding": grounding or []},
        "costDollars": {"total": 0.42},
    }


def seed_profile(name: str) -> dict:
    return {
        "manufacturer": {"name": name, "website": None, "location": "La Rochelle, France"},
        "end_products": ["Sailing catamarans 40-60 ft"],
        "manufacturing_processes": ["Vacuum infusion of hull and deck"],
        "material_stack": {
            "cores": ["PET foam core"],
            "reinforcements": ["E-glass"],
            "resins": ["vinylester"],
            "veils": [],
            "infusion_media": None,
        },
        "production_constraints": ["Shorter cycle time needed"],
        "performance_mandates": ["Weight reduction for sailing performance", "Fully recyclable materials"],
        "core_engineering_challenge": "Keep hulls stiff and light while moving to recyclable materials.",
    }


class FakeExa:
    """Replaces exa.run_agent and remembers every call."""

    def __init__(self):
        self.calls = []

    def run_agent(self, task_name, query, output_schema, run_settings, exclusion=None):
        self.calls.append({"task": task_name, "query": query, "exclusion": exclusion or []})
        if task_name == "taxonomy":
            return finished_run("agent_run_taxonomy", {"products": TAXONOMY_PRODUCTS})
        if task_name.startswith("seed_"):
            company = re.search(r"Company: (.+)", query).group(1)
            return finished_run(f"agent_run_{task_name}", seed_profile(company))
        if task_name == "discovery":
            return finished_run("agent_run_discovery", {"prospects": PROSPECTS}, DISCOVERY_GROUNDING)
        raise AssertionError(f"unexpected Exa task {task_name}")


# Words in leftover phrases and the tokens the fake LLM answers with (some are not allowed on purpose).
LEFTOVER_ANSWERS = {"aluminium": ["ALUMINUM"], "machined": ["MODULAR_KITTING", "PET"]}

ANSWERS = {
    "4": {
        "fits": [
            {"product_id": "AIREX T92", "fit_type": "current_stack", "confidence": 0.9, "why": "PET foam with infusion."},
            {"product_id": " airex c70", "fit_type": "need_bridge", "confidence": "0.4", "why": "Needs shear strength."},
            {"product_id": "MADE-UP CORE", "fit_type": "current_stack", "confidence": 1, "why": "Invented product."},
            {"product_id": "BALTEK SBC", "fit_type": "best_friend", "confidence": 0.7, "why": "Unknown fit type."},
        ]
    },
    "5": {
        "company_types": ["sailing catamaran OEM"],
        "must_have": ["Designs and builds its own boats (OEM)"],
        "centroid_text": "Catamaran builder using vacuum infusion with PET foam cores, glass fiber and vinylester, "
        "focused on weight reduction and recyclable materials.",
        "exa_search_brief": "Find composite boat builders that infuse sandwich hulls.",
    },
    "9": {
        "derived_rules": [
            {"type": "exclude", "rule": "Builders whose hulls are aluminium only", "lead_ids": ["alumarine-example"]},
            {"type": "maybe", "rule": "An unknown rule type is ignored"},
        ],
        "must_have": ["Designs and builds its own boats (OEM)"],
        "must_not": ["Aluminium-only builders"],
        "exa_search_brief": "Find composite boat builders. Avoid aluminium-only yards.",
        "change_summary": "Excluded aluminium-only builders.",
    },
}


def fake_ask_json(prompt: str) -> dict:
    """Replaces llm.ask_json: answers by the stage named on the prompt's first line."""
    match = re.match(r"# Stage (\d)", prompt)
    if match is None:
        return {"ok": True}
    stage = match.group(1)
    if stage == "3":
        phrases = json.loads(prompt.split("Phrases:")[1].split("For each phrase")[0])
        return {
            "results": [
                {"id": phrase["id"], "tokens": [token for word, tokens in LEFTOVER_ANSWERS.items() if word in phrase["text"].lower() for token in tokens]}
                for phrase in phrases
            ]
        }
    return ANSWERS[stage]


class TempOutputTestCase(unittest.TestCase):
    """Writes everything to a temporary output folder."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.output_dir = Path(temp_dir.name)
        self.patch(settings, "OUTPUT_DIR", self.output_dir)

    def patch(self, target, attribute: str, value) -> None:
        patcher = mock.patch.object(target, attribute, value)
        patcher.start()
        self.addCleanup(patcher.stop)


class OfflineTestCase(TempOutputTestCase):
    """Fake Exa and LLM, and word-count similarity: no API calls and no cost."""

    def setUp(self):
        super().setUp()
        self.exa = FakeExa()
        self.patch(settings, "OPENAI_API_KEY", "")
        self.patch(exa, "run_agent", self.exa.run_agent)
        self.patch(llm, "ask_json", fake_ask_json)
