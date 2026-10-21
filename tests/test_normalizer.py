import unittest

from lead_agent.stages.stage3_normalizer import (
    ATTRIBUTE_CATEGORIES,
    fingerprint,
    load_rulebook,
    normalize_documents,
)
from tests.fakes import OfflineTestCase


def match(text: str, attribute: str) -> list[str]:
    return load_rulebook().match(text, ATTRIBUTE_CATEGORIES[attribute])


class RuleMatchingTest(unittest.TestCase):
    def test_examples_from_the_plan(self):
        self.assertEqual(
            match("ArmaForm recycled PET foam, plus PVC foam kept where PET shear strength is not enough.", "material_stack.cores"),
            ["PET", "PVC"],
        )
        self.assertEqual(match("Vacuum infusion of sandwich composites", "manufacturing_processes"), ["VACUUM_INFUSION"])
        self.assertEqual(match("Vinylester resin", "material_stack.resins"), ["VINYLESTER"])
        self.assertEqual(match("Basalt fiber fabrics", "material_stack.reinforcements"), ["BASALT"])
        self.assertEqual(match("Polypropylene copolymer", "material_stack.cores"), ["PP"])
        self.assertEqual(match("BALTEK SBC / end-grain balsa", "material_stack.cores"), ["BALSA"])

    def test_whole_words_only(self):
        self.assertEqual(match("AIREX C70 is a closed-cell PVC core material", "material_composition"), ["PVC"])
        self.assertEqual(match("engine compartment", "manufacturing_processes"), [])
        self.assertEqual(match("Lantor core mats", "material_stack.infusion_media"), ["INFUSION_CORE_MAT"])

    def test_prefix_patterns_plurals_and_british_spelling(self):
        self.assertEqual(match("fully recyclable hull", "performance_mandates"), ["RECYCLABILITY_SUSTAINABILITY"])
        self.assertEqual(match("resists slamming loads", "performance_mandates"), ["HIGH_SHEAR_STRENGTH"])
        self.assertEqual(match("carbon fibre and E-glass", "material_stack.reinforcements"), ["CARBON_FIBER", "GLASS_FIBER"])
        self.assertEqual(match("hand lay-up and prepregs", "manufacturing_processes"), ["PREPREG", "WET_LAYUP"])

    def test_attributes_only_use_their_own_categories(self):
        self.assertEqual(match("VIP cabin forward", "material_stack.cores"), [])
        self.assertEqual(match("PET foam with high shear strength", "material_stack.cores"), ["PET"])


class LeftoverTest(OfflineTestCase):
    def test_leftovers_go_to_the_llm_and_only_allowed_tokens_are_kept(self):
        text = "Machined core panels, delivered as kits."  # no rule matches this
        documents = [{"material_composition": text, "structural_mechanics": "", "engineering_tolerances": None}]
        [result] = normalize_documents(documents, ["material_composition", "structural_mechanics", "engineering_tolerances"])
        # The fake LLM answers MODULAR_KITTING (a process, not allowed here) and PET (allowed).
        self.assertEqual(result["material_composition"], {"raw": text, "normalized": ["PET"]})
        self.assertEqual(result["structural_mechanics"]["normalized"], [])

    def test_nested_attributes_keep_their_structure(self):
        documents = [{"material_stack": {"cores": ["PVC foam"], "resins": ["epoxy"]}, "need_signals": ["lightweight"]}]
        [result] = normalize_documents(documents, ["material_stack.cores", "material_stack.resins", "need_signals"])
        self.assertEqual(result["material_stack"]["cores"], {"raw": ["PVC foam"], "normalized": ["PVC"]})
        self.assertEqual(result["material_stack"]["resins"]["normalized"], ["EPOXY"])
        self.assertEqual(fingerprint(result), {"material": ["PVC", "EPOXY"], "process": [], "need": ["WEIGHT_REDUCTION"]})


if __name__ == "__main__":
    unittest.main()
