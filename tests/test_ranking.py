import unittest

from lead_agent import storage
from lead_agent.stages import stage7_ranking
from lead_agent.stages.stage7_ranking import total_score, weighted_overlap
from tests.fakes import OfflineTestCase


class ScoreTest(unittest.TestCase):
    def test_weighted_overlap(self):
        weights = {"WEIGHT_REDUCTION": 0.9, "RECYCLABILITY_SUSTAINABILITY": 0.6, "HIGH_STIFFNESS": 0.5}
        self.assertEqual(weighted_overlap(["WEIGHT_REDUCTION", "HIGH_STIFFNESS"], weights), 0.7)
        self.assertEqual(weighted_overlap([], weights), 0.0)
        self.assertEqual(weighted_overlap(["PET"], {}), 0.0)

    def test_total_score_subtracts_the_dislike_penalty(self):
        breakdown = {
            "need_overlap": 0.5,
            "stack_overlap": 0.5,
            "profile_similarity": 0.5,
            "exa_confidence": 1.0,
            "dislike_similarity": 0.25,
        }
        self.assertEqual(total_score(breakdown), 2.25)


class DislikeTest(OfflineTestCase):
    def test_a_look_alike_of_a_rejected_lead_is_penalized(self):
        rejected = {
            "name": "Alu Marine",
            "segment": "aluminium workboat builder",
            "country": "Netherlands",
            "selection_rationale": "Builds aluminium workboats.",
            "material_signals": {"raw": ["aluminium hulls"]},
            "need_signals": {"raw": []},
        }
        storage.write_json(
            storage.data_path(stage7_ranking.DISLIKED_EXAMPLES_FILE),
            [{"lead_id": "alumarine", "name": "Alu Marine", "comment": "aluminium only", "text": stage7_ranking.company_text(rejected)}],
        )
        look_alike = {**rejected, "name": "Alu Boats"}
        composite = {
            "name": "Foam Yachts",
            "segment": "composite sailing yacht builder",
            "country": "France",
            "selection_rationale": "Infuses PET foam sandwich hulls for light, recyclable yachts.",
            "material_signals": {"raw": ["PET foam", "vacuum infusion"]},
            "need_signals": {"raw": ["weight reduction"]},
        }
        profile = {"centroid_text": "Composite sailing yacht builder that infuses PET foam sandwich hulls to save weight."}

        look_alike_result, composite_result = stage7_ranking._similarities([look_alike, composite], profile)

        self.assertGreater(look_alike_result["dislike_similarity"], 0)
        self.assertEqual(look_alike_result["closest_rejected"], "Alu Marine")
        self.assertEqual(composite_result["dislike_similarity"], 0)


if __name__ == "__main__":
    unittest.main()
