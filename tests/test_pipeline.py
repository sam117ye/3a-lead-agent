"""The whole loop, offline: stages 1-7, review decisions, learning, and a second search."""
import os
import time
import unittest

from lead_agent import pipeline, settings, storage
from lead_agent.stages import stage9_learning
from lead_agent.stages.stage8_review import service
from lead_agent.utils import domain_of
from tests.fakes import OfflineTestCase


class PipelineTest(OfflineTestCase):
    def test_full_loop(self):
        leads = pipeline.run_all()
        seeds = storage.read_seeds()

        # Stages 1-5 wrote every artifact of the plan's artifact map.
        taxonomy = storage.load_latest("taxonomy")
        self.assertEqual([p["product_id"] for p in taxonomy["products"]][:2], ["AIREX T92", "AIREX C70"])
        self.assertEqual(len(taxonomy["products"]), 5)  # the duplicate AIREX T92 was dropped
        self.assertTrue(storage.versioned_path("taxonomy", 1, "_normalized").exists())
        for seed in seeds:
            for name in ("manufacturer_profile", "normalized", "product_fit"):
                self.assertTrue(storage.seed_file(seed["seed_id"], name).exists(), f"{seed['seed_id']}/{name}")

        najad = storage.read_json(storage.seed_file("najad", "manufacturer_profile"))
        self.assertEqual(najad["manufacturer"]["website"], "https://najad.se/")  # Exa gave no website
        self.assertEqual(najad["material_stack"]["infusion_media"], [])  # Exa gave null

        fits = storage.read_json(storage.seed_file("najad", "product_fit"))["fits"]
        self.assertEqual([(f["product_id"], f["fit_type"]) for f in fits], [("AIREX T92", "current_stack"), ("AIREX C70", "need_bridge")])

        profile = storage.load_latest("ideal_client_profile")
        self.assertEqual(profile["version"], 1)
        self.assertEqual(profile["material_fingerprint"]["PET"], 1.0)
        self.assertEqual(profile["process_fingerprint"], {"VACUUM_INFUSION": 1.0})
        self.assertEqual(profile["brand_priority"][0]["brand"], "AIREX")
        self.assertEqual(len(profile["exclusions"]), len(seeds))

        # Stages 6-7: known companies and prospects without a website were skipped.
        self.assertEqual({lead["name"] for lead in leads}, {"Catana Catamarans", "Alu Marine", "Electric Day Boats"})
        self.assertEqual(leads[0]["name"], "Catana Catamarans")  # best score first
        catana = storage.read_json(storage.lead_path("catana-com"))
        self.assertEqual(catana["status"], "pending_review")
        self.assertEqual(catana["citations"], [{"url": "https://www.catana.com/en/", "title": "Catana"}])
        self.assertIn("VACUUM_INFUSION", catana["tokens"]["process"])
        alu = storage.read_json(storage.lead_path("alumarine-example"))
        self.assertEqual(alu["tokens"]["material"], ["ALUMINUM"])  # mapped by the LLM

        # Stage 8: review decisions.
        service.record_decision("alumarine-example", "reject", "Aluminium boats only")
        service.record_decision("catana-com", "accept")
        summary = service.get_summary()
        self.assertEqual(summary["counts"], {"pending_review": 1, "accepted": 1, "rejected": 1})
        self.assertEqual(summary["new_decisions"], 2)
        reviewed = service.get_leads("rejected")[0]
        self.assertEqual(reviewed["decision"]["comment"], "Aluminium boats only")

        # Stage 9: learning writes profile v2.
        result = stage9_learning.learn()
        self.assertTrue(result["updated"])
        updated = storage.load_latest("ideal_client_profile")
        self.assertEqual(updated["version"], 2)
        self.assertEqual([rule["rule"] for rule in updated["derived_rules"]], ["Builders whose hulls are aluminium only"])
        self.assertEqual(updated["must_not"], ["Aluminium-only builders"])
        self.assertEqual(len(updated["exclusions"]), len(seeds) + 2)
        self.assertEqual(len(updated["positive_examples"]), len(seeds) + 1)
        disliked = storage.read_json(storage.data_path("disliked_examples.json"))
        self.assertEqual([example["name"] for example in disliked], ["Alu Marine"])
        self.assertFalse(stage9_learning.learn()["updated"])  # nothing new the second time

        # A second search excludes seeds and every lead we already have.
        self.assertEqual(pipeline.discover(), [])
        excluded = {domain_of(entry["website"]) for entry in self.exa.calls[-1]["exclusion"]}
        self.assertTrue({"windelo-catamaran.com", "catana.com", "alumarine.example", "e-dayboats.example"} <= excluded)
        self.assertIn("Avoid aluminium-only yards.", self.exa.calls[-1]["query"])

    def test_stages_skip_finished_work(self):
        pipeline.build_profile()
        calls = len(self.exa.calls)
        pipeline.build_profile()
        self.assertEqual(len(self.exa.calls), calls)  # nothing researched twice
        self.assertEqual(storage.latest_version("ideal_client_profile"), 1)
        pipeline.build_profile(force=True)
        self.assertEqual(storage.latest_version("taxonomy"), 2)  # force builds a new version

    def test_a_changed_seed_list_rebuilds_the_profile(self):
        pipeline.build_profile()
        seeds_file = self.output_dir / "seeds.csv"
        seeds_file.write_text("company,website\nNajad,https://najad.se/\nNew Yard,https://new-yard.example\n")
        self.patch(settings, "SEED_COMPANIES_FILE", seeds_file)

        profile = pipeline.build_profile()

        self.assertEqual(profile["version"], 2)
        self.assertEqual(profile["based_on_seeds"], ["najad", "new-yard"])
        self.assertEqual(self.exa.calls[-1]["task"], "seed_new-yard")  # only the new seed was researched

    def test_edited_rules_are_applied_on_the_next_run(self):
        rules_file = self.output_dir / "rules.json"
        rules_file.write_text(settings.NORMALIZATION_RULES_FILE.read_text())
        self.patch(settings, "NORMALIZATION_RULES_FILE", rules_file)
        pipeline.build_profile()
        normalized = storage.seed_file("najad", "normalized")
        before = normalized.stat().st_mtime

        later = time.time() + 5
        os.utime(rules_file, (later, later))  # as if the rules were edited
        pipeline.build_profile()

        self.assertGreater(normalized.stat().st_mtime, before)
        self.assertEqual(storage.latest_version("ideal_client_profile"), 2)


class ReviewServiceTest(OfflineTestCase):
    def test_unknown_leads_and_decisions_are_refused(self):
        with self.assertRaises(service.LeadNotFound):
            service.record_decision("../secrets", "accept")
        with self.assertRaises(ValueError):
            service.record_decision("any", "maybe")


if __name__ == "__main__":
    unittest.main()
