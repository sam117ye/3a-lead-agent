"""Runs the stages in order. Every stage skips work that already exists unless force=True.

    1 Taxonomy -> 2 Seed profiles -> 3 Normalizer -> 4 Product fit -> 5 Ideal Client Profile
    -> 6 Discover -> 7 Match + rank -> 8 Review (web app) -> 9 Learn -> back to 6
"""
import logging

from lead_agent import storage
from lead_agent.stages import (
    stage1_taxonomy,
    stage2_seed_profiles,
    stage3_normalizer,
    stage4_product_fit,
    stage5_ideal_profile,
    stage6_discovery,
    stage7_ranking,
    stage9_learning,
)

log = logging.getLogger(__name__)


def build_profile(force: bool = False) -> dict:
    """Stages 1-5: from the 3A catalogue and the seed companies to the Ideal Client Profile."""
    stage1_taxonomy.build_taxonomy(force)
    stage2_seed_profiles.build_seed_profiles(force)
    stage3_normalizer.normalize_all(force)
    stage4_product_fit.build_product_fits(force)
    return stage5_ideal_profile.build_ideal_profile(force)


def discover(run_id: str | None = None) -> list[dict]:
    """Stages 6-7: find and rank new leads with the latest profile.

    New review decisions are learned first (Stage 9), so every search uses all feedback.
    Pass the id of an earlier run to rank its prospects again without a new Exa search.
    """
    profile = storage.load_latest(stage5_ideal_profile.PROFILE_KIND)
    if stage9_learning.unlearned_decisions(profile):
        stage9_learning.learn()
        profile = storage.load_latest(stage5_ideal_profile.PROFILE_KIND)

    if run_id is None:
        run_id = stage6_discovery.new_run_id()
        stage6_discovery.discover_prospects(profile, run_id)
    return stage7_ranking.rank_run(run_id)


def run_all(force: bool = False) -> list[dict]:
    """Stages 1-7: build whatever is missing, then find and rank new leads."""
    build_profile(force)
    return discover()
