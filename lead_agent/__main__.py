"""Command line for the 3A Lead Agent: python -m lead_agent <command>

Main commands
    check      test the API keys in .env
    run        stages 1-7: build whatever is missing, then find and rank new leads
    review     stage 8: open the review web app in your browser
    learn      stage 9: update the ideal client profile from your decisions
    discover   stages 6-7: find and rank more leads with the latest profile

Single stages (each skips work that already exists unless you add --force)
    taxonomy, seeds, normalize, fit, profile
"""
import argparse
import logging
import sys

from lead_agent import pipeline
from lead_agent.check import check_connections
from lead_agent.errors import ConfigError, LeadAgentError
from lead_agent.stages import (
    stage1_taxonomy,
    stage2_seed_profiles,
    stage3_normalizer,
    stage4_product_fit,
    stage5_ideal_profile,
    stage9_learning,
)

SINGLE_STAGES = {
    "taxonomy": (stage1_taxonomy.build_taxonomy, "stage 1: research the 3A product catalogue with Exa"),
    "seeds": (stage2_seed_profiles.build_seed_profiles, "stage 2: research the seed companies with Exa"),
    "normalize": (stage3_normalizer.normalize_all, "stage 3: turn the research text into comparable tokens"),
    "fit": (stage4_product_fit.build_product_fits, "stage 4: match the seed companies to 3A products"),
    "profile": (stage5_ideal_profile.build_ideal_profile, "stage 5: merge the seeds into the ideal client profile"),
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    try:
        args.handler(args)
    except LeadAgentError as error:
        logging.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logging.warning("Stopped. Run the same command again to continue where it left off.")
        return 130
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lead_agent",
        description="Find new clients for 3A Composites Core Materials.",
    )
    commands = parser.add_subparsers(title="commands", metavar="<command>", required=True)

    def add(name: str, handler, help_text: str, force: bool = False) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.set_defaults(handler=handler)
        if force:
            command.add_argument("--force", action="store_true", help="redo work that already exists")
        return command

    add("check", _check, "test the API keys in .env with cheap calls")
    add("run", _run, "stages 1-7: build whatever is missing, then find and rank new leads", force=True)
    for name, (stage, help_text) in SINGLE_STAGES.items():
        add(name, _run_single_stage, help_text, force=True).set_defaults(stage=stage)
    discover = add("discover", _discover, "stages 6-7: find and rank new leads with the latest profile")
    discover.add_argument("--run", metavar="RUN_ID", help="rank an earlier run's prospects again instead of searching")
    review = add("review", _review, "stage 8: open the review web app")
    review.add_argument("--port", type=int, default=8000, help="port for the web app (default 8000)")
    add("learn", _learn, "stage 9: update the ideal client profile from your review decisions")
    return parser


# --- Command handlers ------------------------------------------------------------------
def _check(args) -> None:
    results = check_connections()
    for result in results:
        print(f"  {'OK  ' if result['ok'] else 'FAIL'}  {result['name']}: {result['detail']}")
    if not all(result["ok"] for result in results):
        raise ConfigError("Fix the failing connection(s) in .env, then run the check again.")


def _run(args) -> None:
    _print_leads(pipeline.run_all(args.force))


def _run_single_stage(args) -> None:
    args.stage(args.force)


def _discover(args) -> None:
    _print_leads(pipeline.discover(args.run))


def _learn(args) -> None:
    result = stage9_learning.learn()
    print(result["change_summary"])
    for rule in result.get("new_rules", []):
        print(f"  {rule['type']}: {rule['rule']}")


def _review(args) -> None:
    import uvicorn  # imported here so the other commands work without the web packages

    from lead_agent.stages.stage8_review.app import app

    print(f"\nReview app running at http://127.0.0.1:{args.port}  (press Ctrl+C to stop)\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def _print_leads(leads: list[dict]) -> None:
    if not leads:
        print("\nNo new leads this time.")
        return
    print(f"\n{len(leads)} new leads, best first:")
    for lead in leads:
        products = ", ".join(fit["product_id"] for fit in lead["fits"][:2]) or "no product fit"
        print(f"  {lead['score']:5.2f}  {lead['name']} ({lead['country']}): {products}")
    print("\nReview them with: python -m lead_agent review")


if __name__ == "__main__":
    sys.exit(main())
