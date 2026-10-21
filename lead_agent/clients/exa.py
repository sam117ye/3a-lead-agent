"""Exa Agent API: start a research run and wait for its result.

Docs: https://exa.ai/docs/reference/agent-api/create-a-run
A run is asynchronous: POST /agent/runs starts it and GET /agent/runs/{id} reports on it.
Ultra runs can take 30 minutes or more, so the run id is saved while we wait. If the
program stops, the next attempt picks up the same run instead of paying for a new one.
"""
import json
import logging
import time

from lead_agent import settings, storage
from lead_agent.clients.http import request_json
from lead_agent.errors import ApiError
from lead_agent.utils import now_iso

log = logging.getLogger(__name__)

BASE_URL = "https://api.exa.ai"
FINISHED_STATUSES = {"completed", "failed", "cancelled"}
HEARTBEAT_SECONDS = 300


def run_agent(task_name: str, query: str, output_schema: dict, run_settings: dict, exclusion: list[dict] | None = None) -> dict:
    """Run an Exa Agent task to completion and return the finished run.

    task_name     unique name for the task, used to resume an interrupted run (e.g. "seed_najad")
    run_settings  one entry of settings.EXA_RUNS (effort and budget)
    exclusion     records Exa should not return, e.g. [{"name": ..., "website": ...}]
    """
    pending_file = storage.pending_exa_run_path(task_name)
    if pending_file.exists():
        run_id = storage.read_json(pending_file)["run_id"]
        log.info("%s: resuming Exa run %s", task_name, run_id)
    else:
        run_id = _start_run(query, output_schema, run_settings, exclusion)
        storage.write_json(pending_file, {"run_id": run_id, "started_at": now_iso()})
        log.info("%s: started Exa run %s (effort: %s)", task_name, run_id, run_settings["effort"])

    try:
        run = _wait_until_finished(task_name, run_id)
    except ApiError as error:
        if error.status_code == 404:  # the run no longer exists: start a fresh one next time
            pending_file.unlink()
        raise

    pending_file.unlink()
    if run["status"] != "completed":
        raise ApiError(f"{task_name}: Exa run {run_id} ended as '{run['status']}' ({run.get('stopReason')}).")
    log.info("%s: Exa run finished (%s, $%.2f)", task_name, run.get("stopReason"), cost_dollars(run))
    return run


def list_runs(limit: int = 1) -> dict:
    """Recent runs. Costs nothing, so it is used to check the API key."""
    return request_json("GET", f"{BASE_URL}/agent/runs?limit={limit}", _headers(), timeout=60)


# --- Reading a finished run -----------------------------------------------------
def structured_output(run: dict) -> dict:
    """The answer shaped by the outputSchema (an empty dict when there is none)."""
    structured = (run.get("output") or {}).get("structured") or {}
    return json.loads(structured) if isinstance(structured, str) else structured


def grounding(run: dict) -> list[dict]:
    """Citations per output field: [{"field", "citations": [{"url", "title"}], "confidence"}]."""
    return (run.get("output") or {}).get("grounding") or []


def cost_dollars(run: dict) -> float:
    return (run.get("costDollars") or {}).get("total") or 0.0


def run_metadata(run: dict) -> dict:
    """Run details worth saving next to every Exa result."""
    return {"exa_run_id": run["id"], "exa_stop_reason": run.get("stopReason"), "exa_cost_dollars": cost_dollars(run)}


# --- Internals ---------------------------------------------------------------------
def _start_run(query: str, output_schema: dict, run_settings: dict, exclusion: list[dict] | None) -> str:
    body = {"query": query, "effort": run_settings["effort"], "outputSchema": output_schema}
    budget = _budget(run_settings)
    if budget:
        body["budget"] = budget
    if exclusion:
        body["input"] = {"exclusion": exclusion}
    run = request_json("POST", f"{BASE_URL}/agent/runs", _headers(), body, timeout=120)
    return run["id"]


def _budget(run_settings: dict) -> dict | None:
    """Exa accepts a cost cap only for "auto" and "ultra" runs, and a time cap only for "ultra"."""
    effort = run_settings["effort"]
    if effort not in ("auto", "ultra"):
        return None
    budget = {"maxCostDollars": run_settings["max_cost_dollars"]}
    if effort == "ultra" and run_settings.get("max_duration_seconds"):
        budget["maxDurationSeconds"] = run_settings["max_duration_seconds"]
    return budget


def _wait_until_finished(task_name: str, run_id: str) -> dict:
    started = time.monotonic()
    last_heartbeat = started
    while True:
        run = request_json("GET", f"{BASE_URL}/agent/runs/{run_id}", _headers(), timeout=60)
        if run["status"] in FINISHED_STATUSES:
            return run
        if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
            minutes = int((time.monotonic() - started) / 60)
            log.info("%s: Exa run still %s (%d min so far)", task_name, run["status"], minutes)
            last_heartbeat = time.monotonic()
        time.sleep(settings.EXA_POLL_SECONDS)


def _headers() -> dict:
    return {"x-api-key": settings.require("EXA_API_KEY")}
