"""Stage 8 — Review web app (FastAPI). Start it with: python -m lead_agent review

    GET  /                               the review page
    GET  /api/summary                    lead counts, profile version, unlearned decisions
    GET  /api/leads?status=...           leads, best score first
    POST /api/leads/{lead_id}/decision   {"decision": "accept" | "reject", "comment": "optional"}
    POST /api/learn                      Stage 9: update the Ideal Client Profile from decisions
    POST /api/discover                   Stages 6-7 in the background
    GET  /api/discover                   progress of that search
"""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from lead_agent.errors import LeadAgentError
from lead_agent.stages import stage9_learning
from lead_agent.stages.stage8_review import service

INDEX_PAGE = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="3A Lead Agent — Review")


class Decision(BaseModel):
    decision: Literal["accept", "reject"]
    comment: str = ""


@app.exception_handler(LeadAgentError)
async def explain_error(request: Request, error: LeadAgentError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.get("/", include_in_schema=False)
def index_page() -> FileResponse:
    return FileResponse(INDEX_PAGE)


@app.get("/api/summary")
def summary() -> dict:
    return service.get_summary()


@app.get("/api/leads")
def leads(status: str | None = None) -> list:
    return service.get_leads(status)


@app.post("/api/leads/{lead_id}/decision")
def decide(lead_id: str, body: Decision) -> dict:
    try:
        return service.record_decision(lead_id, body.decision, body.comment)
    except service.LeadNotFound:
        raise HTTPException(status_code=404, detail=f"Lead '{lead_id}' not found.")


@app.post("/api/learn")
def learn() -> dict:
    return stage9_learning.learn()


@app.post("/api/discover")
def start_discovery() -> dict:
    return service.start_discovery()


@app.get("/api/discover")
def discovery_status() -> dict:
    return service.discovery_status()
