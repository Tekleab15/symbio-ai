from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import CropCase, new_id, utc_now
from .orchestrator import orchestrator
from .settings import settings
from .storage import store

app = FastAPI(title="Symbio.AI Biosecurity Command", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

DEMO_CASES_PATH = Path(__file__).resolve().parent / "data" / "demo_cases.json"


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "service": "symbio-ai-backend",
        "mock_mode": settings.mock_mode,
        "band_enabled": settings.band_enabled,
        "time": utc_now(),
    }


@app.get("/api/demo-cases")
def demo_cases() -> list[dict[str, object]]:
    return json.loads(DEMO_CASES_PATH.read_text(encoding="utf-8"))


@app.post("/api/demo/{demo_id}/run")
async def run_demo_case(demo_id: str) -> dict[str, object]:
    demos = {item["case_id"]: item for item in demo_cases()}
    if demo_id not in demos:
        raise HTTPException(status_code=404, detail="Demo case not found")
    data = demos[demo_id]
    case = CropCase(
        case_id=data["case_id"],
        crop=data["crop"],
        location=data["location"],
        symptoms=data["symptoms"],
        urgency=data.get("urgency", "medium"),
        growth_stage=data.get("growth_stage", "unknown"),
        acreage=float(data.get("acreage", 0.0)),
        image_url=data.get("image_url"),
        metadata={"demo": True, "demo_title": data.get("title")},
    )
    store.clear_case_artifacts(case.case_id)
    store.save_case(case)
    return await orchestrator.run_case(case.case_id)


@app.post("/api/cases")
async def create_case(
    crop: Annotated[str, Form()],
    location: Annotated[str, Form()],
    symptoms: Annotated[str, Form()],
    urgency: Annotated[str, Form()] = "medium",
    growth_stage: Annotated[str, Form()] = "unknown",
    acreage: Annotated[float, Form()] = 0.0,
    image: UploadFile | None = File(default=None),
) -> dict[str, object]:
    case_id = new_id("sym")
    image_path = None
    image_url = None
    if image and image.filename:
        suffix = Path(image.filename).suffix or ".jpg"
        target = settings.upload_dir / f"{case_id}{suffix}"
        with target.open("wb") as f:
            shutil.copyfileobj(image.file, f)
        image_path = str(target)
        image_url = f"{settings.public_base_url.rstrip('/')}/uploads/{target.name}"
    symptom_list = [s.strip() for s in symptoms.replace("\n", ",").split(",") if s.strip()]
    case = CropCase(
        case_id=case_id,
        crop=crop,
        location=location,
        symptoms=symptom_list,
        urgency=urgency,
        growth_stage=growth_stage,
        acreage=acreage,
        image_path=image_path,
        image_url=image_url,
    )
    store.save_case(case)
    return case.to_dict()


@app.post("/api/cases/{case_id}/run")
async def run_case(case_id: str) -> dict[str, object]:
    try:
        return await orchestrator.run_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/cases")
def list_cases() -> list[dict[str, object]]:
    return store.list_cases()


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict[str, object]:
    try:
        return store.get_case(case_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/cases/{case_id}/events")
def get_events(case_id: str) -> list[dict[str, object]]:
    return store.list_events(case_id)


@app.get("/api/cases/{case_id}/band-transcript")
def get_band_transcript(case_id: str) -> list[dict[str, object]]:
    return store.list_band_records(case_id)


@app.get("/api/cases/{case_id}/report")
def get_report(case_id: str) -> dict[str, object]:
    try:
        case = store.get_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return case.final_report or {"case_id": case_id, "status": case.status, "message": "Report not generated yet."}


@app.get("/api/cases/{case_id}/report.html", response_class=HTMLResponse)
def get_report_html(case_id: str) -> str:
    report = get_report(case_id)
    return render_report_html(report)


@app.post("/api/cases/{case_id}/human-review")
def human_review(case_id: str, approved: bool = Form(...), reviewer: str = Form("Human Agronomist"), note: str = Form("")) -> dict[str, object]:
    try:
        case = store.get_case(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    report = case.final_report or {}
    report["human_review_status"] = "approved" if approved else "rejected"
    report["human_reviewer"] = reviewer
    report["human_review_note"] = note
    report["human_reviewed_at"] = utc_now()
    case.final_report = report
    case.status = "approved" if approved else "rejected"
    store.save_case(case)
    return case.to_dict()


def render_report_html(report: dict[str, object]) -> str:
    def section(title: str, content: object) -> str:
        return f"<h2>{title}</h2><pre>{json.dumps(content, indent=2, ensure_ascii=False)}</pre>"

    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Symbio.AI Audit Report</title>
  <style>
    body { font-family: Inter, Arial, sans-serif; margin: 40px; color: #111827; }
    h1 { color: #064e3b; }
    h2 { border-top: 1px solid #e5e7eb; padding-top: 18px; color: #065f46; }
    pre { background: #f8fafc; border: 1px solid #e5e7eb; padding: 14px; border-radius: 10px; white-space: pre-wrap; }
    .badge { display: inline-block; padding: 6px 10px; border-radius: 999px; background: #ecfdf5; color: #065f46; font-weight: 700; }
  </style>
</head>
<body>
  <span class=\"badge\">Symbio.AI Biosecurity Command</span>
  <h1>Audit Report: %s</h1>
  <p>%s</p>
  %s
</body>
</html>
""" % (
        report.get("case_id", "unknown"),
        report.get("executive_summary", "No summary available."),
        "".join(section(k, v) for k, v in report.items() if k not in {"case_id", "executive_summary"}),
    )