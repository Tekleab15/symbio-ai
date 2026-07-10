"""
app/main.py

FastAPI serving layer for symbioAI.

Responsibilities:
- Bind a resilient HTTP API for Evaluation.
- Accept either a single task object or a batch/list of tasks.
- Route each task through app.router.SymbioRouter.
- Apply final canonicalization and optional sandbox verification.
- Return a stable JSON response:
    {"results": [{"task": ..., "answer": ..., "source": ...}]}

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import asyncio, json, logging, os, re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List, Sequence, Tuple

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

# default model values at import time.
os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
os.environ.setdefault("FIREWORKS_MODEL", "accounts/fireworks/models/gemma2-9b-it")
os.environ.setdefault("FIREWORKS_MODEL_CHEAP", "accounts/fireworks/models/gemma2-9b-it")
os.environ.setdefault("FIREWORKS_MODEL_FACTUAL", "accounts/fireworks/models/gemma2-9b-it")
os.environ.setdefault("FIREWORKS_MODEL_CODE", "accounts/fireworks/models/gemma2-27b-it")

from app.router import (  # noqa: E402
    RouteResult, SymbioRouter,
    TaskType, canonicalize_answer,
    get_router, run_sandboxed_python,
)

logger = logging.getLogger("symbioAI.main")

# Pydantic response models

class ResultItem(BaseModel):
    """One processed benchmark task result."""

    task: Any = Field(..., description="Original task object received by the API.")
    answer: str = Field(..., description="Canonicalized final answer.")
    source: str = Field(..., description="Route source: deterministic, cache, fireworks, etc.")

class ProcessResponse(BaseModel):
    """Standard response envelope expected by simple batch evaluators."""
    results: List[ResultItem]

class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    service: str
    cloud_enabled: bool
    fireworks_base_url: str
    cheap_model: str
    factual_model: str
    code_model: str

# App lifecycle

@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """
    Initialize the router once at process startup.

    The router lazy-loads the OpenAI-compatible Fireworks client, so this does
    not make a network call during startup. This keeps container boot fast and
    avoids failing health checks when FIREWORKS_API_KEY is mounted slightly
    later by the platform.
    """
    logging.basicConfig(
        level=os.getenv("SYMBIO_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    app_.state.router = get_router()
    logger.info(
        "symbioAI router initialized | base_url=%s | cheap=%s | factual=%s | code=%s",
        os.getenv("FIREWORKS_BASE_URL"),
        os.getenv("FIREWORKS_MODEL_CHEAP"),
        os.getenv("FIREWORKS_MODEL_FACTUAL"),
        os.getenv("FIREWORKS_MODEL_CODE"),
    )

    yield

    logger.info("symbioAI shutdown complete")

app = FastAPI(
    title="symbioAI",
    description="Hybrid token-efficient routing agent for AMD Developer Hackathon ACT II Track 1.",
    version="1.0.0",
    lifespan=lifespan,
)

# Request normalization

def _get_router_from_app() -> SymbioRouter:
    """Return the startup-initialized router, falling back to singleton."""
    router = getattr(app.state, "router", None)
    if isinstance(router, SymbioRouter):
        return router
    return get_router()

def _coerce_tasks(payload: Any) -> List[Any]:
    """
    Normalize accepted input shapes into a task list.
    Supported:
        1. Single task object:
            {"prompt": "What is 2+2?"}

        2. Direct batch:
            [{"prompt": "..."}, {"prompt": "..."}]

        3. Wrapped batch, for evaluator compatibility:
            {"tasks": [...]}
            {"inputs": [...]}
            {"queries": [...]}
            {"data": [...]}
            {"items": [...]}

        4. Raw scalar task, tolerated defensively:
            "What is 2+2?"
    """
    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("tasks", "inputs", "queries", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]

    if isinstance(payload, (str, int, float, bool)):
        return [payload]

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Payload must be a JSON object, JSON array, or wrapped task list.",
    )

def _task_to_promptish_text(task: Any) -> str:
    """Best-effort serialization for sandbox-replacement heuristics."""
    if isinstance(task, str):
        return task
    try:
        return json.dumps(task, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(task)

# Sandbox post-processing

_CODE_LIKE_RE = re.compile(
    r"```(?:python|py)?|"
    r"(^|\n)\s*(?:from\s+\w+|import\s+\w+|def\s+\w+\s*\(|class\s+\w+\s*[:(]|"
    r"print\s*\(|if\s+__name__\s*==|for\s+\w+\s+in\s+|while\s+.+:)",
    re.IGNORECASE,
)

_STDOUT_INTENT_PHRASES = (
    "what is the output",
    "what output",
    "what does this code print",
    "what will this code print",
    "stdout",
    "standard output",
    "print the result",
    "prints the result",
    "run the code",
    "execute the code",
    "calculate",
    "compute",
    "evaluate",
    "solve",
    "final numeric answer",
)

def _answer_looks_like_python(answer: str) -> bool:
    """Detect whether an answer likely contains executable Python."""
    if not answer:
        return False
    if _CODE_LIKE_RE.search(answer):
        return True

    # Very small generated snippets sometimes omit imports/defs but still form
    # obvious scripts.
    compact = answer.strip()
    return bool(
        "\n" in compact
        and any(token in compact for token in ("=", "+", "-", "*", "/", "print("))
        and not compact.lower().startswith(("the answer", "answer:"))
    )

def _should_replace_with_stdout(
    *,
    task_type: TaskType,
    task_promptish: str,
    stdout: str,
) -> bool:
    """
    Decide whether sandbox stdout should replace the router answer.

    For math tasks, stdout is almost always the final benchmark answer.

    For code-generation tasks, the code itself is often the expected answer.
    Therefore, stdout replacement is guarded unless the prompt asks for output
    execution, or an environment override explicitly enables replacement.
    """
    if not stdout.strip():
        return False

    if task_type == TaskType.MATH:
        return True

    prompt_low = task_promptish.lower()
    if any(phrase in prompt_low for phrase in _STDOUT_INTENT_PHRASES):
        return True

    return os.getenv("SYMBIO_REPLACE_CODEGEN_WITH_STDOUT", "0").strip() == "1"

async def _postprocess_with_sandbox(
    *,
    task: Any,
    route_result: RouteResult,
) -> Tuple[str, str]:
    """
    Canonicalize answer and optionally verify/execute Python output.

    Returns:
        (final_answer, final_source)
    """
    task_type = route_result.task_type
    source = route_result.source
    raw_answer = route_result.answer or ""

    final_answer = canonicalize_answer(raw_answer, task_type)
    final_source = source

    if task_type not in (TaskType.MATH, TaskType.CODE_GENERATION):
        return final_answer, final_source

    if not _answer_looks_like_python(raw_answer):
        return final_answer, final_source

    timeout_seconds = float(os.getenv("SYMBIO_SANDBOX_TIMEOUT_SECONDS", "2.0"))

    try:
        success, stdout, stderr = await asyncio.to_thread(
            run_sandboxed_python,
            raw_answer,
            timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Sandbox execution crashed: %s", exc)
        return final_answer, f"{source}+sandbox_error"

    if not success:
        logger.debug("Sandbox rejected answer | stderr=%s", stderr[:500])
        return final_answer, f"{source}+sandbox_rejected"

    stdout = stdout.strip()
    if not stdout:
        return final_answer, f"{source}+sandbox_verified"

    task_promptish = _task_to_promptish_text(task)
    if _should_replace_with_stdout(
        task_type=task_type,
        task_promptish=task_promptish,
        stdout=stdout,
    ):
        stdout_type = TaskType.MATH if task_type == TaskType.MATH else TaskType.GENERAL
        return canonicalize_answer(stdout, stdout_type), f"{source}+sandbox_stdout"

    return final_answer, f"{source}+sandbox_verified"

async def _process_one_task(task: Any) -> ResultItem:
    """Route, verify, canonicalize, and package one task."""
    router = _get_router_from_app()
    try:
        route_result = await router.route(task)
        answer, source = await _postprocess_with_sandbox(
            task=task,
            route_result=route_result,
        )

        # Final failsafe canonicalization pass.
        answer = canonicalize_answer(answer, route_result.task_type)

        return ResultItem(
            task=task,
            answer=answer,
            source=source,
        )

    except Exception as exc:
        logger.exception("Task processing failed: %s", exc)
        return ResultItem(
            task=task,
            answer="",
            source="server_error",
        )

# API endpoints

@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root health endpoint for simple container pings."""
    return await health()

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness endpoint."""
    cloud_disabled = os.getenv("SYMBIO_DISABLE_CLOUD", "0").strip() == "1"

    return HealthResponse(
        status="ok",
        service="symbioAI",
        cloud_enabled=not cloud_disabled and bool(os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_API_KEY")),
        fireworks_base_url=os.getenv("FIREWORKS_BASE_URL", ""),
        cheap_model=os.getenv("FIREWORKS_MODEL_CHEAP", ""),
        factual_model=os.getenv("FIREWORKS_MODEL_FACTUAL", ""),
        code_model=os.getenv("FIREWORKS_MODEL_CODE", ""),
    )

@app.post("/process", response_model=ProcessResponse)
async def process(request: Request) -> ProcessResponse:
    """
    Main evaluator endpoint.

    Accepts:
        - a single JSON task object
        - a JSON array of task objects
        - a wrapped batch such as {"tasks": [...]}

    Returns:
        {"results": [{"task": ..., "answer": ..., "source": ...}]}
    """
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload.",
        ) from exc

    tasks = _coerce_tasks(payload)

    max_batch_size = int(os.getenv("SYMBIO_MAX_BATCH_SIZE", "512"))
    if len(tasks) > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch too large. Maximum allowed tasks: {max_batch_size}.",
        )

    if not tasks:
        return ProcessResponse(results=[])

    results = await asyncio.gather(*(_process_one_task(task) for task in tasks))
    return ProcessResponse(results=results)

@app.post("/predict", response_model=ProcessResponse)
async def predict_alias(request: Request) -> ProcessResponse:
    """
    Compatibility alias for evaluators that use /predict.
    The official target remains /process.
    """
    return await process(request)

@app.post("/run", response_model=ProcessResponse)
async def run_alias(request: Request) -> ProcessResponse:
    """
    Compatibility alias for evaluators that use /run.
    The official target remains /process.
    """
    return await process(request)