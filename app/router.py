"""
app/router.py

SymbioAI Track 1 router:
- Deterministic zero-token interception for safe arithmetic and structural extraction.
- Micro-prompted Fireworks fallback with tight max_tokens and reasoning suppression.
- Strict answer canonicalization helpers for benchmark-style outputs.
- Async, cache-aware routing with robust error handling.

Dependencies:
    openai
    pydantic

Environment variables:
    FIREWORKS_API_KEY                  Required for cloud fallback.
    FIREWORKS_BASE_URL                 Defaults to Fireworks OpenAI-compatible endpoint.
    FIREWORKS_MODEL                    Global fallback model.
    FIREWORKS_MODEL_CHEAP              Cheap general/sentiment/NER model.
    FIREWORKS_MODEL_FACTUAL            Factual QA model override.
    FIREWORKS_MODEL_CODE               Code model override.
    FIREWORKS_MODEL_GEMMA              Optional Gemma model override.
    SYMBIO_DISABLE_CLOUD               "1" disables Fireworks calls.
    SYMBIO_ENABLE_CACHE                Defaults to "1".
    SYMBIO_TIMEOUT_SECONDS             Defaults to 20.
    SYMBIO_MAX_CONCURRENCY             Defaults to 8.
"""

from __future__ import annotations

import ast, asyncio, contextlib, hashlib, json, logging, math, os, re, subprocess, sys, tempfile, textwrap
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger("symbioAI.router")

# Task categories

class TaskType(str, Enum):
    MATH = "math"
    SENTIMENT = "sentiment"
    NER = "ner"
    SUMMARIZATION = "summarization"
    FACTUAL_QA = "factual_qa"
    LOGIC = "logic"
    CODE_GENERATION = "code_generation"
    CODE_DEBUGGING = "code_debugging"
    STRUCTURAL_EXTRACTION = "structural_extraction"
    GENERAL = "general"

# Data structures

@dataclass(frozen=True)
class ModelProfile:
    """Cloud model/runtime profile for a task type."""
    name: str
    model_env: str
    default_model: str
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    reasoning_effort: Optional[Any] = "none"
    system_prompt: str = (
        "You are symbioAI, a terse benchmark answer engine. "
        "Return only the final answer. No explanation. No markdown."
    )

@dataclass
class RouteResult:
    """Result returned by the router."""
    answer: str
    task_type: TaskType
    source: str  # deterministic | cache | fireworks | fallback_error
    confidence: float = 0.0
    model: Optional[str] = None
    raw_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeterministicHit:
    """A safe zero-token answer produced locally."""

    answer: str
    task_type: TaskType
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# Model profiles and prompt profiles

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

DEFAULT_MODEL = os.getenv(
    "FIREWORKS_MODEL",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
)

DEFAULT_CHEAP_MODEL = os.getenv("FIREWORKS_MODEL_CHEAP", DEFAULT_MODEL)
DEFAULT_FACTUAL_MODEL = os.getenv("FIREWORKS_MODEL_FACTUAL", DEFAULT_CHEAP_MODEL)
DEFAULT_CODE_MODEL = os.getenv("FIREWORKS_MODEL_CODE", DEFAULT_MODEL)
DEFAULT_GEMMA_MODEL = os.getenv("FIREWORKS_MODEL_GEMMA", DEFAULT_CHEAP_MODEL)

MODEL_PROFILES: Dict[TaskType, ModelProfile] = {
    TaskType.SENTIMENT: ModelProfile(
        name="sentiment",
        model_env="FIREWORKS_MODEL_CHEAP",
        default_model=DEFAULT_CHEAP_MODEL,
        max_tokens=3,
        reasoning_effort="none",
    ),
    TaskType.NER: ModelProfile(
        name="ner",
        model_env="FIREWORKS_MODEL_CHEAP",
        default_model=DEFAULT_CHEAP_MODEL,
        max_tokens=96,
        reasoning_effort="none",
    ),
    TaskType.STRUCTURAL_EXTRACTION: ModelProfile(
        name="structural_extraction",
        model_env="FIREWORKS_MODEL_CHEAP",
        default_model=DEFAULT_CHEAP_MODEL,
        max_tokens=96,
        reasoning_effort="none",
    ),
    TaskType.MATH: ModelProfile(
        name="math",
        model_env="FIREWORKS_MODEL_CHEAP",
        default_model=DEFAULT_CHEAP_MODEL,
        max_tokens=48,
        reasoning_effort="low",
    ),
    TaskType.LOGIC: ModelProfile(
        name="logic",
        model_env="FIREWORKS_MODEL_CHEAP",
        default_model=DEFAULT_CHEAP_MODEL,
        max_tokens=80,
        reasoning_effort="low",
    ),
    TaskType.FACTUAL_QA: ModelProfile(
        name="factual_qa",
        model_env="FIREWORKS_MODEL_FACTUAL",
        default_model=DEFAULT_FACTUAL_MODEL,
        max_tokens=64,
        reasoning_effort="none",
    ),
    TaskType.SUMMARIZATION: ModelProfile(
        name="summarization",
        model_env="FIREWORKS_MODEL_CHEAP",
        default_model=DEFAULT_CHEAP_MODEL,
        max_tokens=120,
        reasoning_effort="none",
    ),
    TaskType.CODE_GENERATION: ModelProfile(
        name="code_generation",
        model_env="FIREWORKS_MODEL_CODE",
        default_model=DEFAULT_CODE_MODEL,
        max_tokens=320,
        reasoning_effort="low",
        system_prompt=(
            "You are symbioAI, a terse coding engine. "
            "Return only the requested code or final answer. "
            "No markdown fences. No explanation."
        ),
    ),
    TaskType.CODE_DEBUGGING: ModelProfile(
        name="code_debugging",
        model_env="FIREWORKS_MODEL_CODE",
        default_model=DEFAULT_CODE_MODEL,
        max_tokens=320,
        reasoning_effort="low",
        system_prompt=(
            "You are symbioAI, a terse code repair engine. "
            "Return only corrected code or the requested final output. "
            "No markdown fences. No explanation."
        ),
    ),
    TaskType.GENERAL: ModelProfile(
        name="general",
        model_env="FIREWORKS_MODEL_CHEAP",
        default_model=DEFAULT_CHEAP_MODEL,
        max_tokens=96,
        reasoning_effort="none",
    ),
}

# Text and prompt utilities

_WHITESPACE_RE = re.compile(r"\s+")
_CODE_FENCE_RE = re.compile(r"```(?:python|py|json|javascript|js|java|cpp|c\+\+|c|go|bash|sh)?\s*([\s\S]*?)```", re.IGNORECASE)
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")

def compact_text(text: Any, max_chars: int = 8000) -> str:
    """Normalize whitespace and trim very long inputs defensively."""
    if text is None:
        return ""
    s = str(text).replace("\x00", " ")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    if len(s) > max_chars:
        return s[:max_chars].rstrip()
    return s

def stable_prompt_hash(text: str) -> str:
    """Stable cache key for normalized prompts."""
    normalized = compact_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def strip_code_fences(text: str) -> str:
    """Removing common markdown code fences while preserving code content."""
    if not text:
        return ""
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        return "\n\n".join(m.strip() for m in matches if m.strip()).strip()
    return text.strip()

def first_json_like(text: str) -> Optional[str]:
    """Extract the first JSON array/object-like substring from model output."""
    if not text:
        return None
    array_match = _JSON_ARRAY_RE.search(text)
    if array_match:
        return array_match.group(0).strip()
    object_match = _JSON_OBJECT_RE.search(text)
    if object_match:
        return object_match.group(0).strip()
    return None

def remove_common_preambles(text: str) -> str:
    """Remove verbose LLM preambles that commonly hurt exact-match benchmarks."""
    s = text.strip()
    s = re.sub(r"^(the\s+answer\s+is|answer:|final\s+answer:|result:)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+$", "", s)
    return s.strip()

def normalize_output_text(text: Any) -> str:
    """General output cleanup."""
    s = "" if text is None else str(text)
    s = s.replace("\x00", "").strip()
    s = strip_code_fences(s)
    s = remove_common_preambles(s)
    s = s.strip()

    # Remove wrapping quotes for simple scalar answers.
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()

    return s

# Task extraction and categorization

def extract_prompt(task: Any) -> str:
    """
    Extract useful prompt text from flexible task structures.

    Supports:
        - raw string
        - dict with prompt/question/input/text/instruction/code fields
        - arbitrary object converted to string
    """
    if isinstance(task, str):
        return task.strip()

    if isinstance(task, Mapping):
        preferred_keys = (
            "prompt",
            "question",
            "query",
            "instruction",
            "task",
            "input",
            "text",
            "content",
            "problem",
            "code",
        )

        chunks: List[str] = []
        for key in preferred_keys:
            value = task.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                chunks.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                chunks.append(str(value))

        # Include type/category hints if present.
        for key in ("type", "category", "task_type"):
            value = task.get(key)
            if value:
                chunks.insert(0, f"{key}: {value}")

        if chunks:
            return "\n".join(chunk.strip() for chunk in chunks if chunk is not None).strip()

        return json.dumps(task, ensure_ascii=False)

    return str(task).strip()

def explicit_task_type(task: Any) -> Optional[TaskType]:
    """Read task type hints if the benchmark supplies them."""
    if not isinstance(task, Mapping):
        return None

    raw = str(
        task.get("type")
        or task.get("category")
        or task.get("task_type")
        or task.get("label")
        or ""
    ).strip().lower()

    if not raw:
        return None

    aliases = {
        "sentiment": TaskType.SENTIMENT,
        "sentiment_classification": TaskType.SENTIMENT,
        "classification": TaskType.SENTIMENT,
        "ner": TaskType.NER,
        "named_entity_recognition": TaskType.NER,
        "entity_extraction": TaskType.NER,
        "extract_entities": TaskType.NER,
        "math": TaskType.MATH,
        "arithmetic": TaskType.MATH,
        "calculation": TaskType.MATH,
        "logic": TaskType.LOGIC,
        "reasoning": TaskType.LOGIC,
        "summary": TaskType.SUMMARIZATION,
        "summarization": TaskType.SUMMARIZATION,
        "summarisation": TaskType.SUMMARIZATION,
        "factual": TaskType.FACTUAL_QA,
        "factual_qa": TaskType.FACTUAL_QA,
        "qa": TaskType.FACTUAL_QA,
        "question_answering": TaskType.FACTUAL_QA,
        "code": TaskType.CODE_GENERATION,
        "code_generation": TaskType.CODE_GENERATION,
        "programming": TaskType.CODE_GENERATION,
        "debug": TaskType.CODE_DEBUGGING,
        "debugging": TaskType.CODE_DEBUGGING,
        "code_debugging": TaskType.CODE_DEBUGGING,
    }

    for key, value in aliases.items():
        if key in raw:
            return value

    return None

def infer_task_type(prompt: str, task: Any = None) -> TaskType:
    """Fast heuristic categorizer. Conservative: only strong signals win."""
    hinted = explicit_task_type(task)
    if hinted:
        return hinted

    p = compact_text(prompt, max_chars=3000)
    low = p.lower()

    if any(k in low for k in ("sentiment", "positive", "negative", "neutral")) and any(
        k in low for k in ("classify", "label", "review", "tone")
    ):
        return TaskType.SENTIMENT

    if any(k in low for k in ("named entities", "named entity", "extract entities", "ner")):
        return TaskType.NER

    if any(k in low for k in ("extract email", "extract emails", "extract url", "extract urls", "phone number", "phone numbers")):
        return TaskType.STRUCTURAL_EXTRACTION

    if any(k in low for k in ("summarize", "summarise", "summary", "tl;dr", "tldr")):
        return TaskType.SUMMARIZATION

    if any(k in low for k in ("debug", "fix the code", "traceback", "bug", "error in this code")):
        return TaskType.CODE_DEBUGGING

    if any(k in low for k in ("write a function", "write code", "generate code", "implement", "python function", "javascript function")):
        return TaskType.CODE_GENERATION

    if any(k in low for k in ("logic puzzle", "if and only if", "truth table", "who is lying", "constraint", "deduce")):
        return TaskType.LOGIC

    if looks_like_math_prompt(p):
        return TaskType.MATH

    if low.endswith("?") or any(low.startswith(w) for w in ("who ", "what ", "when ", "where ", "why ", "how ")):
        return TaskType.FACTUAL_QA

    return TaskType.GENERAL

# Deterministic arithmetic engine

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

class UnsafeExpression(ValueError):
    """Expression cannot be safely evaluated."""

def looks_like_math_prompt(text: str) -> bool:
    """Detect arithmetic prompts with high precision."""
    s = compact_text(text).lower()

    math_keywords = (
        "calculate",
        "compute",
        "evaluate",
        "solve",
        "what is",
        "find the value",
        "sum of",
        "product of",
        "difference between",
        "quotient of",
        "average of",
        "mean of",
        "square root",
        "sqrt",
        "plus",
        "minus",
        "times",
        "multiplied by",
        "divided by",
    )

    has_operator = bool(re.search(r"\d\s*[\+\-\*/%^]\s*\d", s))
    has_numbers = len(re.findall(r"-?\d+(?:\.\d+)?", s)) >= 1
    has_keyword = any(k in s for k in math_keywords)

    return has_numbers and (has_operator or has_keyword)

def _replace_math_words(text: str) -> str:
    """Convert simple English arithmetic terms into symbols."""
    s = text.lower()

    replacements = {
        "multiplied by": "*",
        "times": "*",
        "x": "*",
        "divided by": "/",
        "over": "/",
        "plus": "+",
        "added to": "+",
        "minus": "-",
        "less": "-",
        "to the power of": "**",
        "raised to": "**",
    }

    for phrase, op in replacements.items():
        s = re.sub(rf"\b{re.escape(phrase)}\b", op, s)

    s = re.sub(r"\bsquared\b", "**2", s)
    s = re.sub(r"\bcubed\b", "**3", s)

    return s

def _extract_arithmetic_expression(text: str) -> Optional[str]:
    """
    Extract an arithmetic expression from a prompt.

    This intentionally avoids aggressive extraction from general text. It only
    accepts expressions made of numbers, parentheses, decimal points, and
    arithmetic operators.
    """
    original = compact_text(text, max_chars=1000)
    low = _replace_math_words(original)

    # Percent pattern: "15% of 200" -> "(15/100)*200"
    percent_match = re.search(r"(-?\d+(?:\.\d+)?)\s*%\s+of\s+(-?\d+(?:\.\d+)?)", low)
    if percent_match:
        a, b = percent_match.groups()
        return f"({a}/100)*{b}"

    # Average/mean of a list of numbers.
    if re.search(r"\b(average|mean)\b", low):
        nums = re.findall(r"-?\d+(?:\.\d+)?", low)
        if nums:
            return "(" + "+".join(nums) + f")/{len(nums)}"

    # Sum/product phrases.
    if "sum of" in low:
        nums = re.findall(r"-?\d+(?:\.\d+)?", low)
        if len(nums) >= 2:
            return "+".join(nums)

    if "product of" in low:
        nums = re.findall(r"-?\d+(?:\.\d+)?", low)
        if len(nums) >= 2:
            return "*".join(nums)

    # Remove common command wrappers.
    low = re.sub(
        r"^(type:\s*\w+\s*)?(calculate|compute|evaluate|solve|what is|find the value of|question:)\s*",
        "",
        low,
    )
    low = low.replace("^", "**")

    # Candidate substrings consisting mostly of expression chars.
    candidates = re.findall(r"[-+*/().%\d\s]+(?:\*\*[-+*/().%\d\s]+)?", low)
    candidates = sorted((c.strip() for c in candidates), key=len, reverse=True)

    for cand in candidates:
        if not cand:
            continue
        if len(cand) > 160:
            continue
        if "%" in cand:
            # Treat % as modulo only if explicitly between two numbers.
            if not re.search(r"\d\s*%\s*\d", cand):
                continue
        if len(re.findall(r"\d", cand)) == 0:
            continue
        if not re.search(r"[\+\-\*/%]", cand):
            continue
        if re.fullmatch(r"[-+*/().%\d\s]+", cand):
            return cand.strip()

    return None

def _safe_fraction_from_constant(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise UnsafeExpression("booleans are not numeric constants")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsafeExpression("non-finite float")
        return Fraction(str(value))
    raise UnsafeExpression(f"unsupported constant: {type(value).__name__}")

def _eval_ast_node(node: ast.AST, depth: int = 0) -> Fraction:
    """Evaluate a restricted arithmetic AST into Fraction."""
    if depth > 32:
        raise UnsafeExpression("expression too deep")

    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        return _safe_fraction_from_constant(node.value)

    # Python <3.8 compatibility, harmless on newer versions.
    if isinstance(node, ast.Num):  # pragma: no cover
        return _safe_fraction_from_constant(node.n)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        val = _eval_ast_node(node.operand, depth + 1)
        if isinstance(node.op, ast.USub):
            return -val
        return val

    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_ast_node(node.left, depth + 1)
        right = _eval_ast_node(node.right, depth + 1)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsafeExpression("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise UnsafeExpression("floor division by zero")
            return Fraction(left.numerator // left.denominator, 1) // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise UnsafeExpression("modulo by zero")
            if left.denominator != 1 or right.denominator != 1:
                raise UnsafeExpression("modulo only supported for integers")
            return Fraction(left.numerator % right.numerator, 1)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1:
                raise UnsafeExpression("fractional exponents are not allowed")
            exponent = right.numerator
            if abs(exponent) > 12:
                raise UnsafeExpression("exponent too large")
            return left ** exponent

    raise UnsafeExpression(f"disallowed expression node: {type(node).__name__}")

def safe_eval_arithmetic_expression(expr: str) -> Fraction:
    """Safely evaluate a simple arithmetic expression."""
    expr = expr.strip()
    if not expr:
        raise UnsafeExpression("empty expression")
    if len(expr) > 160:
        raise UnsafeExpression("expression too long")
    if not re.fullmatch(r"[-+*/().%\d\s]+|\s*[-+*/().%\d\s*]+\s*", expr):
        # Allows ** because * is already included; rejects names, calls, brackets.
        raise UnsafeExpression("illegal characters in expression")

    parsed = ast.parse(expr, mode="eval")
    return _eval_ast_node(parsed)

def format_fraction_result(value: Fraction, prompt: str = "") -> str:
    """Format Fraction into a compact benchmark-friendly answer."""
    low = prompt.lower()

    if value.denominator == 1:
        return str(value.numerator)

    if "fraction" in low or "as a fraction" in low:
        return f"{value.numerator}/{value.denominator}"

    # Terminating decimal if denominator factors only into 2s and 5s.
    denom = value.denominator
    for factor in (2, 5):
        while denom % factor == 0:
            denom //= factor

    if denom == 1:
        decimal = value.numerator / value.denominator
        out = f"{decimal:.12f}".rstrip("0").rstrip(".")
        return out if out else "0"

    # Non-terminating: give concise decimal for decimal-looking prompts, else fraction.
    if any(k in low for k in ("decimal", "approx", "nearest", "round")):
        decimal = value.numerator / value.denominator
        return f"{decimal:.10f}".rstrip("0").rstrip(".")

    return f"{value.numerator}/{value.denominator}"

def try_deterministic_math(prompt: str) -> Optional[DeterministicHit]:
    """Try zero-token deterministic arithmetic."""
    expr = _extract_arithmetic_expression(prompt)
    if not expr:
        return None
    try:
        result = safe_eval_arithmetic_expression(expr)
        answer = format_fraction_result(result, prompt)
        return DeterministicHit(
            answer=answer,
            task_type=TaskType.MATH,
            confidence=0.99,
            reason="safe_arithmetic_ast",
            metadata={"expression": expr},
        )
    except Exception as exc:
        logger.debug("Math deterministic miss: %s", exc)
        return None

# Deterministic structural extraction

_EMAIL_RE = re.compile(r"[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+|www\.[^\s)\]}>\"']+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"(?:[$€£]\s?\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s?(?:usd|eur|gbp|dollars|euros|pounds))", re.IGNORECASE)

def _json_list(items: Iterable[str]) -> str:
    cleaned: List[str] = []
    seen = set()
    for item in items:
        v = str(item).strip().strip(".,;:)")
        if not v:
            continue
        key = v.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(v)
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))

def _extract_payload_after_marker(prompt: str) -> str:
    """
    Extract the text payload from task prompts.

    Tries common markers first, otherwise returns the whole prompt.
    """
    markers = (
        "text:",
        "input:",
        "sentence:",
        "review:",
        "document:",
        "content:",
        "extract from:",
    )
    low = prompt.lower()
    for marker in markers:
        idx = low.rfind(marker)
        if idx != -1:
            return prompt[idx + len(marker) :].strip()
    return prompt

def try_structural_extraction(prompt: str) -> Optional[DeterministicHit]:
    """Safely extract emails, URLs, phones, dates, money values for 0 tokens."""
    low = prompt.lower()
    payload = _extract_payload_after_marker(prompt)

    extractors: List[Tuple[str, Callable[[str], List[str]]]] = []

    if "email" in low:
        extractors.append(("email", lambda s: _EMAIL_RE.findall(s)))
    if "url" in low or "link" in low or "website" in low:
        extractors.append(("url", lambda s: _URL_RE.findall(s)))
    if "phone" in low or "telephone" in low:
        extractors.append(("phone", lambda s: _PHONE_RE.findall(s)))
    if "date" in low:
        extractors.append(("date", lambda s: _DATE_RE.findall(s)))
    if "money" in low or "price" in low or "amount" in low or "currency" in low:
        extractors.append(("money", lambda s: _MONEY_RE.findall(s)))

    if not extractors:
        return None

    found: List[str] = []
    kinds: List[str] = []
    for kind, extractor in extractors:
        values = extractor(payload)
        if values:
            kinds.append(kind)
            found.extend(values)

    # Empty extraction can still be correct if task expects [].
    return DeterministicHit(
        answer=_json_list(found),
        task_type=TaskType.STRUCTURAL_EXTRACTION,
        confidence=0.95 if found else 0.80,
        reason="regex_structural_extraction",
        metadata={"kinds": kinds},
    )

# Deterministic sentiment for obvious cases

_POSITIVE_WORDS = {
    "amazing", "awesome", "best", "excellent", "fantastic", "good",
    "great", "happy", "love", "loved", "perfect", "recommend",
    "satisfied", "wonderful", "delightful", "brilliant", "positive",
}

_NEGATIVE_WORDS = {
    "awful", "bad", "broken", "disappointed", "hate", "hated",
    "horrible", "poor", "refund", "sad", "terrible", "worst", 
    "useless", "angry", "negative", "buggy", "slow",
}

def try_deterministic_sentiment(prompt: str) -> Optional[DeterministicHit]:
    """
    Conservative lexical sentiment.
    Only accepts high-margin obvious sentiment. Ambiguous language falls through
    to Fireworks.
    """
    low = prompt.lower()
    if not any(k in low for k in ("sentiment", "classify", "label", "review")):
        return None

    payload = _extract_payload_after_marker(prompt).lower()
    tokens = re.findall(r"[a-z']+", payload)

    pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)

    # Strong negation handling for the most common cases.
    joined = " ".join(tokens)
    if re.search(r"\bnot\s+(good|great|happy|satisfied|recommend)\b", joined):
        neg += 2
        pos = max(0, pos - 1)
    if re.search(r"\bnot\s+(bad|terrible|awful|horrible)\b", joined):
        pos += 2
        neg = max(0, neg - 1)

    if pos >= neg + 2:
        return DeterministicHit("positive", TaskType.SENTIMENT, 0.93, "lexical_high_margin")
    if neg >= pos + 2:
        return DeterministicHit("negative", TaskType.SENTIMENT, 0.93, "lexical_high_margin")

    # Explicit single strong clue.
    if pos == 1 and neg == 0 and any(w in payload for w in ("love", "excellent", "awesome", "fantastic", "perfect")):
        return DeterministicHit("positive", TaskType.SENTIMENT, 0.90, "lexical_strong_single")
    if neg == 1 and pos == 0 and any(w in payload for w in ("hate", "worst", "terrible", "awful", "horrible")):
        return DeterministicHit("negative", TaskType.SENTIMENT, 0.90, "lexical_strong_single")

    return None

# Python sandbox helper

_DANGEROUS_CODE_PATTERNS = re.compile(
    r"(__import__|eval\s*\(|exec\s*\(|open\s*\(|subprocess|socket|requests|urllib|pathlib|shutil|os\.system|"
    r"pickle|marshal|compile\s*\(|globals\s*\(|locals\s*\(|input\s*\()",
    re.IGNORECASE,
)

def run_sandboxed_python(code: str, timeout_seconds: float = 2.0) -> Tuple[bool, str, str]:
    """
    Run generated Python in a constrained subprocess.

    This is intentionally conservative and should be used for model-generated
    helper scripts, not arbitrary untrusted long-running programs.

    Returns:
        (success, stdout, stderr)
    """
    clean = strip_code_fences(code).strip()
    if not clean:
        return False, "", "empty code"

    if len(clean) > 5000:
        return False, "", "code too long"

    if _DANGEROUS_CODE_PATTERNS.search(clean):
        return False, "", "blocked dangerous code pattern"

    with tempfile.TemporaryDirectory(prefix="symbio_sandbox_") as tmpdir:
        script_path = os.path.join(tmpdir, "runner.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(clean)

        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        preexec_fn = None
        if os.name == "posix":
            try:
                import resource

                def _limit_resources() -> None:
                    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
                    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
                    # 256 MB address-space cap. Ignore if platform refuses.
                    with contextlib.suppress(Exception):
                        resource.setrlimit(
                            resource.RLIMIT_AS,
                            (256 * 1024 * 1024, 256 * 1024 * 1024),
                        )

                preexec_fn = _limit_resources
            except Exception:
                preexec_fn = None

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", script_path],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                preexec_fn=preexec_fn,
                check=False,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            return completed.returncode == 0, stdout, stderr
        except subprocess.TimeoutExpired:
            return False, "", "timeout"
        except Exception as exc:
            return False, "", f"sandbox error: {exc}"

# Canonicalization

_SENTIMENT_ALIASES = {
    "a": "positive",
    "pos": "positive",
    "positive": "positive",
    "positiv": "positive",
    "b": "negative",
    "neg": "negative",
    "negative": "negative",
    "c": "neutral",
    "neu": "neutral",
    "neutral": "neutral",
    "mixed": "neutral",
}

def canonicalize_answer(answer: Any, task_type: TaskType | str = TaskType.GENERAL, prompt: str = "") -> str:
    """
    Canonicalize model/local output into benchmark-friendly minimal form.
    """
    try:
        tt = task_type if isinstance(task_type, TaskType) else TaskType(str(task_type))
    except Exception:
        tt = TaskType.GENERAL

    s = normalize_output_text(answer)

    if tt == TaskType.SENTIMENT:
        low = re.sub(r"[^a-z]", "", s.lower())
        if low in _SENTIMENT_ALIASES:
            return _SENTIMENT_ALIASES[low]
        for key, value in _SENTIMENT_ALIASES.items():
            if key in low and key not in {"a", "b", "c"}:
                return value
        return s.split()[0].lower().strip(".,;:") if s else ""

    if tt in (TaskType.NER, TaskType.STRUCTURAL_EXTRACTION):
        json_like = first_json_like(s)
        if json_like:
            try:
                parsed = json.loads(json_like)
                if isinstance(parsed, list):
                    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                if isinstance(parsed, dict):
                    # Flatten common {"entities": [...]} form.
                    for key in ("entities", "items", "names", "result"):
                        if isinstance(parsed.get(key), list):
                            return json.dumps(parsed[key], ensure_ascii=False, separators=(",", ":"))
                    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                return json_like

        # Fallback: comma/newline-separated entities.
        pieces = [p.strip(" -•\t\r\n\"'") for p in re.split(r"[,;\n]", s)]
        pieces = [p for p in pieces if p]
        if pieces:
            return _json_list(pieces)
        return "[]"

    if tt in (TaskType.CODE_GENERATION, TaskType.CODE_DEBUGGING):
        return strip_code_fences(str(answer)).strip()

    if tt == TaskType.MATH:
        s = s.strip()
        # Keep just the first numeric-looking scalar when the model is verbose.
        scalar = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
        if scalar and len(s) > len(scalar.group(0)) + 8:
            return scalar.group(0)
        return s.rstrip(".")

    if tt == TaskType.SUMMARIZATION:
        return s.strip()

    # General QA / logic: trim trailing punctuation only when scalar-like.
    if re.fullmatch(r"[A-Za-z0-9_\-/ .]+", s):
        return s.strip()
    return s

def canonicalize_batch_item(result: RouteResult) -> RouteResult:
    """Canonicalize a RouteResult in place and return it."""
    result.answer = canonicalize_answer(result.answer, result.task_type)
    return result

# Micro-prompt builder

def build_micro_prompt(task_type: TaskType, prompt: str) -> str:
    """Create task-specific minimal Fireworks prompt."""
    p = prompt.strip()

    if task_type == TaskType.SENTIMENT:
        return (
            "Classify the sentiment. Return exactly one word: positive, negative, or neutral.\n"
            f"Text: {p}\n"
            "Answer:"
        )

    if task_type in (TaskType.NER, TaskType.STRUCTURAL_EXTRACTION):
        return (
            "Extract named entities as a JSON array of strings only. "
            "Return [] if none.\n"
            f"Text: {p}\n"
            "JSON:"
        )

    if task_type == TaskType.MATH:
        return (
            "Solve. Return only the final numeric answer, no steps.\n"
            f"Problem: {p}\n"
            "Answer:"
        )

    if task_type == TaskType.LOGIC:
        return (
            "Solve the logic problem. Return only the final answer, no explanation.\n"
            f"Problem: {p}\n"
            "Answer:"
        )

    if task_type == TaskType.SUMMARIZATION:
        return (
            "Summarize concisely. Return only the summary, no preamble.\n"
            f"Text: {p}\n"
            "Summary:"
        )

    if task_type == TaskType.FACTUAL_QA:
        return (
            "Answer the question directly. Return only the answer, no explanation.\n"
            f"Question: {p}\n"
            "Answer:"
        )

    if task_type == TaskType.CODE_DEBUGGING:
        return (
            "Fix the code or answer the debugging task. "
            "Return only corrected code or the requested final output. No markdown.\n"
            f"{p}\n"
            "Answer:"
        )

    if task_type == TaskType.CODE_GENERATION:
        return (
            "Write the requested code. Return only code. No markdown fences. No explanation.\n"
            f"{p}\n"
            "Code:"
        )

    return (
        "Return only the final answer. No explanation. No markdown.\n"
        f"Task: {p}\n"
        "Answer:"
    )

# Router

class SymbioRouter:
    """
    symbioAI uncertainty-gated router.
    Core routing order:
        1. Exact normalized cache.
        2. Deterministic zero-token interceptors.
        3. Fireworks micro-prompt fallback.
        4. Safe fallback on error.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_cache: Optional[bool] = None,
        disable_cloud: Optional[bool] = None,
        timeout_seconds: Optional[float] = None,
        max_concurrency: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("FIREWORKS_BASE_URL", DEFAULT_FIREWORKS_BASE_URL)

        self.disable_cloud = (
            disable_cloud
            if disable_cloud is not None
            else os.getenv("SYMBIO_DISABLE_CLOUD", "0").strip() == "1"
        )

        self.enable_cache = (
            enable_cache
            if enable_cache is not None
            else os.getenv("SYMBIO_ENABLE_CACHE", "1").strip() != "0"
        )

        self.timeout_seconds = float(timeout_seconds or os.getenv("SYMBIO_TIMEOUT_SECONDS", "20"))
        self.max_concurrency = int(max_concurrency or os.getenv("SYMBIO_MAX_CONCURRENCY", "8"))

        self._client: Optional[AsyncOpenAI] = None
        self._cache: Dict[str, RouteResult] = {}
        self._semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy OpenAI-compatible async client."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("FIREWORKS_API_KEY is not configured.")
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _profile_for(self, task_type: TaskType) -> ModelProfile:
        return MODEL_PROFILES.get(task_type, MODEL_PROFILES[TaskType.GENERAL])

    def _model_for(self, profile: ModelProfile) -> str:
        return os.getenv(profile.model_env, profile.default_model)

    def _cache_get(self, key: str) -> Optional[RouteResult]:
        if not self.enable_cache:
            return None
        hit = self._cache.get(key)
        if not hit:
            return None
        return RouteResult(
            answer=hit.answer,
            task_type=hit.task_type,
            source="cache",
            confidence=hit.confidence,
            model=hit.model,
            raw_answer=hit.raw_answer,
            metadata={**hit.metadata, "cache_hit": True},
        )

    def _cache_set(self, key: str, result: RouteResult) -> None:
        if not self.enable_cache:
            return
        # Avoid unbounded growth in long-running services.
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = RouteResult(
            answer=result.answer,
            task_type=result.task_type,
            source=result.source,
            confidence=result.confidence,
            model=result.model,
            raw_answer=result.raw_answer,
            metadata=dict(result.metadata),
        )

    def try_deterministic(self, prompt: str, task_type: TaskType) -> Optional[DeterministicHit]:
        """Run local zero-token interceptors."""
        # Structural extraction before NER cloud fallback.
        if task_type in (TaskType.STRUCTURAL_EXTRACTION, TaskType.NER):
            hit = try_structural_extraction(prompt)
            if hit:
                return hit

        if task_type == TaskType.SENTIMENT:
            hit = try_deterministic_sentiment(prompt)
            if hit:
                return hit

        if task_type == TaskType.MATH or looks_like_math_prompt(prompt):
            hit = try_deterministic_math(prompt)
            if hit:
                return hit

        # A second structural pass catches explicit emails/URLs even if category
        # inference was general.
        hit = try_structural_extraction(prompt)
        if hit:
            return hit

        return None

    async def route(self, task: Any) -> RouteResult:
        """
        Route one task and return a canonicalized RouteResult.
        """
        prompt = extract_prompt(task)
        task_type = infer_task_type(prompt, task)
        cache_key = stable_prompt_hash(f"{task_type.value}\n{prompt}")

        cached = self._cache_get(cache_key)
        if cached:
            return cached

        deterministic = self.try_deterministic(prompt, task_type)
        if deterministic:
            result = RouteResult(
                answer=canonicalize_answer(deterministic.answer, deterministic.task_type, prompt),
                task_type=deterministic.task_type,
                source="deterministic",
                confidence=deterministic.confidence,
                metadata={
                    "reason": deterministic.reason,
                    **deterministic.metadata,
                },
            )
            self._cache_set(cache_key, result)
            return result

        if self.disable_cloud:
            result = self._fallback_without_cloud(prompt, task_type, reason="cloud_disabled")
            self._cache_set(cache_key, result)
            return result

        try:
            async with self._semaphore:
                result = await self._call_fireworks(prompt, task_type)
            self._cache_set(cache_key, result)
            return result
        except Exception as exc:
            logger.exception("Fireworks route failed for task_type=%s: %s", task_type, exc)
            result = self._fallback_without_cloud(prompt, task_type, reason=str(exc))
            self._cache_set(cache_key, result)
            return result

    async def route_batch(self, tasks: Sequence[Any]) -> List[RouteResult]:
        """Route a batch concurrently with a bounded semaphore."""
        return list(await asyncio.gather(*(self.route(task) for task in tasks)))

    async def _call_fireworks(self, prompt: str, task_type: TaskType) -> RouteResult:
        profile = self._profile_for(task_type)
        model = self._model_for(profile)
        user_prompt = build_micro_prompt(task_type, prompt)

        base_payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": profile.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "max_tokens": profile.max_tokens,
        }

        extra_body: Dict[str, Any] = {}
        if profile.reasoning_effort is not None:
            extra_body["reasoning_effort"] = profile.reasoning_effort

        try:
            response = await self.client.chat.completions.create(
                **base_payload,
                extra_body=extra_body or None,
            )
        except Exception as exc:
            # Some models reject reasoning_effort. Retry once without it.
            if extra_body and self._looks_like_reasoning_param_error(exc):
                logger.warning("Retrying Fireworks call without reasoning_effort for model=%s", model)
                response = await self.client.chat.completions.create(**base_payload)
            else:
                raise

        raw = ""
        try:
            raw = response.choices[0].message.content or ""
        except Exception:
            raw = str(response)

        answer = canonicalize_answer(raw, task_type, prompt)

        usage = getattr(response, "usage", None)
        usage_dict = {}
        if usage is not None:
            with contextlib.suppress(Exception):
                usage_dict = usage.model_dump()
            if not usage_dict:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

        return RouteResult(
            answer=answer,
            task_type=task_type,
            source="fireworks",
            confidence=0.75,
            model=model,
            raw_answer=raw,
            metadata={
                "profile": profile.name,
                "usage": usage_dict,
                "max_tokens": profile.max_tokens,
                "reasoning_effort": profile.reasoning_effort,
            },
        )

    @staticmethod
    def _looks_like_reasoning_param_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "reasoning_effort" in msg
            or "extra_body" in msg
            or "unsupported parameter" in msg
            or "unknown parameter" in msg
        )

    def _fallback_without_cloud(self, prompt: str, task_type: TaskType, reason: str) -> RouteResult:
        """
        Last-resort fallback.

        This should rarely be used in scoring, but it prevents hard crashes and
        keeps output structure valid.
        """
        if task_type in (TaskType.NER, TaskType.STRUCTURAL_EXTRACTION):
            answer = "[]"
        elif task_type == TaskType.SENTIMENT:
            answer = "neutral"
        elif task_type == TaskType.MATH:
            # Try one final deterministic math pass.
            hit = try_deterministic_math(prompt)
            answer = hit.answer if hit else "0"
        else:
            answer = ""

        return RouteResult(
            answer=canonicalize_answer(answer, task_type, prompt),
            task_type=task_type,
            source="fallback_error",
            confidence=0.0,
            metadata={"reason": reason},
        )

# Convenience singleton
_router_singleton: Optional[SymbioRouter] = None

def get_router() -> SymbioRouter:
    """FastAPI-friendly singleton accessor."""
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = SymbioRouter()
    return _router_singleton

async def route_task(task: Any) -> RouteResult:
    """Convenience async function for one task."""
    return await get_router().route(task)

async def route_tasks(tasks: Sequence[Any]) -> List[RouteResult]:
    """Convenience async function for a batch."""
    return await get_router().route_batch(tasks)

__all__ = [
    "TaskType", "ModelProfile", "RouteResult", "DeterministicHit",
    "SymbioRouter", "get_router", "route_task", "route_tasks",
    "canonicalize_answer", "extract_prompt", "infer_task_type",
    "try_deterministic_math", "try_structural_extraction",
    "try_deterministic_sentiment", "safe_eval_arithmetic_expression",
    "run_sandboxed_python", "MODEL_PROFILES",
]