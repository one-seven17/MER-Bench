from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from openai import OpenAI

# Best-effort exception imports (names can vary by SDK version)
try:
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover
    RateLimitError = type("RateLimitError", (Exception,), {})
    APIError = type("APIError", (Exception,), {})
    APITimeoutError = type("APITimeoutError", (Exception,), {})
    APIConnectionError = type("APIConnectionError", (Exception,), {})

# =============================
# Logging
# =============================


def setup_logging() -> None:
    """
    Env:
      LOG_LEVEL: TRACE|DEBUG|INFO|WARNING|ERROR (default INFO)
      LOG_JSON: 1 => serialize JSON logs
      LOG_FILE: optional file sink
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
    log_json = os.environ.get("LOG_JSON", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    log_file = (os.environ.get("LOG_FILE") or "").strip()

    logger.remove()
    logger.add(lambda m: print(m, end=""), level=level, serialize=log_json)
    if log_file:
        logger.add(
            log_file, level=level, rotation="10 MB", retention="7 days", enqueue=True
        )


setup_logging()

# =============================
# Constants & helpers
# =============================

EMOTIONS_8 = ["angry", "sad", "tense", "bored", "calm", "happy", "excited", "relaxed"]
EMOTIONS_9 = EMOTIONS_8 + ["other"]


def image_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stable_json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception as e:
        return {"error": f"json_parse_failed: {e}", "raw_text": text}


def _get_retry_after_seconds(exc: Exception) -> Optional[float]:
    for attr in ("response", "http_response"):
        resp = getattr(exc, attr, None)
        if resp is not None:
            headers = getattr(resp, "headers", None)
            if headers:
                ra = headers.get("retry-after") or headers.get("Retry-After")
                if ra:
                    try:
                        return float(ra)
                    except Exception:
                        pass
    headers = getattr(exc, "headers", None)
    if headers:
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except Exception:
                pass
    return None


# =============================
# Data model
# =============================


@dataclass(frozen=True)
class MemeSample:
    sample_id: str
    src_image_path: str
    gen_image_path: str
    src_emotion: str
    tgt_emotion: str
    src_caption_text: Optional[str] = None
    tgt_caption_text: Optional[str] = None
    edit_instruction: Optional[str] = None  # renamed from edit_spec


# =============================
# SQLite cache
# =============================


class SQLiteStore:
    """
    Minimal unified cache: key -> payload JSON

    Notes:
    - Thread-safe using a process-level lock.
    - Uses WAL to reduce writer blocking.
    """

    def __init__(self, db_path: str = "judge_cache.sqlite3", enabled: bool = True):
        self.enabled = enabled
        self.db_path = db_path
        self._lock = threading.Lock()
        if self.enabled:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS judge_cache (
                        cache_key TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        is_error INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.commit()
                logger.debug("SQLiteStore initialized at {}", self.db_path)
            finally:
                conn.close()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT payload_json, is_error FROM judge_cache WHERE cache_key=?",
                    (cache_key,),
                ).fetchone()
                if row is None:
                    return None
                payload_json, is_error = row
                payload = json.loads(payload_json)
                payload["_cached_is_error"] = bool(is_error)
                payload["_cache_key"] = cache_key
                return payload
            finally:
                conn.close()

    def upsert(self, cache_key: str, payload: Dict[str, Any], is_error: bool) -> None:
        if not self.enabled:
            return
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO judge_cache(cache_key, payload_json, is_error, updated_at)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        payload_json=excluded.payload_json,
                        is_error=excluded.is_error,
                        updated_at=excluded.updated_at
                    """,
                    (
                        cache_key,
                        json.dumps(payload, ensure_ascii=False),
                        1 if is_error else 0,
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()


# =============================
# Schema building (STRICT)
# =============================


@dataclass
class CostConfig:
    enabled: bool = False
    # If pricing is missing for a model, you can set these env vars:
    #   OPENAI_PRICE_INPUT_PER_1M, OPENAI_PRICE_CACHED_INPUT_PER_1M, OPENAI_PRICE_OUTPUT_PER_1M
    #
    # Or provide a JSON table via:
    #   OPENAI_PRICE_TABLE_JSON='{"gpt-5.2":{"input":1.75,"cached_input":0.18,"output":14.0}}'
    #
    # NOTE: This script does NOT fetch pricing from the web. You must keep these aligned
    # with OpenAI's official pricing page yourself.


@dataclass
class JudgeConfig:
    # Global switch for ALL rationale ordering across ALL blocks
    rationale_first: bool = True
    # Ablation switch: images go before or after the instruction text
    images_first: bool = True
    temperature: float = 0.0
    max_output_tokens: int = 900
    cost: CostConfig = field(default_factory=lambda: CostConfig(enabled=True))


def _obj(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    """
    Helper to produce strict object schema blocks required by OpenAI:
    - MUST include additionalProperties: false
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _ordered_fields(
    rationale_first: bool, core_fields: List[Tuple[str, Dict[str, Any]]]
) -> Tuple[Dict[str, Any], List[str]]:
    props: Dict[str, Any] = {}
    req: List[str] = []

    if rationale_first:
        props["rationale"] = {"type": "string"}
        req.append("rationale")

    for k, v in core_fields:
        props[k] = v
        req.append(k)

    if not rationale_first:
        props["rationale"] = {"type": "string"}
        req.append("rationale")

    return props, req


def build_judge_schema(rationale_first: bool) -> Tuple[str, Dict[str, Any]]:
    schema_name = "judge_rationale_first" if rationale_first else "judge_rationale_last"

    vt_props, vt_req = _ordered_fields(
        rationale_first,
        core_fields=[
            ("generation_quality_1_5", {"type": "integer", "minimum": 1, "maximum": 5}),
            ("emotion_accuracy_1_5", {"type": "integer", "minimum": 1, "maximum": 5}),
        ],
    )

    layout_props, layout_req = _ordered_fields(
        rationale_first,
        core_fields=[("consistent", {"type": "boolean"})],
    )

    emo_props, emo_req = _ordered_fields(
        rationale_first,
        core_fields=[
            ("label", {"type": "string", "enum": EMOTIONS_9}),
            ("other_emotion", {"type": ["string", "null"]}),
        ],
    )

    shift_props, shift_req = _ordered_fields(
        rationale_first,
        core_fields=[
            ("magnitude_1_5", {"type": "integer", "minimum": 1, "maximum": 5})
        ],
    )

    oq_props, oq_req = _ordered_fields(
        rationale_first,
        core_fields=[
            (
                "overall_generation_quality_1_5",
                {"type": "integer", "minimum": 1, "maximum": 5},
            )
        ],
    )

    schema = _obj(
        properties={
            "visual_assessment": _obj(vt_props, vt_req),
            "text_assessment": _obj(vt_props, vt_req),
            "layout_consistency": _obj(layout_props, layout_req),
            "overall_emotion_classification": _obj(emo_props, emo_req),
            "perceived_emotion_shift": _obj(shift_props, shift_req),
            "overall_generation_quality": _obj(oq_props, oq_req),
        },
        required=[
            "visual_assessment",
            "text_assessment",
            "layout_consistency",
            "overall_emotion_classification",
            "perceived_emotion_shift",
            "overall_generation_quality",
        ],
    )
    return schema_name, schema


# =============================
# Prompt + input builders
# =============================


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_text: str


class PromptBuilder:
    @staticmethod
    def build_prompts(sample: MemeSample) -> PromptBundle:
        order_hint = (
            "Output JSON keys in the exact order implied by the schema. "
            "Do not add extra keys."
        )

        system_prompt = (
            "You are a strict expert judge for meme emotion reframing.\n"
            "You MUST compare SOURCE vs EDITED.\n"
            "ONLY the affect/emotion should change; everything else should remain consistent.\n"
            "\n"
            "Critical rules:\n"
            "1) Layout/structure: Preserve panel/grid structure. If SOURCE is single-panel, EDITED must be single-panel.\n"
            "   If SOURCE is multi-panel/grid meme, EDITED must also be multi-panel/grid and preserve panel order and caption regions.\n"
            "2) Emotion scoring MUST be target-gated:\n"
            "   - All emotion-related scores shown in the JSON must first check alignment with TARGET emotion.\n"
            "   - If EDITED does NOT match TARGET emotion, emotion_accuracy_1_5 MUST be <= 2.\n"
            "3) Emotion classification:\n"
            "   - Choose ONE primary emotion label for the EDITED meme.\n"
            "   - If it fits none of the 8 predefined labels, use label='other' and set other_emotion to the concrete emotion name.\n"
            "   - If label != 'other', set other_emotion to null.\n"
            "4) Text-visual semantic alignment:\n"
            "   - In rationales, explicitly mention whether the caption and visual cues jointly support the same scenario and intended affect.\n"
            "\n"
            f"{order_hint}\n"
            "Return JSON only."
        )

        user_lines: List[str] = []
        user_lines.append(
            f"Allowed emotions (classification): {', '.join(EMOTIONS_8)} + other"
        )
        user_lines.append(f"SOURCE emotion (reference): {sample.src_emotion}")
        user_lines.append(f"TARGET emotion (reference): {sample.tgt_emotion}")

        if sample.src_caption_text:
            user_lines.append(f"SOURCE caption (reference):\n{sample.src_caption_text}")
        if sample.tgt_caption_text:
            user_lines.append(f"TARGET caption (reference):\n{sample.tgt_caption_text}")
        if sample.edit_instruction:
            user_lines.append(
                f"Editing instruction (reference):\n{sample.edit_instruction}"
            )

        user_lines.append(
            "Scoring instructions:\n"
            "- visual_assessment.generation_quality_1_5: visual cleanliness / realism / artifact-free.\n"
            "- visual_assessment.emotion_accuracy_1_5: ONLY based on visual cues matching TARGET emotion; if not matched => <=2.\n"
            "- text_assessment.generation_quality_1_5: text legibility, completeness, placement.\n"
            "- text_assessment.emotion_accuracy_1_5: text affect matches TARGET emotion while describing the same scenario; if not matched => <=2.\n"
            "- layout_consistency.consistent: true ONLY if panel/grid type and layout match SOURCE (single vs multi-grid).\n"
            "- overall_emotion_classification: primary emotion of EDITED (8 labels or 'other').\n"
            "- perceived_emotion_shift.magnitude_1_5: perceived change magnitude from SOURCE to EDITED.\n"
            "- overall_generation_quality.overall_generation_quality_1_5: holistic output quality.\n"
            "\n"
            "Be strict: if non-affective elements drift (subject identity, style, layout, scenario), penalize accordingly.\n"
            "In each rationale, explicitly state: (a) TARGET emotion alignment, (b) scenario preservation, (c) text-visual semantic alignment."
        )

        return PromptBundle(
            system_prompt=system_prompt, user_text="\n\n".join(user_lines)
        )


class InputBuilder:
    @staticmethod
    def build_image_blocks(sample: MemeSample) -> List[Dict[str, Any]]:
        src_img = {
            "type": "input_image",
            "image_url": image_to_data_url(sample.src_image_path),
            "detail": "high",
        }
        gen_img = {
            "type": "input_image",
            "image_url": image_to_data_url(sample.gen_image_path),
            "detail": "high",
        }
        return [
            {"type": "input_text", "text": "SOURCE image:"},
            src_img,
            {"type": "input_text", "text": "EDITED image:"},
            gen_img,
        ]

    @staticmethod
    def build_text_block(user_text: str) -> List[Dict[str, Any]]:
        return [{"type": "input_text", "text": user_text}]

    @staticmethod
    def build_user_content(
        sample: MemeSample, user_text: str, images_first: bool
    ) -> List[Dict[str, Any]]:
        images_block = InputBuilder.build_image_blocks(sample)
        text_block = InputBuilder.build_text_block(user_text)
        return images_block + text_block if images_first else text_block + images_block

    @staticmethod
    def build_messages(
        system_prompt: str, user_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": user_content},
        ]


def _post_validate_payload(payload: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    try:
        cls = payload.get("overall_emotion_classification", {})
        label = cls.get("label")
        other_emotion = cls.get("other_emotion")
        if label == "other":
            if not isinstance(other_emotion, str) or not other_emotion.strip():
                errs.append(
                    "overall_emotion_classification.other_emotion must be a non-empty string when label=='other'"
                )
        else:
            if other_emotion is not None:
                errs.append(
                    "overall_emotion_classification.other_emotion must be null when label!='other'"
                )
    except Exception as e:
        errs.append(f"post_validate_exception: {type(e).__name__}: {e}")
    return errs


# =============================
# Cost estimation helpers
# =============================


def _coerce_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_usage(resp: Any) -> Dict[str, int]:
    """
    Best-effort extraction across SDK variants.
    Returns:
      input_tokens, output_tokens, cached_input_tokens, reasoning_tokens
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }

    input_tokens = _coerce_int(getattr(usage, "input_tokens", 0))
    output_tokens = _coerce_int(getattr(usage, "output_tokens", 0))
    reasoning_tokens = _coerce_int(getattr(usage, "reasoning_tokens", 0))

    cached_input_tokens = 0
    # common pattern: usage.input_tokens_details.cached_tokens
    itd = getattr(usage, "input_tokens_details", None)
    if itd is not None:
        cached_input_tokens = _coerce_int(getattr(itd, "cached_tokens", 0))
        if cached_input_tokens == 0 and isinstance(itd, dict):
            cached_input_tokens = _coerce_int(itd.get("cached_tokens", 0))

    # some variants might report cached tokens elsewhere; keep best-effort only.
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _load_pricing_table_from_env() -> Dict[str, Dict[str, float]]:
    """
    OPENAI_PRICE_TABLE_JSON='{"gpt-5.2":{"input":1.75,"cached_input":0.18,"output":14.0}}'
    """
    raw = (os.environ.get("OPENAI_PRICE_TABLE_JSON") or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out: Dict[str, Dict[str, float]] = {}
            for k, v in obj.items():
                if isinstance(v, dict):
                    out[str(k)] = {
                        "input": float(v.get("input", 0.0)),
                        "cached_input": float(v.get("cached_input", 0.0)),
                        "output": float(v.get("output", 0.0)),
                        # optional: "reasoning": float(v.get("reasoning", 0.0)),
                    }
            return out
    except Exception:
        return {}
    return {}


def _get_pricing_for_model(model: str) -> Dict[str, float]:
    """
    Prices are USD per 1M tokens.
    1) try OPENAI_PRICE_TABLE_JSON
    2) try OPENAI_PRICE_* env vars
    3) fallback to a safe "unknown" (zeros)
    """
    table = _load_pricing_table_from_env()
    if model in table:
        return table[model]

    # per-model env override
    # Example:
    #   OPENAI_PRICE_INPUT_PER_1M=1.75
    #   OPENAI_PRICE_CACHED_INPUT_PER_1M=0.18
    #   OPENAI_PRICE_OUTPUT_PER_1M=14.0
    try:
        return {
            "input": float(os.environ.get("OPENAI_PRICE_INPUT_PER_1M", "0") or "0"),
            "cached_input": float(
                os.environ.get("OPENAI_PRICE_CACHED_INPUT_PER_1M", "0") or "0"
            ),
            "output": float(os.environ.get("OPENAI_PRICE_OUTPUT_PER_1M", "0") or "0"),
        }
    except Exception:
        return {"input": 0.0, "cached_input": 0.0, "output": 0.0}


def estimate_cost_usd(model: str, usage: Dict[str, int]) -> Dict[str, Any]:
    """
    Returns a structured cost object.
    """
    pricing = _get_pricing_for_model(model)
    input_tokens = int(usage.get("input_tokens", 0))
    cached_input_tokens = int(usage.get("cached_input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    reasoning_tokens = int(usage.get("reasoning_tokens", 0))

    # Typical billing: cached tokens are a subset of input tokens
    billable_input_tokens = max(0, input_tokens - cached_input_tokens)

    cost = 0.0
    cost += billable_input_tokens / 1_000_000 * float(pricing.get("input", 0.0))
    cost += cached_input_tokens / 1_000_000 * float(pricing.get("cached_input", 0.0))
    cost += output_tokens / 1_000_000 * float(pricing.get("output", 0.0))

    # If your org/account bills reasoning tokens separately, add it here by providing pricing["reasoning"]
    # cost += reasoning_tokens / 1_000_000 * float(pricing.get("reasoning", 0.0))

    return {
        "model": model,
        "pricing_usd_per_1m_tokens": pricing,
        "usage_tokens": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
        "estimated_cost_usd": cost,
    }


# =============================
# Retry config + client
# =============================


@dataclass
class RetryConfig:
    max_retries: int = 6
    base_delay_s: float = 0.6
    max_delay_s: float = 20.0
    jitter: float = 0.25
    timeout_s: Optional[float] = 120


@dataclass
class CacheConfig:
    enabled: bool = True
    db_path: str = "judge_cache.sqlite3"
    return_cached_errors: bool = False


class JudgeClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        retry: Optional[RetryConfig] = None,
        max_in_flight: int = 8,
        cache: Optional[CacheConfig] = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE"),
        )
        self.model = model
        self.retry = retry or RetryConfig()
        self._sem = threading.Semaphore(max_in_flight)

        self.cache_cfg = cache or CacheConfig()
        self.store = SQLiteStore(self.cache_cfg.db_path, enabled=self.cache_cfg.enabled)

        logger.info(
            "JudgeClient ready | model={} | cache_enabled={} | db={}",
            self.model,
            self.store.enabled,
            self.cache_cfg.db_path,
        )

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIError):
            status = getattr(exc, "status_code", None)
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
                return True
        status = getattr(exc, "status_code", None)
        if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
            return True
        return False

    def _sleep_backoff(self, attempt: int, exc: Exception) -> None:
        ra = _get_retry_after_seconds(exc)
        if ra is not None and ra > 0:
            wait = min(ra, self.retry.max_delay_s)
            logger.warning("Retry-After honored: {:.2f}s", wait)
            time.sleep(wait)
            return

        delay = min(self.retry.base_delay_s * (2**attempt), self.retry.max_delay_s)
        if self.retry.jitter and delay > 0:
            delay *= max(
                0.0, 1.0 + random.uniform(-self.retry.jitter, self.retry.jitter)
            )
        logger.warning("Backoff sleep: {:.2f}s (attempt={})", delay, attempt + 1)
        time.sleep(delay)

    def _build_cache_key(
        self,
        *,
        schema_name: str,
        schema: Dict[str, Any],
        prompts: PromptBundle,
        sample: MemeSample,
        cfg: JudgeConfig,
        src_sha: str,
        gen_sha: str,
    ) -> str:
        schema_sha = sha256_bytes(stable_json_dumps(schema).encode("utf-8"))
        descriptor = {
            "judge_model": self.model,
            "schema_name": schema_name,
            "schema_sha256": schema_sha,
            "system_prompt_sha256": sha256_bytes(prompts.system_prompt.encode("utf-8")),
            "user_text_sha256": sha256_bytes(prompts.user_text.encode("utf-8")),
            "cfg": {
                "rationale_first": cfg.rationale_first,
                "images_first": cfg.images_first,
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_output_tokens,
                "cost_enabled": bool(getattr(cfg.cost, "enabled", False)),
            },
            "sample_id": sample.sample_id,
            "src_emotion": sample.src_emotion,
            "tgt_emotion": sample.tgt_emotion,
            "src_caption_text": sample.src_caption_text,
            "tgt_caption_text": sample.tgt_caption_text,
            "edit_instruction": sample.edit_instruction,
            "src_image_sha256": src_sha,
            "gen_image_sha256": gen_sha,
        }
        return sha256_bytes(stable_json_dumps(descriptor).encode("utf-8"))

    def judge(
        self,
        sample: MemeSample,
        cfg: JudgeConfig,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Returns: (judge_payload, cost_info_or_none)
        """
        schema_name, schema = build_judge_schema(cfg.rationale_first)
        prompts = PromptBuilder.build_prompts(sample)

        src_sha = sha256_file(sample.src_image_path)
        gen_sha = sha256_file(sample.gen_image_path)

        cache_key = self._build_cache_key(
            schema_name=schema_name,
            schema=schema,
            prompts=prompts,
            sample=sample,
            cfg=cfg,
            src_sha=src_sha,
            gen_sha=gen_sha,
        )
        cache_short = cache_key[:12]

        # cache read
        if use_cache and self.store.enabled and not force_refresh:
            cached = self.store.get(cache_key)
            if cached is not None:
                is_err = bool(cached.get("_cached_is_error", False))
                if (not is_err) or self.cache_cfg.return_cached_errors:
                    logger.info(
                        "CACHE HIT | sample={} | key={}", sample.sample_id, cache_short
                    )
                    # cost for cache-hit is "unknown" unless you store it separately.
                    cost_info = None
                    if cfg.cost.enabled:
                        cost_info = {
                            "model": self.model,
                            "cached_result": True,
                            "note": "cache hit: no API usage tokens available; estimated_cost_usd set to 0.0",
                            "estimated_cost_usd": 0.0,
                        }
                    return cached, cost_info
                logger.warning(
                    "CACHE ERROR (ignored) | sample={} | key={}",
                    sample.sample_id,
                    cache_short,
                )

        if force_refresh:
            logger.warning(
                "FORCE REFRESH | sample={} | key={}", sample.sample_id, cache_short
            )
        else:
            logger.info(
                "CACHE MISS | sample={} | key={}", sample.sample_id, cache_short
            )

        user_content = InputBuilder.build_user_content(
            sample, prompts.user_text, images_first=cfg.images_first
        )
        messages = InputBuilder.build_messages(prompts.system_prompt, user_content)

        try:
            logger.debug(
                "IMG INFO | sample={} | src_bytes={} | gen_bytes={}",
                sample.sample_id,
                os.path.getsize(sample.src_image_path),
                os.path.getsize(sample.gen_image_path),
            )
        except Exception:
            pass

        last_exc: Optional[Exception] = None
        t0 = time.time()

        with self._sem:
            for attempt in range(self.retry.max_retries + 1):
                try:
                    logger.debug(
                        "API CALL | sample={} | attempt={}/{} | images_first={} | rationale_first={} | cost_enabled={}",
                        sample.sample_id,
                        attempt + 1,
                        self.retry.max_retries + 1,
                        cfg.images_first,
                        cfg.rationale_first,
                        cfg.cost.enabled,
                    )

                    resp = self.client.responses.create(
                        model=self.model,
                        input=messages,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "strict": True,
                                "schema": schema,
                            }
                        },
                        temperature=cfg.temperature,
                        max_output_tokens=cfg.max_output_tokens,
                        timeout=self.retry.timeout_s,
                    )

                    payload = _safe_json_loads(resp.output_text)
                    latency_ms = (time.time() - t0) * 1000.0
                    is_err = "error" in payload

                    if not is_err:
                        post_errs = _post_validate_payload(payload)
                        if post_errs:
                            payload["_post_validation_errors"] = post_errs
                            is_err = True
                            logger.error(
                                "POST-VALIDATION FAILED | sample={} | errs={}",
                                sample.sample_id,
                                post_errs,
                            )

                    # cost estimation (based on usage tokens from this API response)
                    cost_info: Optional[Dict[str, Any]] = None
                    if cfg.cost.enabled:
                        usage = _extract_usage(resp)
                        cost_info = estimate_cost_usd(self.model, usage)
                        cost_info["latency_ms"] = latency_ms
                        logger.info(
                            "COST | sample={} | input={} | cached_in={} | output={} | est_usd={:.8f}",
                            sample.sample_id,
                            cost_info["usage_tokens"]["input_tokens"],
                            cost_info["usage_tokens"]["cached_input_tokens"],
                            cost_info["usage_tokens"]["output_tokens"],
                            float(cost_info["estimated_cost_usd"]),
                        )

                    if is_err:
                        logger.error(
                            "API ERROR/INVALID PAYLOAD | sample={} | latency_ms={:.1f}",
                            sample.sample_id,
                            latency_ms,
                        )
                    else:
                        logger.info(
                            "API OK | sample={} | latency_ms={:.1f}",
                            sample.sample_id,
                            latency_ms,
                        )

                    if use_cache and self.store.enabled:
                        self.store.upsert(cache_key, payload, is_error=is_err)
                        logger.debug(
                            "DB UPSERT | sample={} | key={} | is_error={}",
                            sample.sample_id,
                            cache_short,
                            is_err,
                        )

                    return payload, cost_info

                except Exception as e:
                    last_exc = e
                    if attempt >= self.retry.max_retries or not self._is_retryable(e):
                        out = {"error": f"request_failed: {type(e).__name__}: {e}"}
                        logger.error(
                            "API FAILED (final) | sample={} | err={}",
                            sample.sample_id,
                            out["error"],
                        )
                        if use_cache and self.store.enabled:
                            self.store.upsert(cache_key, out, is_error=True)
                        return out, None

                    logger.warning(
                        "API FAILED (retrying) | sample={} | attempt={}/{} | err={}",
                        sample.sample_id,
                        attempt + 1,
                        self.retry.max_retries + 1,
                        f"{type(e).__name__}: {e}",
                    )
                    self._sleep_backoff(attempt, e)

        return {"error": f"request_failed_unknown: {last_exc}"}, None


# =============================
# Batch pipeline (sample-level concurrency)
# =============================


@dataclass
class ConcurrencyConfig:
    sample_workers: int = 4


@dataclass
class RunConfig:
    use_cache: bool = True
    force_refresh: bool = False


class JudgePipeline:
    def __init__(
        self,
        client: JudgeClient,
        cfg: JudgeConfig,
        ccfg: Optional[ConcurrencyConfig] = None,
        rcfg: Optional[RunConfig] = None,
    ):
        self.client = client
        self.cfg = cfg
        self.ccfg = ccfg or ConcurrencyConfig()
        self.rcfg = rcfg or RunConfig()

    def evaluate_sample(self, sample: MemeSample) -> Dict[str, Any]:
        logger.info("SAMPLE START | {}", sample.sample_id)
        payload, cost_info = self.client.judge(
            sample,
            self.cfg,
            use_cache=self.rcfg.use_cache,
            force_refresh=self.rcfg.force_refresh,
        )
        logger.info("SAMPLE DONE | {}", sample.sample_id)

        out: Dict[str, Any] = {"sample": asdict(sample), "judge": payload}
        if self.cfg.cost.enabled:
            out["cost"] = cost_info or {
                "model": self.client.model,
                "note": "cost enabled but no cost info available (e.g., request failed before usage was returned).",
            }
        return out

    def evaluate_batch(self, samples: Sequence[MemeSample]) -> List[Dict[str, Any]]:
        if not samples:
            return []
        logger.info("BATCH START | n={}", len(samples))
        out: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.ccfg.sample_workers) as ex:
            futs = {ex.submit(self.evaluate_sample, s): s for s in samples}
            done = 0
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    out.append(fut.result())
                except Exception as e:
                    logger.exception("SAMPLE FAILED | {}", s.sample_id)
                    err_obj: Dict[str, Any] = {
                        "sample": asdict(s),
                        "judge": {
                            "error": f"sample_exception: {type(e).__name__}: {e}"
                        },
                    }
                    if self.cfg.cost.enabled:
                        err_obj["cost"] = {
                            "model": self.client.model,
                            "note": "sample exception: cost unavailable",
                        }
                    out.append(err_obj)
                done += 1
                if done % 10 == 0 or done == len(samples):
                    logger.info("BATCH PROGRESS | {}/{}", done, len(samples))

        out.sort(key=lambda x: x["sample"]["sample_id"])
        logger.info("BATCH DONE | n={}", len(out))
        return out
