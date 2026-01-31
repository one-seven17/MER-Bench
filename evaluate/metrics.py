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
from dataclasses import asdict, dataclass
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


# -----------------------------
# Logging setup
# -----------------------------


def setup_logging() -> None:
    """
    Controls (env):
      - LOG_LEVEL: TRACE|DEBUG|INFO|WARNING|ERROR (default INFO)
      - LOG_JSON: 1 to output JSON logs (default 0)
      - LOG_FILE: path to write logs (optional)
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
    log_json = os.environ.get("LOG_JSON", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    log_file = os.environ.get("LOG_FILE", "").strip()

    logger.remove()

    if log_json:
        # JSON-like logs via serialize=True (loguru)
        logger.add(lambda msg: print(msg, end=""), level=level, serialize=True)
    else:
        logger.add(lambda msg: print(msg, end=""), level=level)

    if log_file:
        # Rotating file sink; keep it simple and safe
        logger.add(
            log_file, level=level, rotation="10 MB", retention="10 days", enqueue=True
        )

    logger.debug(
        "Logging initialized: level={}, json={}, file={}", level, log_json, log_file
    )


setup_logging()


# -----------------------------
# Constants & helpers
# -----------------------------

EMOTIONS_8 = ["angry", "sad", "fear", "disgust", "calm", "happy", "excited", "relaxed"]


def image_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception as e:
        return {"error": f"json_parse_failed: {e}", "raw_text": text}


def likert_to_unit(x: int, k: int = 5) -> float:
    x = max(1, min(k, int(x)))
    return (x - 1) / (k - 1) if k > 1 else 0.0


def tri_to_unit(label: str) -> float:
    m = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    return m.get(str(label).strip().lower(), 0.0)


def _get_retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    Best-effort parse Retry-After from exception/response.
    openai-python exception surface can differ across versions.
    """
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


# -----------------------------
# Data model
# -----------------------------


@dataclass(frozen=True)
class MemeSample:
    sample_id: str
    src_image_path: str
    src_emotion: str
    gen_image_path: str
    tgt_emotion: str
    src_caption_text: Optional[str] = None
    tgt_caption_text: Optional[str] = None
    edit_spec: Optional[str] = None


@dataclass
class MetricResult:
    metric_id: str
    metric_name: str
    score: float  # normalized 0..1
    label: str  # short categorical label
    rationale: str  # short explanation
    evidence: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None  # non-null if failed after retries / parse issues


@dataclass
class SampleJudgement:
    sample_id: str
    src_emotion: str
    tgt_emotion: str
    results: List[MetricResult]

    def aggregate(self, weights: Optional[Dict[str, float]] = None) -> float:
        ok = [r for r in self.results if r.error is None]
        if not ok:
            return 0.0
        if not weights:
            return sum(r.score for r in ok) / len(ok)
        total_w = 0.0
        total = 0.0
        for r in ok:
            w = float(weights.get(r.metric_id, 0.0))
            total += r.score * w
            total_w += w
        return (total / total_w) if total_w > 0 else 0.0


# -----------------------------
# SQLite cache + results store (unified)
# -----------------------------


class SQLiteStore:
    """
    Unified cache + results log store using SQLite (stdlib).

    Concurrency:
    - Thread-safe with a process-level lock.
    - SQLite WAL improves concurrent access.
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
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS judge_cache (
                        cache_key TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,

                        judge_model TEXT NOT NULL,
                        metric_id TEXT NOT NULL,
                        schema_name TEXT NOT NULL,

                        sample_id TEXT NOT NULL,
                        src_emotion TEXT,
                        tgt_emotion TEXT,

                        src_image_sha256 TEXT,
                        gen_image_sha256 TEXT,

                        system_prompt_sha256 TEXT NOT NULL,
                        user_text_sha256 TEXT NOT NULL,

                        payload_json TEXT NOT NULL,
                        is_error INTEGER NOT NULL DEFAULT 0,

                        last_latency_ms REAL,
                        last_attempts INTEGER,
                        last_cache_hit INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_metric ON judge_cache(metric_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sample ON judge_cache(sample_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_model ON judge_cache(judge_model)"
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

    def upsert(
        self,
        *,
        cache_key: str,
        judge_model: str,
        metric_id: str,
        schema_name: str,
        sample: MemeSample,
        src_image_sha256: Optional[str],
        gen_image_sha256: Optional[str],
        system_prompt_sha256: str,
        user_text_sha256: str,
        payload: Dict[str, Any],
        is_error: bool,
        latency_ms: Optional[float],
        attempts: Optional[int],
        cache_hit: bool,
        overwrite: bool,
    ) -> None:
        if not self.enabled:
            return

        now = time.time()
        payload_json = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            conn = self._connect()
            try:
                if overwrite:
                    conn.execute(
                        """
                        INSERT INTO judge_cache(
                            cache_key, created_at, updated_at,
                            judge_model, metric_id, schema_name,
                            sample_id, src_emotion, tgt_emotion,
                            src_image_sha256, gen_image_sha256,
                            system_prompt_sha256, user_text_sha256,
                            payload_json, is_error,
                            last_latency_ms, last_attempts, last_cache_hit
                        ) VALUES(
                            ?, ?, ?,
                            ?, ?, ?,
                            ?, ?, ?,
                            ?, ?,
                            ?, ?,
                            ?, ?,
                            ?, ?, ?
                        )
                        ON CONFLICT(cache_key) DO UPDATE SET
                            updated_at=excluded.updated_at,
                            judge_model=excluded.judge_model,
                            metric_id=excluded.metric_id,
                            schema_name=excluded.schema_name,
                            sample_id=excluded.sample_id,
                            src_emotion=excluded.src_emotion,
                            tgt_emotion=excluded.tgt_emotion,
                            src_image_sha256=excluded.src_image_sha256,
                            gen_image_sha256=excluded.gen_image_sha256,
                            system_prompt_sha256=excluded.system_prompt_sha256,
                            user_text_sha256=excluded.user_text_sha256,
                            payload_json=excluded.payload_json,
                            is_error=excluded.is_error,
                            last_latency_ms=excluded.last_latency_ms,
                            last_attempts=excluded.last_attempts,
                            last_cache_hit=excluded.last_cache_hit
                        """,
                        (
                            cache_key,
                            now,
                            now,
                            judge_model,
                            metric_id,
                            schema_name,
                            sample.sample_id,
                            sample.src_emotion,
                            sample.tgt_emotion,
                            src_image_sha256,
                            gen_image_sha256,
                            system_prompt_sha256,
                            user_text_sha256,
                            payload_json,
                            1 if is_error else 0,
                            latency_ms,
                            attempts,
                            1 if cache_hit else 0,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO judge_cache(
                            cache_key, created_at, updated_at,
                            judge_model, metric_id, schema_name,
                            sample_id, src_emotion, tgt_emotion,
                            src_image_sha256, gen_image_sha256,
                            system_prompt_sha256, user_text_sha256,
                            payload_json, is_error,
                            last_latency_ms, last_attempts, last_cache_hit
                        ) VALUES(
                            ?, ?, ?,
                            ?, ?, ?,
                            ?, ?, ?,
                            ?, ?,
                            ?, ?,
                            ?, ?,
                            ?, ?, ?
                        )
                        """,
                        (
                            cache_key,
                            now,
                            now,
                            judge_model,
                            metric_id,
                            schema_name,
                            sample.sample_id,
                            sample.src_emotion,
                            sample.tgt_emotion,
                            src_image_sha256,
                            gen_image_sha256,
                            system_prompt_sha256,
                            user_text_sha256,
                            payload_json,
                            1 if is_error else 0,
                            latency_ms,
                            attempts,
                            1 if cache_hit else 0,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()


# -----------------------------
# OpenAI client wrapper with retries + caching/store
# -----------------------------


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
        model: str = "gpt-5-mini",
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
        self._in_flight = threading.Semaphore(max_in_flight)
        self.cache_cfg = cache or CacheConfig()
        self.store = SQLiteStore(
            db_path=self.cache_cfg.db_path, enabled=self.cache_cfg.enabled
        )
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

    def build_cache_key(
        self,
        *,
        metric_id: str,
        schema_name: str,
        system_prompt: str,
        sample: MemeSample,
        user_parts: List[Dict[str, Any]],
        src_image_sha256: Optional[str],
        gen_image_sha256: Optional[str],
    ) -> Tuple[str, str, str]:
        text_parts: List[str] = []
        for p in user_parts:
            if p.get("type") == "input_text":
                text_parts.append(str(p.get("text", "")))

        system_prompt_sha = sha256_bytes(system_prompt.encode("utf-8"))
        user_text_sha = sha256_bytes(stable_json_dumps(text_parts).encode("utf-8"))

        descriptor = {
            "judge_model": self.model,
            "metric_id": metric_id,
            "schema_name": schema_name,
            "system_prompt_sha256": system_prompt_sha,
            "sample_id": sample.sample_id,
            "src_emotion": sample.src_emotion,
            "tgt_emotion": sample.tgt_emotion,
            "src_caption_text": sample.src_caption_text,
            "tgt_caption_text": sample.tgt_caption_text,
            "edit_spec": sample.edit_spec,
            "src_image_sha256": src_image_sha256,
            "gen_image_sha256": gen_image_sha256,
            "user_text_sha256": user_text_sha,
        }
        cache_key = sha256_bytes(stable_json_dumps(descriptor).encode("utf-8"))
        return cache_key, system_prompt_sha, user_text_sha

    def run_json_schema(
        self,
        *,
        metric_id: str,
        sample: MemeSample,
        system_prompt: str,
        user_parts: List[Dict[str, Any]],
        schema_name: str,
        schema: Dict[str, Any],
        temperature: float = 0.0,
        max_output_tokens: int = 400,
        use_cache: bool = True,
        force_refresh: bool = False,
        src_image_sha256: Optional[str] = None,
        gen_image_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        cache_key, sp_sha, ut_sha = self.build_cache_key(
            metric_id=metric_id,
            schema_name=schema_name,
            system_prompt=system_prompt,
            sample=sample,
            user_parts=user_parts,
            src_image_sha256=src_image_sha256,
            gen_image_sha256=gen_image_sha256,
        )

        cache_key_short = cache_key[:12]

        # Cache read
        if use_cache and self.store.enabled and not force_refresh:
            cached = self.store.get(cache_key)
            if cached is not None:
                is_err = bool(cached.get("_cached_is_error", False))
                if (not is_err) or self.cache_cfg.return_cached_errors:
                    logger.info(
                        "CACHE HIT | sample={} | metric={} | key={}",
                        sample.sample_id,
                        metric_id,
                        cache_key_short,
                    )
                    # Record the hit (light touch, does not overwrite payload)
                    self.store.upsert(
                        cache_key=cache_key,
                        judge_model=self.model,
                        metric_id=metric_id,
                        schema_name=schema_name,
                        sample=sample,
                        src_image_sha256=src_image_sha256,
                        gen_image_sha256=gen_image_sha256,
                        system_prompt_sha256=sp_sha,
                        user_text_sha256=ut_sha,
                        payload=cached,
                        is_error=is_err,
                        latency_ms=None,
                        attempts=None,
                        cache_hit=True,
                        overwrite=False,
                    )
                    return cached
                logger.warning(
                    "CACHE HAS ERROR (ignored) | sample={} | metric={} | key={}",
                    sample.sample_id,
                    metric_id,
                    cache_key_short,
                )

        if force_refresh:
            logger.warning(
                "FORCE REFRESH | sample={} | metric={} | key={}",
                sample.sample_id,
                metric_id,
                cache_key_short,
            )
        else:
            logger.info(
                "CACHE MISS | sample={} | metric={} | key={}",
                sample.sample_id,
                metric_id,
                cache_key_short,
            )

        # API call with retries
        last_exc: Optional[Exception] = None
        t0 = time.time()
        attempts_used: Optional[int] = None

        with self._in_flight:
            for attempt in range(self.retry.max_retries + 1):
                attempts_used = attempt + 1
                try:
                    logger.debug(
                        "API CALL | sample={} | metric={} | attempt={}/{}",
                        sample.sample_id,
                        metric_id,
                        attempts_used,
                        self.retry.max_retries + 1,
                    )
                    resp = self.client.responses.create(
                        model=self.model,
                        input=[
                            {
                                "role": "system",
                                "content": [
                                    {"type": "input_text", "text": system_prompt}
                                ],
                            },
                            {"role": "user", "content": user_parts},
                        ],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "strict": True,
                                "schema": schema,
                            }
                        },
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        timeout=self.retry.timeout_s,
                    )
                    parsed = _safe_json_loads(resp.output_text)
                    latency_ms = (time.time() - t0) * 1000.0
                    is_err = "error" in parsed

                    if is_err:
                        logger.error(
                            "API RETURNED ERROR PAYLOAD | sample={} | metric={} | latency_ms={:.1f}",
                            sample.sample_id,
                            metric_id,
                            latency_ms,
                        )
                    else:
                        logger.info(
                            "API OK | sample={} | metric={} | latency_ms={:.1f}",
                            sample.sample_id,
                            metric_id,
                            latency_ms,
                        )

                    # Store (fresh result becomes canonical for that key)
                    if use_cache and self.store.enabled:
                        self.store.upsert(
                            cache_key=cache_key,
                            judge_model=self.model,
                            metric_id=metric_id,
                            schema_name=schema_name,
                            sample=sample,
                            src_image_sha256=src_image_sha256,
                            gen_image_sha256=gen_image_sha256,
                            system_prompt_sha256=sp_sha,
                            user_text_sha256=ut_sha,
                            payload=parsed,
                            is_error=is_err,
                            latency_ms=latency_ms,
                            attempts=attempts_used,
                            cache_hit=False,
                            overwrite=True,
                        )
                        logger.debug(
                            "DB UPSERT | sample={} | metric={} | key={} | overwrite=1",
                            sample.sample_id,
                            metric_id,
                            cache_key_short,
                        )
                    return parsed

                except Exception as e:
                    last_exc = e
                    if attempt >= self.retry.max_retries or not self._is_retryable(e):
                        latency_ms = (time.time() - t0) * 1000.0
                        out = {
                            "error": f"request_failed: {type(e).__name__}: {e}",
                            "attempts": attempts_used,
                        }
                        logger.error(
                            "API FAILED (final) | sample={} | metric={} | attempts={} | latency_ms={:.1f} | err={}",
                            sample.sample_id,
                            metric_id,
                            attempts_used,
                            latency_ms,
                            out["error"],
                        )
                        if use_cache and self.store.enabled:
                            self.store.upsert(
                                cache_key=cache_key,
                                judge_model=self.model,
                                metric_id=metric_id,
                                schema_name=schema_name,
                                sample=sample,
                                src_image_sha256=src_image_sha256,
                                gen_image_sha256=gen_image_sha256,
                                system_prompt_sha256=sp_sha,
                                user_text_sha256=ut_sha,
                                payload=out,
                                is_error=True,
                                latency_ms=latency_ms,
                                attempts=attempts_used,
                                cache_hit=False,
                                overwrite=True,
                            )
                        return out

                    logger.warning(
                        "API FAILED (retrying) | sample={} | metric={} | attempt={}/{} | err={}",
                        sample.sample_id,
                        metric_id,
                        attempts_used,
                        self.retry.max_retries + 1,
                        f"{type(e).__name__}: {e}",
                    )
                    self._sleep_backoff(attempt, e)

        out = {"error": f"request_failed_unknown: {last_exc}"}
        logger.error(
            "API FAILED (unknown) | sample={} | metric={} | err={}",
            sample.sample_id,
            metric_id,
            out["error"],
        )
        if use_cache and self.store.enabled:
            self.store.upsert(
                cache_key=cache_key,
                judge_model=self.model,
                metric_id=metric_id,
                schema_name=schema_name,
                sample=sample,
                src_image_sha256=src_image_sha256,
                gen_image_sha256=gen_image_sha256,
                system_prompt_sha256=sp_sha,
                user_text_sha256=ut_sha,
                payload=out,
                is_error=True,
                latency_ms=(time.time() - t0) * 1000.0,
                attempts=attempts_used,
                cache_hit=False,
                overwrite=True,
            )
        return out


# -----------------------------
# Metric base
# -----------------------------


class Metric:
    metric_id: str = "base"
    name: str = "BaseMetric"
    needs_src_image: bool = False

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        raise NotImplementedError

    def evaluate(
        self,
        client: JudgeClient,
        sample: MemeSample,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> MetricResult:
        schema_name, schema = self.schema()

        t0 = time.time()
        try:
            src_sha = (
                sha256_file(sample.src_image_path) if self.needs_src_image else None
            )
            gen_sha = sha256_file(sample.gen_image_path)

            user_parts = self.build_user_parts(sample)
            payload = client.run_json_schema(
                metric_id=self.metric_id,
                sample=sample,
                system_prompt=self.build_system_prompt(),
                user_parts=user_parts,
                schema_name=schema_name,
                schema=schema,
                src_image_sha256=src_sha,
                gen_image_sha256=gen_sha,
                use_cache=use_cache,
                force_refresh=force_refresh,
            )

            if "error" in payload:
                return MetricResult(
                    metric_id=self.metric_id,
                    metric_name=self.name,
                    score=0.0,
                    label="error",
                    rationale="",
                    evidence=None,
                    raw=payload,
                    error=str(payload.get("error")),
                )

            result = self.to_metric_result(payload, sample)
            return result

        except Exception as e:
            return MetricResult(
                metric_id=self.metric_id,
                metric_name=self.name,
                score=0.0,
                label="error",
                rationale="",
                evidence=None,
                raw=None,
                error=f"metric_exception: {type(e).__name__}: {e}",
            )
        finally:
            elapsed_ms = (time.time() - t0) * 1000.0
            logger.debug(
                "METRIC DONE | sample={} | metric={} | elapsed_ms={:.1f}",
                sample.sample_id,
                self.metric_id,
                elapsed_ms,
            )


# -----------------------------
# Metrics
# -----------------------------


class TargetEmotionAccuracy(Metric):
    metric_id = "A1_TEA"
    name = "Target Emotion Accuracy"

    def build_system_prompt(self) -> str:
        return (
            "You are an expert affective computing judge for memes.\n"
            "Infer the PRIMARY emotion conveyed by the GENERATED meme.\n"
            "Pick exactly one label from the allowed list.\n"
            "Also report confidence (1-5).\n"
            "Be strict: do not confuse 'less negative' with genuinely positive emotion.\n"
            "Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        gen_url = image_to_data_url(sample.gen_image_path)
        return [
            {
                "type": "input_text",
                "text": (
                    f"Allowed emotions: {', '.join(EMOTIONS_8)}\n"
                    f"TARGET emotion: {sample.tgt_emotion}\n"
                    "Determine the PRIMARY emotion of the GENERATED meme."
                ),
            },
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
            {
                "type": "input_text",
                "text": "Output one allowed label and confidence 1(low)-5(high).",
            },
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "tea_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "predicted_emotion": {"type": "string", "enum": EMOTIONS_8},
                "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
            },
            "required": ["predicted_emotion", "confidence", "rationale"],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        pred = payload["predicted_emotion"]
        conf = int(payload["confidence"])
        match = pred == sample.tgt_emotion
        score = 1.0 if match else 0.0
        label = "match" if match else "mismatch"
        evidence = {
            "predicted_emotion": pred,
            "confidence": conf,
            "target": sample.tgt_emotion,
        }
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=label,
            rationale=payload.get("rationale", ""),
            evidence=evidence,
            raw=payload,
        )


class PositiveEmotionStrength(Metric):
    metric_id = "A2_PES"
    name = "Positive Emotion Strength"

    def build_system_prompt(self) -> str:
        return (
            "You are a calibrated rater of emotional intensity.\n"
            "Rate how strongly the GENERATED meme evokes the TARGET emotion.\n"
            "Use an integer 1-5 Likert scale.\n"
            "Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        gen_url = image_to_data_url(sample.gen_image_path)
        return [
            {
                "type": "input_text",
                "text": (
                    f"TARGET emotion: {sample.tgt_emotion}\n"
                    "Rate intensity on 1-5:\n"
                    "1 none, 2 weak, 3 moderate, 4 strong, 5 very strong."
                ),
            },
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "pes_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "likert_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
            },
            "required": ["likert_1_5", "rationale"],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        x = int(payload["likert_1_5"])
        score = likert_to_unit(x, 5)
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=f"likert_{x}",
            rationale=payload.get("rationale", ""),
            evidence={"likert_1_5": x},
            raw=payload,
        )


class EmotionalShiftSufficiency(Metric):
    metric_id = "A3_ESS"
    name = "Emotional Shift Sufficiency"
    needs_src_image = True

    def build_system_prompt(self) -> str:
        return (
            "You are an expert judge of emotional transformation.\n"
            "Compare SOURCE vs GENERATED memes.\n"
            "Rate how sufficient/obvious the shift is from SOURCE emotion to TARGET emotion.\n"
            "Use a 1-5 Likert scale; be strict about 'neutralization'.\n"
            "Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        src_url = image_to_data_url(sample.src_image_path)
        gen_url = image_to_data_url(sample.gen_image_path)
        return [
            {
                "type": "input_text",
                "text": (
                    f"SOURCE emotion label: {sample.src_emotion}\n"
                    f"TARGET emotion label: {sample.tgt_emotion}\n"
                    "Likert 1-5 shift sufficiency:\n"
                    "1 no change, 2 small, 3 moderate, 4 large, 5 very large (clear negative→target-positive)."
                ),
            },
            {"type": "input_text", "text": "SOURCE image:"},
            {"type": "input_image", "image_url": src_url, "detail": "high"},
            {"type": "input_text", "text": "GENERATED image:"},
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "ess_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "likert_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
                "changed_cues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 6,
                },
            },
            "required": ["likert_1_5", "rationale", "changed_cues"],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        x = int(payload["likert_1_5"])
        score = likert_to_unit(x, 5)
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=f"likert_{x}",
            rationale=payload.get("rationale", ""),
            evidence={"likert_1_5": x, "changed_cues": payload.get("changed_cues", [])},
            raw=payload,
        )


class ScenarioConsistency(Metric):
    metric_id = "B1_SC"
    name = "Scenario Consistency"
    needs_src_image = True

    def build_system_prompt(self) -> str:
        return (
            "You are a strict meme scenario consistency checker.\n"
            "The GENERATED meme must preserve the same underlying scenario/joke/event as SOURCE.\n"
            "Emotion may change; scenario must not.\n"
            "Return 1-5 Likert and a brief rationale.\n"
            "Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        src_url = image_to_data_url(sample.src_image_path)
        gen_url = image_to_data_url(sample.gen_image_path)

        extra_lines: List[str] = []
        if sample.src_caption_text:
            extra_lines.append(f"SOURCE caption hint: {sample.src_caption_text}")
        if sample.tgt_caption_text:
            extra_lines.append(f"TARGET caption reference: {sample.tgt_caption_text}")
        if sample.edit_spec:
            extra_lines.append(f"Edit spec (reference): {sample.edit_spec}")
        extra = ("\n" + "\n".join(extra_lines)) if extra_lines else ""

        return [
            {
                "type": "input_text",
                "text": (
                    "Likert 1-5 scenario consistency:\n"
                    "1 different scenario (changed joke/event),\n"
                    "2 mostly different,\n"
                    "3 mixed/unclear,\n"
                    "4 mostly same with minor drift,\n"
                    "5 same scenario; only emotional reframing.\n"
                    "Compare SOURCE vs GENERATED." + extra
                ),
            },
            {"type": "input_text", "text": "SOURCE image:"},
            {"type": "input_image", "image_url": src_url, "detail": "high"},
            {"type": "input_text", "text": "GENERATED image:"},
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "sc_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "likert_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
                "preserved_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 6,
                },
                "changed_elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 6,
                },
            },
            "required": [
                "likert_1_5",
                "rationale",
                "preserved_elements",
                "changed_elements",
            ],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        x = int(payload["likert_1_5"])
        score = likert_to_unit(x, 5)
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=f"likert_{x}",
            rationale=payload.get("rationale", ""),
            evidence={
                "likert_1_5": x,
                "preserved_elements": payload.get("preserved_elements", []),
                "changed_elements": payload.get("changed_elements", []),
            },
            raw=payload,
        )


class CoreEntityPreservation(Metric):
    metric_id = "B2_CEP"
    name = "Core Entity Preservation"
    needs_src_image = True

    def build_system_prompt(self) -> str:
        return (
            "You are an identity/entity preservation auditor for image editing.\n"
            "Judge whether the core entities (main subject/object) are preserved from SOURCE to GENERATED.\n"
            "Return one of: yes / partial / no.\n"
            "Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        src_url = image_to_data_url(sample.src_image_path)
        gen_url = image_to_data_url(sample.gen_image_path)
        return [
            {
                "type": "input_text",
                "text": (
                    "Entity preservation label:\n"
                    "- yes: same main entity/identity clearly preserved\n"
                    "- partial: mostly same but noticeable identity drift or key object replaced\n"
                    "- no: different main entity/identity\n"
                    "Compare SOURCE vs GENERATED."
                ),
            },
            {"type": "input_text", "text": "SOURCE image:"},
            {"type": "input_image", "image_url": src_url, "detail": "high"},
            {"type": "input_text", "text": "GENERATED image:"},
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "cep_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": ["yes", "partial", "no"]},
                "rationale": {"type": "string"},
                "main_entity": {"type": "string"},
            },
            "required": ["label", "rationale", "main_entity"],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        lab = payload["label"]
        score = tri_to_unit(lab)
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=lab,
            rationale=payload.get("rationale", ""),
            evidence={"main_entity": payload.get("main_entity", "")},
            raw=payload,
        )


class VisualGenerationQuality(Metric):
    metric_id = "C1_VGQ"
    name = "Visual Generation Quality"

    def build_system_prompt(self) -> str:
        return (
            "You are a perceptual quality rater for image editing outputs.\n"
            "Rate the GENERATED meme for artifacts/distortions/unnatural edits.\n"
            "Use a 1-5 Likert. Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        gen_url = image_to_data_url(sample.gen_image_path)
        return [
            {
                "type": "input_text",
                "text": (
                    "Likert 1-5 visual quality:\n"
                    "1 unusable (severe artifacts), 2 poor, 3 acceptable, 4 good, 5 excellent/clean.\n"
                    "Focus on: face collapse, edge artifacts (esp. around text), sudden style shifts."
                ),
            },
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "vgq_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "likert_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
                "defects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 8,
                },
            },
            "required": ["likert_1_5", "rationale", "defects"],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        x = int(payload["likert_1_5"])
        score = likert_to_unit(x, 5)
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=f"likert_{x}",
            rationale=payload.get("rationale", ""),
            evidence={"likert_1_5": x, "defects": payload.get("defects", [])},
            raw=payload,
        )


class TextRenderingQuality(Metric):
    metric_id = "C2_TRQ"
    name = "Text Rendering Quality"

    def build_system_prompt(self) -> str:
        return (
            "You are a strict text-rendering quality evaluator for memes.\n"
            "Judge whether text in the GENERATED meme is readable and intact.\n"
            "Use a 1-5 Likert. Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        gen_url = image_to_data_url(sample.gen_image_path)
        hint = (
            f"\nTarget caption reference: {sample.tgt_caption_text}"
            if sample.tgt_caption_text
            else ""
        )
        return [
            {
                "type": "input_text",
                "text": (
                    "Likert 1-5 text quality:\n"
                    "1 unreadable/garbled, 2 mostly unreadable, 3 partially readable, 4 mostly readable, 5 fully readable.\n"
                    + hint
                ),
            },
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "trq_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "likert_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 8,
                },
            },
            "required": ["likert_1_5", "rationale", "issues"],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        x = int(payload["likert_1_5"])
        score = likert_to_unit(x, 5)
        label = f"likert_{x}"
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=label,
            rationale=payload.get("rationale", ""),
            evidence={
                "likert_1_5": x,
                "issues": payload.get("issues", []),
            },
            raw=payload,
        )


class LayoutStructureConsistency(Metric):
    metric_id = "C3_LSC"
    name = "Layout & Structure Consistency"
    needs_src_image = True

    def build_system_prompt(self) -> str:
        return (
            "You are a meme layout/structure consistency checker.\n"
            "Compare SOURCE vs GENERATED for panel count/order and text-box placement.\n"
            "Use 1-5 Likert. Return JSON only."
        )

    def build_user_parts(self, sample: MemeSample) -> List[Dict[str, Any]]:
        src_url = image_to_data_url(sample.src_image_path)
        gen_url = image_to_data_url(sample.gen_image_path)
        return [
            {
                "type": "input_text",
                "text": (
                    "Likert 1-5 structure consistency:\n"
                    "1 broken structure, 2 major drift, 3 moderate drift, 4 minor drift, 5 preserved.\n"
                    "Consider: panel count, panel order, caption regions (top/bottom text boxes), text placement."
                ),
            },
            {"type": "input_text", "text": "SOURCE image:"},
            {"type": "input_image", "image_url": src_url, "detail": "high"},
            {"type": "input_text", "text": "GENERATED image:"},
            {"type": "input_image", "image_url": gen_url, "detail": "high"},
        ]

    def schema(self) -> Tuple[str, Dict[str, Any]]:
        return "lsc_schema", {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "likert_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
                "rationale": {"type": "string"},
                "mismatches": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 8,
                },
            },
            "required": ["likert_1_5", "rationale", "mismatches"],
        }

    def to_metric_result(
        self, payload: Dict[str, Any], sample: MemeSample
    ) -> MetricResult:
        x = int(payload["likert_1_5"])
        score = likert_to_unit(x, 5)
        return MetricResult(
            metric_id=self.metric_id,
            metric_name=self.name,
            score=score,
            label=f"likert_{x}",
            rationale=payload.get("rationale", ""),
            evidence={"likert_1_5": x, "mismatches": payload.get("mismatches", [])},
            raw=payload,
        )


# -----------------------------
# Metric set + aggregation
# -----------------------------


def default_metrics() -> List[Metric]:
    return [
        TargetEmotionAccuracy(),
        PositiveEmotionStrength(),
        EmotionalShiftSufficiency(),
        ScenarioConsistency(),
        CoreEntityPreservation(),
        VisualGenerationQuality(),
        TextRenderingQuality(),
        LayoutStructureConsistency(),
    ]


def compute_ecmrq(results: List[MetricResult]) -> Dict[str, Any]:
    by_id = {r.metric_id: r for r in results}

    tea = by_id.get("A1_TEA")
    pes = by_id.get("A2_PES")
    ess = by_id.get("A3_ESS")
    sc = by_id.get("B1_SC")
    cep = by_id.get("B2_CEP")
    vgq = by_id.get("C1_VGQ")
    trq = by_id.get("C2_TRQ")
    lsc = by_id.get("C3_LSC")

    tea_score = tea.score if tea and tea.error is None else 0.0
    pes_score = pes.score if (pes and pes.error is None and tea_score == 1.0) else 0.0
    ess_score = ess.score if ess and ess.error is None else 0.0
    axis_a = (tea_score + pes_score + ess_score) / 3.0

    b_scores = []
    if sc and sc.error is None:
        b_scores.append(sc.score)
    if cep and cep.error is None:
        b_scores.append(cep.score)
    axis_b = sum(b_scores) / len(b_scores) if b_scores else 0.0

    c_scores = []
    for x in (vgq, trq, lsc):
        if x and x.error is None:
            c_scores.append(x.score)
    axis_c = sum(c_scores) / len(c_scores) if c_scores else 0.0

    overall = (axis_a + axis_b + axis_c) / 3.0
    failed = False
    fail_reasons = []

    return {
        "failed": failed,
        "fail_reasons": fail_reasons,
        "axis_scores": {
            "A_affective": axis_a,
            "B_preservation": axis_b,
            "C_quality": axis_c,
        },
        "ecmrq": overall,
        "gating": {"PES_gated_by_TEA": True},
    }


# -----------------------------
# Pipeline with concurrency + cache refresh controls
# -----------------------------


@dataclass
class ConcurrencyConfig:
    metric_workers: int = 6
    sample_workers: int = 4
    parallel_metrics_per_sample: bool = True


@dataclass
class RunConfig:
    use_cache: bool = True
    force_refresh: bool = False


class JudgePipeline:
    def __init__(
        self,
        client: JudgeClient,
        metrics: Sequence[Metric],
        ccfg: Optional[ConcurrencyConfig] = None,
        rcfg: Optional[RunConfig] = None,
    ):
        self.client = client
        self.metrics = list(metrics)
        self.ccfg = ccfg or ConcurrencyConfig()
        self.rcfg = rcfg or RunConfig()

    def _eval_one_metric(self, metric: Metric, sample: MemeSample) -> MetricResult:
        return metric.evaluate(
            self.client,
            sample,
            use_cache=self.rcfg.use_cache,
            force_refresh=self.rcfg.force_refresh,
        )

    def evaluate_sample(self, sample: MemeSample) -> SampleJudgement:
        logger.info("SAMPLE START | {}", sample.sample_id)
        t0 = time.time()

        if not self.ccfg.parallel_metrics_per_sample or len(self.metrics) <= 1:
            results = [self._eval_one_metric(m, sample) for m in self.metrics]
            results.sort(key=lambda r: r.metric_id)
            elapsed_ms = (time.time() - t0) * 1000.0
            logger.info(
                "SAMPLE DONE | {} | elapsed_ms={:.1f}", sample.sample_id, elapsed_ms
            )
            return SampleJudgement(
                sample.sample_id, sample.src_emotion, sample.tgt_emotion, results
            )

        results: List[MetricResult] = []
        with ThreadPoolExecutor(max_workers=self.ccfg.metric_workers) as ex:
            futs = {
                ex.submit(self._eval_one_metric, m, sample): m for m in self.metrics
            }
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    m = futs[fut]
                    logger.exception(
                        "METRIC FUTURE FAILED | sample={} | metric={}",
                        sample.sample_id,
                        m.metric_id,
                    )
                    results.append(
                        MetricResult(
                            metric_id=m.metric_id,
                            metric_name=m.name,
                            score=0.0,
                            label="error",
                            rationale="",
                            evidence=None,
                            raw=None,
                            error=f"metric_future_exception: {type(e).__name__}: {e}",
                        )
                    )

        results.sort(key=lambda r: r.metric_id)
        elapsed_ms = (time.time() - t0) * 1000.0
        logger.info(
            "SAMPLE DONE | {} | elapsed_ms={:.1f}", sample.sample_id, elapsed_ms
        )
        return SampleJudgement(
            sample.sample_id, sample.src_emotion, sample.tgt_emotion, results
        )

    def evaluate_batch(self, samples: Sequence[MemeSample]) -> List[SampleJudgement]:
        if not samples:
            return []
        logger.info("BATCH START | n={}", len(samples))

        judgements: List[SampleJudgement] = []
        with ThreadPoolExecutor(max_workers=self.ccfg.sample_workers) as ex:
            futs = {ex.submit(self.evaluate_sample, s): s for s in samples}
            done = 0
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    judgements.append(fut.result())
                except Exception as e:
                    logger.exception("SAMPLE FAILED | {}", s.sample_id)
                    judgements.append(
                        SampleJudgement(
                            sample_id=s.sample_id,
                            src_emotion=s.src_emotion,
                            tgt_emotion=s.tgt_emotion,
                            results=[
                                MetricResult(
                                    metric_id="pipeline_error",
                                    metric_name="pipeline_error",
                                    score=0.0,
                                    label="error",
                                    rationale="",
                                    evidence=None,
                                    raw=None,
                                    error=f"sample_exception: {type(e).__name__}: {e}",
                                )
                            ],
                        )
                    )
                done += 1
                if done % 10 == 0 or done == len(samples):
                    logger.info("BATCH PROGRESS | {}/{}", done, len(samples))

        judgements.sort(key=lambda j: j.sample_id)
        logger.info("BATCH DONE | n={}", len(judgements))
        return judgements


# -----------------------------
# Example main
# -----------------------------

if __name__ == "__main__":
    # Example:
    #   LOG_LEVEL=DEBUG python mllm_judge.py
    #   LOG_LEVEL=INFO LOG_FILE=run.log python mllm_judge.py
    #   LOG_LEVEL=TRACE LOG_JSON=1 python mllm_judge.py

    sample = MemeSample(
        sample_id="21.png",
        src_image_path="datasets/Original/21.png",
        src_emotion="angry",
        gen_image_path="Generated/FLUX.2-klein-4B/21.png",
        tgt_emotion="calm",
        src_caption_text="Me when the customer puts their money on the\ncounter instead of my outstretched hand\nMemeCenter.com\n",
        tgt_caption_text='"Me when the customer smiles and thanks me while paying — what a great start to the day!"',
        edit_spec='"Transform the dog’s squint into a calm, content expression with slightly open eyes and a gentle smile. The lighting should be warmer and softer, evoking friendliness and mutual respect."',
    )

    client = JudgeClient(
        model="gpt-5.2",
        retry=RetryConfig(
            max_retries=6,
            base_delay_s=0.6,
            max_delay_s=20.0,
            jitter=0.25,
            timeout_s=120,
        ),
        max_in_flight=8,
        cache=CacheConfig(
            enabled=True, db_path="judge_cache.sqlite3", return_cached_errors=False
        ),
    )

    pipeline = JudgePipeline(
        client=client,
        metrics=default_metrics(),
        ccfg=ConcurrencyConfig(
            metric_workers=6, sample_workers=4, parallel_metrics_per_sample=True
        ),
        rcfg=RunConfig(use_cache=True, force_refresh=False),
    )

    judgement = pipeline.evaluate_sample(sample)
    summary = compute_ecmrq(judgement.results)

    out = {
        "sample": asdict(sample),
        "axis_summary": summary,
        "aggregate_equal_metrics": judgement.aggregate(),
        "results": [asdict(r) for r in judgement.results],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
