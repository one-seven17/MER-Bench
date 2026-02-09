#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

High-concurrency batch evaluator for meme emotion reframing.

Behaviors:
1) Uses judge.py concurrency + retry:
   - JudgeClient: retry + in-flight semaphore
   - JudgePipeline: ThreadPoolExecutor concurrency
2) factor_view and model_view are mutually exclusive.
3) visual_type / sentiment_polarity / layout_type: each can be specified with at most ONE value, or omitted.
4) model_view: evaluates all selected stratum rows; per run samples rows then evaluates ALL models.
5) factor_view: evaluates ONLY the specified filter-combination stratum (or all if no filters).
   - Sampling is evenly split across models each run.
   - Aggregation is from stratum perspective (models marginalized), report mean/std/CI over runs.
6) Supports configurable concurrency: --sample-workers and --max-in-flight.
7) Optional cost accumulation excludes cache hits.
8) --dry-run: only stats + sampling plan, no judge calls.
9) Save per-run jsonl outputs by model under:
     output_save_dir/{exp}/{model_view|factor_view}/{model}/{run_idx}.jsonl
   Each line records sampled row_id/gen_id and the out item from pipeline.
10) Save final summary JSON under:
     out_dir/{model_view|factor_view}/{filename}.json
   where filename encodes models + factor filters + sampling.

Notes:
- This script assumes you place it next to judge.py and judge.py exports:
  CacheConfig, ConcurrencyConfig, JudgeClient, JudgeConfig, JudgePipeline, MemeSample,
  RetryConfig, RunConfig
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

try:
    from judge import (
        CacheConfig,
        ConcurrencyConfig,
        JudgeClient,
        JudgeConfig,
        JudgePipeline,
        MemeSample,
        RetryConfig,
        RunConfig,
    )
except Exception as e:
    raise RuntimeError(
        "Cannot import from judge.py. Put this main.py next to judge.py or fix PYTHONPATH.\n"
        f"Import error: {type(e).__name__}: {e}"
    )


# =============================
# Index loading
# =============================


def load_index(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - .json: array of objects
      - .jsonl: one JSON object per line
    Tolerates trailing commas in JSONL lines.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON root is not a list")
        return data

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f2:
        for ln, line in enumerate(f2, start=1):
            s = line.strip()
            if not s:
                continue
            if s.endswith(","):
                s = s[:-1].rstrip()
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    logger.warning("Line {} is not an object; skipped.", ln)
            except Exception as e:
                logger.warning("JSONL parse failed at line {}: {}", ln, e)
    return rows


# =============================
# Filtering helpers (single-or-none)
# =============================


def _match_single(value: str, selected: Optional[str]) -> bool:
    return True if selected is None else value == selected


def filter_rows_single(
    rows: Sequence[Dict[str, Any]],
    visual_type: Optional[str],
    sentiment_polarity: Optional[str],
    layout_type: Optional[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        vt = str(r.get("visual_type", ""))
        sp = str(r.get("sentiment_polarity", ""))
        lt = str(r.get("layout_type", ""))
        if (
            _match_single(vt, visual_type)
            and _match_single(sp, sentiment_polarity)
            and _match_single(lt, layout_type)
        ):
            out.append(r)
    return out


def aggregate_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        k = (
            str(r.get("visual_type", "")),
            str(r.get("sentiment_polarity", "")),
            str(r.get("layout_type", "")),
        )
        agg[k] = agg.get(k, 0) + 1

    return {
        "by_visual_type_sentiment_layout": [
            {
                "visual_type": k[0],
                "sentiment_polarity": k[1],
                "layout_type": k[2],
                "count": v,
            }
            for k, v in sorted(agg.items(), key=lambda x: (-x[1], x[0]))
        ]
    }


# =============================
# Missing checks
# =============================


def build_src_path(src_dir: str, row: Dict[str, Any]) -> str:
    return os.path.join(src_dir, str(row["id"]))


def build_gen_path(gen_root: str, model: str, row: Dict[str, Any]) -> str:
    return os.path.join(gen_root, model, str(row["gen_id"]))


def verify_missing(
    rows: Sequence[Dict[str, Any]],
    src_dir: str,
    gen_root: str,
    models: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    missing_report: Dict[str, Any] = {
        "missing_source": 0,
        "missing_by_model": {m: 0 for m in models},
        "missing_examples": {m: [] for m in models},
        "total": len(rows),
    }

    for r in rows:
        src_path = build_src_path(src_dir, r)
        if not os.path.exists(src_path):
            missing_report["missing_source"] += 1
            continue

        missing_any = False
        for m in models:
            gen_path = build_gen_path(gen_root, m, r)
            if not os.path.exists(gen_path):
                missing_report["missing_by_model"][m] += 1
                if len(missing_report["missing_examples"][m]) < 20:
                    missing_report["missing_examples"][m].append(
                        {
                            "id": r.get("id"),
                            "gen_id": r.get("gen_id"),
                            "expected_path": gen_path,
                        }
                    )
                missing_any = True
        if missing_any:
            continue

        kept.append(r)

    missing_report["kept"] = len(kept)
    return kept, missing_report


# =============================
# Metrics extraction
# =============================


METRIC_KEYS = [
    "visual_generation_quality_1_5",
    "visual_emotion_accuracy_1_5",
    "text_generation_quality_1_5",
    "text_emotion_accuracy_1_5",
    "layout_consistency_accuracy",
    "tgt_emotion_hit",
    "perceived_emotion_shift_1_5",
    "overall_generation_quality_1_5",
]


def safe_get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def extract_metrics(payload: Dict[str, Any], tgt_emotion: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    vq = safe_get(payload, "visual_assessment", "generation_quality_1_5")
    va = safe_get(payload, "visual_assessment", "emotion_accuracy_1_5")
    tq = safe_get(payload, "text_assessment", "generation_quality_1_5")
    ta = safe_get(payload, "text_assessment", "emotion_accuracy_1_5")
    lc = safe_get(payload, "layout_consistency", "consistent")
    cls_label = safe_get(payload, "overall_emotion_classification", "label")
    shift = safe_get(payload, "perceived_emotion_shift", "magnitude_1_5")
    oq = safe_get(
        payload, "overall_generation_quality", "overall_generation_quality_1_5"
    )

    def to_1_5(x: Any) -> Optional[float]:
        if isinstance(x, (int, float)) and 1 <= float(x) <= 5:
            return float(x)
        return None

    out["visual_generation_quality_1_5"] = to_1_5(vq) or 0.0
    out["visual_emotion_accuracy_1_5"] = to_1_5(va) or 0.0
    out["text_generation_quality_1_5"] = to_1_5(tq) or 0.0
    out["text_emotion_accuracy_1_5"] = to_1_5(ta) or 0.0
    out["perceived_emotion_shift_1_5"] = to_1_5(shift) or 0.0
    out["overall_generation_quality_1_5"] = to_1_5(oq) or 0.0

    out["layout_consistency_accuracy"] = 1.0 if isinstance(lc, bool) and lc else 0.0
    out["tgt_emotion_hit"] = (
        1.0 if (isinstance(cls_label, str) and cls_label == tgt_emotion) else 0.0
    )
    return out


# =============================
# Cache / Cost helpers
# =============================


def cache_flag(payload: Dict[str, Any]) -> bool:
    if "error" in payload:
        return False
    return ("_cache_key" in payload) and ("_cached_is_error" in payload)


def extract_cost_usd_from_cost_info(cost_info: Optional[Dict[str, Any]]) -> float:
    if not isinstance(cost_info, dict):
        return 0.0
    v = cost_info.get("estimated_cost_usd")
    return float(v) if isinstance(v, (int, float)) else 0.0


# =============================
# Sampling utilities
# =============================


def allocate_quota(total: int, n_buckets: int, rng: random.Random) -> List[int]:
    if n_buckets <= 0:
        return []
    base = total // n_buckets
    rem = total - base * n_buckets
    q = [base] * n_buckets
    if rem > 0:
        idxs = list(range(n_buckets))
        rng.shuffle(idxs)
        for i in range(rem):
            q[idxs[i]] += 1
    return q


def sample_rows(
    rows: Sequence[Dict[str, Any]], k: int, rng: random.Random
) -> List[Dict[str, Any]]:
    if k <= 0 or not rows:
        return []
    if len(rows) >= k:
        return rng.sample(list(rows), k=k)
    return [rng.choice(list(rows)) for _ in range(k)]


# =============================
# Output saving helpers
# =============================


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))


def write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _safe_slug(s: str) -> str:
    # Keep filenames portable.
    return "".join(c if (c.isalnum() or c in "._-+=") else "_" for c in s)


def build_experiment_dirname(
    *,
    models: Sequence[str],
    visual_type: Optional[str],
    sentiment_polarity: Optional[str],
    layout_type: Optional[str],
    rationale_first: bool,
    images_first: bool,
    sample_size: int,
    runs: int,
    seed: int,
) -> str:
    parts: List[str] = []
    parts.append("models=" + "+".join(models))
    parts.append(f"vt={visual_type or 'all'}")
    parts.append(f"sp={sentiment_polarity or 'all'}")
    parts.append(f"lt={layout_type or 'all'}")
    parts.append(f"n={sample_size}")
    parts.append(f"runs={runs}")
    parts.append(f"seed={seed}")
    parts.append("ra=first" if rationale_first else "ra=last")
    parts.append("im=first" if images_first else "im=last")
    return _safe_slug("__".join(parts))


def build_summary_filename(
    *,
    models: Sequence[str],
    visual_type: Optional[str],
    sentiment_polarity: Optional[str],
    layout_type: Optional[str],
    rationale_first: bool,
    images_first: bool,
    sample_size: int,
    runs: int,
    seed: int,
) -> str:
    parts: List[str] = []
    parts.append("models=" + "+".join(models))
    parts.append(f"vt={visual_type or 'all'}")
    parts.append(f"sp={sentiment_polarity or 'all'}")
    parts.append(f"lt={layout_type or 'all'}")
    parts.append(f"n={sample_size}")
    parts.append(f"runs={runs}")
    parts.append(f"seed={seed}")
    parts.append("ra=first" if rationale_first else "ra=last")
    parts.append("im=first" if images_first else "im=last")
    return _safe_slug("__".join(parts)) + ".json"


def build_run_jsonl_path(
    *,
    base_output_dir: str,
    mode: str,
    model: str,
    run_idx: int,
) -> str:
    return os.path.join(base_output_dir, mode, model, f"{run_idx}.jsonl")


# =============================
# Sample builder (embed model in sample_id for bookkeeping)
# =============================


def make_sample(
    src_dir: str, gen_root: str, model: str, row: Dict[str, Any]
) -> MemeSample:
    row_id = str(row.get("id"))
    return MemeSample(
        sample_id=f"{row_id}::{model}",
        src_image_path=os.path.join(src_dir, row_id),
        gen_image_path=os.path.join(gen_root, model, str(row.get("gen_id"))),
        src_emotion=str(row.get("det_emo", "")),
        tgt_emotion=str(row.get("tgt_emo", "")),
        src_caption_text=row.get("text"),
        tgt_caption_text=row.get("pos_txt"),
        edit_instruction=row.get("spec"),
    )


def parse_model_from_sample_id(sample_id: str) -> str:
    return sample_id.split("::", 1)[1] if "::" in sample_id else ""


def parse_row_id_from_sample_id(sample_id: str) -> str:
    return sample_id.split("::", 1)[0] if "::" in sample_id else sample_id


# =============================
# Stats
# =============================


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def ci95_of_replicates(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    ys = sorted(xs)
    lo = ys[int(0.025 * (len(ys) - 1))]
    hi = ys[int(0.975 * (len(ys) - 1))]
    return lo, hi


# =============================
# Evaluation bookkeeping
# =============================


def init_book(models: Sequence[str]) -> Dict[str, Any]:
    return {
        "per_model": {
            m: {"api_calls": 0, "cache_hits": 0, "cost_usd": 0.0, "errors": 0}
            for m in models
        },
        "overall": {"api_calls": 0, "cache_hits": 0, "cost_usd": 0.0, "errors": 0},
    }


def update_book(
    book: Dict[str, Any],
    model: str,
    payload: Dict[str, Any],
    cost_info: Optional[Dict[str, Any]],
    compute_cost: bool,
) -> None:
    if model not in book["per_model"]:
        book["per_model"][model] = {
            "api_calls": 0,
            "cache_hits": 0,
            "cost_usd": 0.0,
            "errors": 0,
        }

    if "error" in payload:
        book["per_model"][model]["errors"] += 1
        book["overall"]["errors"] += 1
        return

    if cache_flag(payload):
        book["per_model"][model]["cache_hits"] += 1
        book["overall"]["cache_hits"] += 1
        return

    book["per_model"][model]["api_calls"] += 1
    book["overall"]["api_calls"] += 1
    if compute_cost:
        c = extract_cost_usd_from_cost_info(cost_info)
        book["per_model"][model]["cost_usd"] += c
        book["overall"]["cost_usd"] += c


# =============================
# Evaluation modes
# =============================


def evaluate_model_view(
    *,
    rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
    src_dir: str,
    gen_root: str,
    pipeline: JudgePipeline,
    runs: int,
    sample_size: int,
    seed: int,
    compute_cost: bool,
    base_output_dir: str,
    save_outputs: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    # replicate means per model per metric (length = runs)
    rep_means: Dict[str, Dict[str, List[float]]] = {
        m: {k: [] for k in METRIC_KEYS} for m in models
    }
    book = init_book(models)

    row_by_id: Dict[str, Dict[str, Any]] = {str(r.get("id")): r for r in rows}

    for r_i in range(runs):
        batch_rows = sample_rows(rows, k=sample_size, rng=rng)

        samples: List[MemeSample] = []
        tgt_by_sample_id: Dict[str, str] = {}
        for row in batch_rows:
            for m in models:
                s = make_sample(src_dir, gen_root, m, row)
                samples.append(s)
                tgt_by_sample_id[s.sample_id] = str(row.get("tgt_emo", ""))

        results = pipeline.evaluate_batch(samples)

        run_vals: Dict[str, Dict[str, List[float]]] = {
            m: {k: [] for k in METRIC_KEYS} for m in models
        }
        save_buf: Dict[str, List[Dict[str, Any]]] = (
            {m: [] for m in models} if save_outputs else {}
        )

        for item in results:
            sample_obj = item.get("sample", {})
            sample_id = str(sample_obj.get("sample_id", ""))
            model = parse_model_from_sample_id(sample_id)
            payload = item.get("judge", {}) if isinstance(item, dict) else {}
            cost_info = item.get("cost") if isinstance(item, dict) else None

            update_book(book, model, payload, cost_info, compute_cost)

            if save_outputs and model in save_buf:
                row_id = parse_row_id_from_sample_id(sample_id)
                row = row_by_id.get(row_id, {})
                save_buf[model].append(
                    {
                        "run": r_i,
                        "row_id": row_id,
                        "gen_id": row.get("gen_id"),
                        "model": model,
                        "out": item,
                    }
                )

            if not isinstance(payload, dict) or "error" in payload:
                continue

            tgt = tgt_by_sample_id.get(sample_id, "")
            metrics = extract_metrics(payload, tgt)
            for kk, vv in metrics.items():
                run_vals[model][kk].append(float(vv))

        if save_outputs:
            for m in models:
                path = build_run_jsonl_path(
                    base_output_dir=base_output_dir,
                    mode="model_view",
                    model=m,
                    run_idx=r_i,
                )
                write_jsonl(path, save_buf.get(m, []))

        for m in models:
            for kk in METRIC_KEYS:
                rep_means[m][kk].append(mean(run_vals[m][kk]))

        logger.info("[model_view] Run {}/{} done.", r_i + 1, runs)

    summary: Dict[str, Any] = {"models": {}, "bookkeeping": book}
    for m in models:
        mm: Dict[str, Any] = {}
        for kk in METRIC_KEYS:
            xs = rep_means[m][kk]
            mm[kk] = {
                "mean_over_runs": mean(xs),
                "std_over_runs": std(xs),
                "ci95_over_runs": list(ci95_of_replicates(xs)),
                "runs": runs,
                "replicate_means": xs,
            }
        summary["models"][m] = mm
    return summary


def evaluate_factor_view_stratum(
    *,
    rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
    src_dir: str,
    gen_root: str,
    pipeline: JudgePipeline,
    runs: int,
    sample_size: int,
    seed: int,
    compute_cost: bool,
    stratum_label: Dict[str, Optional[str]],
    base_output_dir: str,
    save_outputs: bool,
) -> Dict[str, Any]:
    """
    factor_view (models marginalized), only on the specified stratum:
      - Each run: split sample_size evenly across models, sample rows, evaluate, pool metrics across models.
      - Over runs: mean/std/CI of pooled means.
    """
    rng = random.Random(seed)

    rep_means: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    book = init_book(models)
    row_by_id: Dict[str, Dict[str, Any]] = {str(r.get("id")): r for r in rows}

    for r_i in range(runs):
        quotas = allocate_quota(sample_size, len(models), rng)

        samples: List[MemeSample] = []
        tgt_by_sample_id: Dict[str, str] = {}

        for m, q in zip(models, quotas):
            mbatch = sample_rows(rows, k=q, rng=rng)
            for row in mbatch:
                s = make_sample(src_dir, gen_root, m, row)
                samples.append(s)
                tgt_by_sample_id[s.sample_id] = str(row.get("tgt_emo", ""))

        results = pipeline.evaluate_batch(samples)

        pooled_vals: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
        save_buf: Dict[str, List[Dict[str, Any]]] = (
            {m: [] for m in models} if save_outputs else {}
        )

        for item in results:
            sample_obj = item.get("sample", {})
            sample_id = str(sample_obj.get("sample_id", ""))
            model = parse_model_from_sample_id(sample_id)
            payload = item.get("judge", {}) if isinstance(item, dict) else {}
            cost_info = item.get("cost") if isinstance(item, dict) else None

            update_book(book, model, payload, cost_info, compute_cost)

            if save_outputs and model in save_buf:
                row_id = parse_row_id_from_sample_id(sample_id)
                row = row_by_id.get(row_id, {})
                save_buf[model].append(
                    {
                        "run": r_i,
                        "row_id": row_id,
                        "gen_id": row.get("gen_id"),
                        "model": model,
                        "out": item,
                    }
                )

            if not isinstance(payload, dict) or "error" in payload:
                continue

            tgt = tgt_by_sample_id.get(sample_id, "")
            metrics = extract_metrics(payload, tgt)
            for kk, vv in metrics.items():
                pooled_vals[kk].append(float(vv))

        if save_outputs:
            for m in models:
                path = build_run_jsonl_path(
                    base_output_dir=base_output_dir,
                    mode="factor_view",
                    model=m,
                    run_idx=r_i,
                )
                write_jsonl(path, save_buf.get(m, []))

        for kk in METRIC_KEYS:
            rep_means[kk].append(mean(pooled_vals[kk]))

        logger.info(
            "[factor_view:stratum] Run {}/{} done. (n_samples={}, n_models={})",
            r_i + 1,
            runs,
            len(samples),
            len(models),
        )

    metrics_summary: Dict[str, Any] = {}
    for kk, xs in rep_means.items():
        metrics_summary[kk] = {
            "mean_over_runs": mean(xs),
            "std_over_runs": std(xs),
            "ci95_over_runs": list(ci95_of_replicates(xs)),
            "runs": runs,
            "replicate_means": xs,
        }

    return {
        "stratum": stratum_label,
        "group_size": len(rows),
        "models_marginalized": True,
        "metrics": metrics_summary,
        "bookkeeping": book,
        "sampling": {
            "sample_size_per_run_total": sample_size,
            "even_split_over_models": True,
        },
    }


# =============================
# Dry run stats
# =============================


def print_dry_run_stats(
    *,
    rows_all: Sequence[Dict[str, Any]],
    rows_filtered: Sequence[Dict[str, Any]],
    rows_kept: Sequence[Dict[str, Any]],
    missing_report: Dict[str, Any],
    models: Sequence[str],
    runs: int,
    sample_size: int,
    mode: str,
) -> None:
    rng = random.Random(0)
    quotas = allocate_quota(sample_size, len(models), rng)
    plan = {m: q for m, q in zip(models, quotas)}

    logger.info("=== DRY RUN STATS ===")
    logger.info("Mode: {}", mode)
    logger.info("Index total rows: {}", len(rows_all))
    logger.info("After stratum filters: {}", len(rows_filtered))
    logger.info("After missing-check kept: {}", len(rows_kept))
    logger.info("Missing report: {}", missing_report)
    logger.info("Sampling: runs={}, sample_size_per_run_total={}", runs, sample_size)
    logger.info("Per-run per-model quota (even split): {}", plan)

    # model_view total calls per run = sample_size * n_models
    if mode == "model_view":
        logger.info(
            "Planned evaluations (per run): {} rows × {} models = {} samples",
            sample_size,
            len(models),
            sample_size * len(models),
        )
        logger.info(
            "Planned evaluations (total): {}",
            runs * sample_size * len(models),
        )
    else:
        logger.info(
            "Planned evaluations (per run): {} (split over {} models) = {} samples",
            sample_size,
            len(models),
            sample_size,
        )
        logger.info("Planned evaluations (total): {}", runs * sample_size)


# =============================
# Model discovery
# =============================


def discover_models(gen_root: str) -> List[str]:
    if not os.path.isdir(gen_root):
        raise ValueError(f"gen_root is not a directory: {gen_root}")

    models: List[str] = []
    for name in sorted(os.listdir(gen_root)):
        path = os.path.join(gen_root, name)
        if os.path.isdir(path):
            models.append(name)

    if not models:
        raise ValueError(f"No model subdirectories found under gen_root={gen_root}")
    return models


# =============================
# CLI
# =============================


def _enforce_single_or_none(name: str, values: Optional[List[str]]) -> Optional[str]:
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(
            f"--{name} can specify at most ONE value (or omit it). Got: {values}"
        )
    return values[0]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate meme emotion reframing outputs with high concurrency."
    )

    p.add_argument(
        "--index",
        default="./data/index_final.json",
        help="Path to index file (.json or .jsonl).",
    )
    p.add_argument(
        "--src-dir", default="./data/Original", help="Directory of source images."
    )
    p.add_argument(
        "--gen-root",
        default="./data/EditedResults",
        help="Root directory of generated images (subdirs are model names).",
    )

    p.add_argument(
        "--models",
        default=None,
        nargs="+",
        help="One or more model subdir names under --gen-root. If omitted, auto-discover all subdirs.",
    )

    p.add_argument(
        "--visual-type",
        nargs="*",
        default=None,
        help="Filter: ONE visual_type value (or omit).",
    )
    p.add_argument(
        "--sentiment-polarity",
        nargs="*",
        default=None,
        help="Filter: ONE sentiment_polarity value (or omit).",
    )
    p.add_argument(
        "--layout-type",
        nargs="*",
        default=None,
        help="Filter: ONE layout_type value (or omit).",
    )

    p.add_argument(
        "--runs", type=int, default=10, help="Number of repeated runs (default 10)."
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Total sample size per run (default 50).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")

    p.add_argument(
        "--use-cache", action="store_true", help="Enable SQLite cache read/write."
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force API refresh even if cache exists.",
    )
    p.add_argument("--db", default="db.sqlite3", help="SQLite cache db path.")

    p.add_argument(
        "--judge-model",
        default="gemini-3-pro-preview",
        help="Judge model name (default gemini-3-pro-preview).",
    )
    p.add_argument("--max-output-tokens", type=int, default=900)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument(
        "--rationale-first", action="store_true", help="Rationales first in schema."
    )
    p.add_argument(
        "--rationale-last", action="store_true", help="Rationales last in schema."
    )
    p.add_argument(
        "--images-first", action="store_true", help="Images before instruction text."
    )
    p.add_argument(
        "--images-last", action="store_true", help="Images after instruction text."
    )

    p.add_argument(
        "--compute-cost",
        action="store_true",
        help="Include cost sum (exclude cache hits).",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--mode-model-view", action="store_true", help="Run model_view only."
    )
    g.add_argument(
        "--mode-factor-view",
        action="store_true",
        help="Run factor_view only (only specified stratum; models marginalized).",
    )

    p.add_argument(
        "--sample-workers",
        type=int,
        default=16,
        help="Pipeline sample-level worker threads (default 16).",
    )
    p.add_argument(
        "--max-in-flight",
        type=int,
        default=128,
        help="Client max in-flight requests (default 128).",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="Per-request timeout seconds (default 120).",
    )
    p.add_argument(
        "--max-retries", type=int, default=6, help="Max retries (default 6)."
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print stats and sampling plan; do not call judge.",
    )

    p.add_argument(
        "--output-save-dir",
        default="./eval_outputs",
        help="Directory to save per-run jsonl outputs.",
    )
    p.add_argument(
        "--no-output-save",
        action="store_true",
        help="Disable saving per-run jsonl outputs.",
    )

    p.add_argument(
        "--out",
        default="./summaries",
        help="Summary output ROOT directory. Summary is saved to out/{mode}/{filename}.json",
    )
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    # enforce single-or-none filters
    try:
        visual_type = _enforce_single_or_none("visual-type", args.visual_type)
        sentiment_polarity = _enforce_single_or_none(
            "sentiment-polarity", args.sentiment_polarity
        )
        layout_type = _enforce_single_or_none("layout-type", args.layout_type)
    except ValueError as e:
        logger.error(str(e))
        return 2

    if args.rationale_first and args.rationale_last:
        logger.error("Choose only one: --rationale-first or --rationale-last")
        return 2
    rationale_first = False if args.rationale_last else True

    if args.images_first and args.images_last:
        logger.error("Choose only one: --images-first or --images-last")
        return 2
    images_first = False if args.images_last else True

    if args.sample_size <= 0 or args.runs <= 0:
        logger.error("--sample-size and --runs must be > 0")
        return 2

    mode = "model_view" if args.mode_model_view else "factor_view"

    # load + filter
    rows_all = load_index(args.index)
    logger.info("Loaded index rows: {}", len(rows_all))
    overview_before = aggregate_counts(rows_all)

    rows_filtered = filter_rows_single(
        rows_all,
        visual_type=visual_type,
        sentiment_polarity=sentiment_polarity,
        layout_type=layout_type,
    )
    logger.info("After field filters (stratum): {}", len(rows_filtered))

    # resolve models
    if not args.models:
        logger.info(
            "No --models specified. Discovering all model subdirectories under {}",
            args.gen_root,
        )
        try:
            models = discover_models(args.gen_root)
        except Exception as e:
            logger.error(str(e))
            return 2
    else:
        models = list(args.models)
    logger.info("Using models: {}", models)

    # missing-check
    rows_kept, missing_report = verify_missing(
        rows_filtered, args.src_dir, args.gen_root, models
    )
    logger.info("After missing-check keep: {}", len(rows_kept))
    if missing_report["missing_source"] > 0 or any(
        missing_report["missing_by_model"].values()
    ):
        logger.warning("Missing report: {}", missing_report)

    overview_after = aggregate_counts(rows_kept)
    if not rows_kept:
        logger.error("No samples left after filtering + missing-check.")
        return 1

    # experiment output dir for per-run payloads
    exp_dirname = build_experiment_dirname(
        models=models,
        visual_type=visual_type,
        sentiment_polarity=sentiment_polarity,
        layout_type=layout_type,
        rationale_first=rationale_first,
        images_first=images_first,
        sample_size=args.sample_size,
        runs=args.runs,
        seed=args.seed,
    )
    base_output_dir = os.path.join(args.output_save_dir, exp_dirname)
    save_outputs = not bool(args.no_output_save)

    # dry run
    if bool(args.dry_run):
        print_dry_run_stats(
            rows_all=rows_all,
            rows_filtered=rows_filtered,
            rows_kept=rows_kept,
            missing_report=missing_report,
            models=models,
            runs=int(args.runs),
            sample_size=int(args.sample_size),
            mode=mode,
        )
        logger.info("Planned per-run jsonl save base dir: {}", base_output_dir)
        return 0

    # ensure save dirs
    if save_outputs:
        for m in models:
            _ensure_dir(os.path.join(base_output_dir, mode, m))

    # build client + pipeline
    client = JudgeClient(
        model=args.judge_model,
        retry=RetryConfig(
            max_retries=int(args.max_retries), timeout_s=float(args.timeout_s)
        ),
        max_in_flight=int(args.max_in_flight),
        cache=CacheConfig(
            enabled=bool(args.use_cache),
            db_path=str(args.db),
            return_cached_errors=False,
        ),
    )

    cfg = JudgeConfig(
        rationale_first=bool(rationale_first),
        images_first=bool(images_first),
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
    )
    # align judge-side cost enablement
    try:
        cfg.cost.enabled = bool(args.compute_cost)
    except Exception:
        pass

    pipeline = JudgePipeline(
        client=client,
        cfg=cfg,
        ccfg=ConcurrencyConfig(sample_workers=int(args.sample_workers)),
        rcfg=RunConfig(
            use_cache=bool(args.use_cache), force_refresh=bool(args.force_refresh)
        ),
    )

    # evaluate
    if mode == "model_view":
        results = {
            "model_view": evaluate_model_view(
                rows=rows_kept,
                models=models,
                src_dir=args.src_dir,
                gen_root=args.gen_root,
                pipeline=pipeline,
                runs=int(args.runs),
                sample_size=int(args.sample_size),
                seed=int(args.seed),
                compute_cost=bool(args.compute_cost),
                base_output_dir=base_output_dir,
                save_outputs=save_outputs,
            )
        }
    else:
        stratum_label = {
            "visual_type": visual_type,
            "sentiment_polarity": sentiment_polarity,
            "layout_type": layout_type,
        }
        results = {
            "factor_view": evaluate_factor_view_stratum(
                rows=rows_kept,
                models=models,
                src_dir=args.src_dir,
                gen_root=args.gen_root,
                pipeline=pipeline,
                runs=int(args.runs),
                sample_size=int(args.sample_size),
                seed=int(args.seed),
                compute_cost=bool(args.compute_cost),
                stratum_label=stratum_label,
                base_output_dir=base_output_dir,
                save_outputs=save_outputs,
            )
        }

    out_obj: Dict[str, Any] = {
        "config": {
            "index": args.index,
            "src_dir": args.src_dir,
            "gen_root": args.gen_root,
            "models": models,
            "filters": {
                "visual_type": visual_type,
                "sentiment_polarity": sentiment_polarity,
                "layout_type": layout_type,
            },
            "sampling": {
                "runs": int(args.runs),
                "sample_size": int(args.sample_size),
                "seed": int(args.seed),
            },
            "judge": {
                "judge_model": args.judge_model,
                "rationale_first": bool(rationale_first),
                "images_first": bool(images_first),
                "temperature": float(args.temperature),
                "max_output_tokens": int(args.max_output_tokens),
                "compute_cost": bool(args.compute_cost),
            },
            "cache": {
                "enabled": bool(args.use_cache),
                "db": str(args.db),
                "force_refresh": bool(args.force_refresh),
            },
            "concurrency": {
                "sample_workers": int(args.sample_workers),
                "max_in_flight": int(args.max_in_flight),
            },
            "retry": {
                "max_retries": int(args.max_retries),
                "timeout_s": float(args.timeout_s),
            },
            "mode": mode,
            "saving": {
                "enabled": bool(save_outputs),
                "output_save_dir": args.output_save_dir,
                "experiment_dir": exp_dirname,
                "per_run_base_dir": base_output_dir,
                "layout": "{output_save_dir}/{experiment}/{mode}/{model}/{run}.jsonl",
            },
        },
        "overview_before_filters": overview_before,
        "missing_report": missing_report,
        "overview_after_missing_filter": overview_after,
        "results": results,
        "notes": {
            "dry_run": "If --dry-run is set, prints stats and exits without calling judge.",
            "factor_view_behavior": "factor_view evaluates ONLY the specified filter-combination stratum. Sampling is evenly split across models; metrics pooled across models and summarized with mean/std/CI over runs.",
            "model_view_behavior": "model_view samples rows each run and evaluates ALL models on the same sampled rows.",
            "cost_behavior": "Cost is summed ONLY for non-cache calls, using cost_info['estimated_cost_usd'] returned by judge.py.",
            "save_outputs": "Per-run outputs are saved as JSONL per model; each line includes row_id/gen_id/model and the full pipeline output item.",
        },
    }

    # save summary JSON to out/{mode}/{filename}.json
    out_root = args.out or "."
    mode_dir = os.path.join(out_root, mode)
    _ensure_dir(mode_dir)

    fname = build_summary_filename(
        models=models,
        visual_type=visual_type,
        sentiment_polarity=sentiment_polarity,
        layout_type=layout_type,
        runs=int(args.runs),
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        rationale_first=bool(rationale_first),
        images_first=bool(images_first),
    )
    out_path = os.path.join(mode_dir, fname)

    write_json(out_path, out_obj)
    logger.info("Wrote summary to {}", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
