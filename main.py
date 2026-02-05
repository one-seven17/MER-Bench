#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Batch evaluator for meme emotion reframing:
- Load index (json or jsonl).
- Verify missing generations per model and filter out.
- Filter by visual_type/sentiment_polarity/layout_type.
- Two evaluation modes (can enable both):
    (1) model_view: per-model metrics with uniform sampling over (visual_type,sentiment_polarity,layout_type) combinations.
    (2) factor_view: per-factor (visual_type / sentiment_polarity / layout_type) metrics;
                    within each factor value, per-run sample budget is evenly split across models.
- Output per-model metric performance + (optional) cost summary (exclude cache hits).

IMPORTANT:
- This script expects your judge.py's JudgeClient.judge(...) returns (payload, cost_info).
  payload: dict (strict JSON result)
  cost_info: dict like:
    {
      "model": ...,
      "pricing_usd_per_1m_tokens": {...} or ...,
      "usage_tokens": {
        "input_tokens": ...,
        "cached_input_tokens": ...,
        "output_tokens": ...,
        "reasoning_tokens": ...,
      },
      "estimated_cost_usd": ...,
      "latency_ms": ...,
    }
- Cache hits are counted by presence of payload["_cache_key"] and payload["_cached_is_error"] and no payload["error"].
  Cost is summed ONLY when NOT cache hit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# ---- import your judge module ----
# expects judge.py exposes these symbols
try:
    from judge import (
        CacheConfig,
        JudgeClient,
        JudgeConfig,
        MemeSample,
        RetryConfig,
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
    Also tolerates trailing commas in lines by best-effort stripping.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Heuristic: if starts with "[" treat as JSON array
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("JSON root is not a list")
            return data
        except Exception as e:
            raise ValueError(f"Failed to parse JSON array: {e}")

    # Otherwise JSONL
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f2:
        for ln, line in enumerate(f2, start=1):
            s = line.strip()
            if not s:
                continue
            # tolerate trailing commas like: {...},
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
# Filtering / grouping helpers
# =============================


def key3(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("visual_type", "")),
        str(row.get("sentiment_polarity", "")),
        str(row.get("layout_type", "")),
    )


def match_optional(value: str, allowed: Optional[Sequence[str]]) -> bool:
    if not allowed:
        return True
    return value in set(allowed)


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    visual_type: Optional[Sequence[str]],
    sentiment_polarity: Optional[Sequence[str]],
    layout_type: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        vt = str(r.get("visual_type", ""))
        sp = str(r.get("sentiment_polarity", ""))
        lt = str(r.get("layout_type", ""))
        if (
            match_optional(vt, visual_type)
            and match_optional(sp, sentiment_polarity)
            and match_optional(lt, layout_type)
        ):
            out.append(r)
    return out


def aggregate_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates by visual_type/sentiment_polarity/layout_type.
    """
    agg: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        k = key3(r)
        agg[k] = agg.get(k, 0) + 1
    # convert keys to json-friendly
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


def group_rows_by_key3(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    g: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        g.setdefault(key3(r), []).append(r)
    return g


def allocate_quota(total: int, n_buckets: int, rng: random.Random) -> List[int]:
    """
    Evenly allocate 'total' items into 'n_buckets' buckets.
    Any remainder is randomly distributed to buckets.
    """
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


def sample_uniform_over_groups(
    rows: Sequence[Dict[str, Any]], sample_size: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """
    Uniform sampling over key3 groups (visual_type, sentiment_polarity, layout_type).
    - Allocate sample_size evenly across groups.
    - Within each group, sample without replacement if enough; else sample with replacement.
    """
    if sample_size <= 0:
        return []
    groups = group_rows_by_key3(rows)
    if not groups:
        return []

    keys = list(groups.keys())
    quotas = allocate_quota(sample_size, len(keys), rng)
    out: List[Dict[str, Any]] = []

    for k, q in zip(keys, quotas):
        if q <= 0:
            continue
        bucket = groups[k]
        if len(bucket) >= q:
            out.extend(rng.sample(bucket, k=q))
        else:
            out.extend([rng.choice(bucket) for _ in range(q)])

    rng.shuffle(out)
    return out


# =============================
# Path + missing checks
# =============================


def build_src_path(src_dir: str, row: Dict[str, Any]) -> str:
    # index field: id
    return os.path.join(src_dir, str(row["id"]))


def build_gen_path(gen_root: str, model: str, row: Dict[str, Any]) -> str:
    # index field: gen_id
    return os.path.join(gen_root, model, str(row["gen_id"]))


def verify_missing(
    rows: Sequence[Dict[str, Any]],
    src_dir: str,
    gen_root: str,
    models: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (kept_rows, report)
    - A row is kept only if:
        - SOURCE exists
        - for each requested model, GENERATED exists
    """
    kept: List[Dict[str, Any]] = []
    missing_report: Dict[str, Any] = {
        "missing_source": 0,
        "missing_by_model": {m: 0 for m in models},
        "missing_examples": {m: [] for m in models},
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
    missing_report["total"] = len(rows)
    return kept, missing_report


# =============================
# Metric extraction / scoring
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
    """
    Normalize to numeric scores where applicable.

    Assumes judge schema:
      - visual_assessment.generation_quality_1_5
      - visual_assessment.emotion_accuracy_1_5
      - text_assessment.generation_quality_1_5
      - text_assessment.emotion_accuracy_1_5
      - layout_consistency.consistent
      - overall_emotion_classification.label (+ other_emotion)
      - perceived_emotion_shift.magnitude_1_5
      - overall_generation_quality.overall_generation_quality_1_5
    """
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
        if isinstance(x, (int, float)):
            if 1 <= float(x) <= 5:
                return float(x)
        return None

    out["visual_generation_quality_1_5"] = to_1_5(vq) or 0.0
    out["visual_emotion_accuracy_1_5"] = to_1_5(va) or 0.0
    out["text_generation_quality_1_5"] = to_1_5(tq) or 0.0
    out["text_emotion_accuracy_1_5"] = to_1_5(ta) or 0.0
    out["perceived_emotion_shift_1_5"] = to_1_5(shift) or 0.0
    out["overall_generation_quality_1_5"] = to_1_5(oq) or 0.0

    if isinstance(lc, bool):
        out["layout_consistency_accuracy"] = 1.0 if lc else 0.0
    else:
        out["layout_consistency_accuracy"] = 0.0

    # classification hit wrt tgt_emotion (target-gated)
    out["tgt_emotion_hit"] = (
        1.0 if (isinstance(cls_label, str) and cls_label == tgt_emotion) else 0.0
    )

    return out


# =============================
# Cost / cache helpers
# =============================


def cache_flag(payload: Dict[str, Any]) -> bool:
    """
    Your SQLiteStore marks:
      payload["_cache_key"], payload["_cached_is_error"]
    We treat it as cache-hit if those keys exist AND there's no "error" in payload.
    """
    if "error" in payload:
        return False
    return ("_cache_key" in payload) and ("_cached_is_error" in payload)


def extract_cost_usd_from_cost_info(cost_info: Optional[Dict[str, Any]]) -> float:
    if not isinstance(cost_info, dict):
        return 0.0
    v = cost_info.get("estimated_cost_usd")
    return float(v) if isinstance(v, (int, float)) else 0.0


# =============================
# Evaluation runner
# =============================


def make_sample(
    src_dir: str, gen_root: str, model: str, row: Dict[str, Any]
) -> MemeSample:
    return MemeSample(
        sample_id=str(row.get("id")),
        src_image_path=build_src_path(src_dir, row),
        gen_image_path=build_gen_path(gen_root, model, row),
        src_emotion=str(row.get("det_emo", "")),
        tgt_emotion=str(row.get("tgt_emo", "")),
        src_caption_text=row.get("text"),
        tgt_caption_text=row.get("pos_txt"),
        edit_instruction=row.get("spec"),
    )


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def ci95_of_replicates(xs: Sequence[float]) -> Tuple[float, float]:
    """
    Simple percentile CI across replicate means.
    """
    if not xs:
        return 0.0, 0.0
    ys = sorted(xs)
    lo = ys[int(0.025 * (len(ys) - 1))]
    hi = ys[int(0.975 * (len(ys) - 1))]
    return lo, hi


def _init_rep_struct(models: Sequence[str]) -> Dict[str, Dict[str, List[float]]]:
    return {m: {k: [] for k in METRIC_KEYS} for m in models}


def _init_book_struct(models: Sequence[str]) -> Dict[str, Any]:
    return {
        "per_model": {
            m: {"api_calls": 0, "cache_hits": 0, "cost_usd": 0.0, "errors": 0}
            for m in models
        }
    }


def _update_bookkeeping(
    book: Dict[str, Any],
    m: str,
    payload: Dict[str, Any],
    cost_info: Optional[Dict[str, Any]],
    compute_cost: bool,
) -> None:
    if "error" in payload:
        book["per_model"][m]["errors"] += 1
        return

    if cache_flag(payload):
        book["per_model"][m]["cache_hits"] += 1
    else:
        book["per_model"][m]["api_calls"] += 1
        if compute_cost:
            book["per_model"][m]["cost_usd"] += extract_cost_usd_from_cost_info(
                cost_info
            )


def _summarize_rep_means(
    rep_means: Dict[str, Dict[str, List[float]]], runs: int
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"models": {}}
    for m, mmetrics in rep_means.items():
        mm: Dict[str, Any] = {}
        for k, xs in mmetrics.items():
            mm[k] = {
                "mean_over_runs": mean(xs),
                "std_over_runs": std(xs),
                "ci95_over_runs": list(ci95_of_replicates(xs)),
                "runs": runs,
            }
        summary["models"][m] = mm
    return summary


def evaluate_models(
    *,
    rows: Sequence[Dict[str, Any]],
    models: Sequence[str],
    src_dir: str,
    gen_root: str,
    client: JudgeClient,
    cfg: JudgeConfig,
    runs: int,
    sample_size: int,
    seed: int,
    compute_cost: bool,
    use_cache: bool,
    force_refresh: bool,
    mode_model_view: bool,
    mode_factor_view: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")
    if runs <= 0:
        raise ValueError("--runs must be > 0")

    out: Dict[str, Any] = {}

    # -------------------------
    # Mode 1: model_view
    # -------------------------
    if mode_model_view:
        model_view_rep_means = _init_rep_struct(models)
        model_view_book = _init_book_struct(models)

        for r_i in range(runs):
            # uniform over key3 combinations (visual_type, sentiment_polarity, layout_type)
            batch = sample_uniform_over_groups(rows, sample_size=sample_size, rng=rng)

            rep_vals = _init_rep_struct(models)

            for row in batch:
                tgt = str(row.get("tgt_emo", ""))

                for m in models:
                    sample = make_sample(src_dir, gen_root, m, row)
                    payload, cost_info = client.judge(
                        sample,
                        cfg,
                        use_cache=use_cache,
                        force_refresh=force_refresh,
                    )

                    _update_bookkeeping(
                        model_view_book, m, payload, cost_info, compute_cost
                    )
                    if "error" in payload:
                        continue

                    metrics = extract_metrics(payload, tgt)
                    for k, v in metrics.items():
                        rep_vals[m][k].append(float(v))

            for m in models:
                for k in METRIC_KEYS:
                    model_view_rep_means[m][k].append(mean(rep_vals[m][k]))

            logger.info("[model_view] Run {}/{} done.", r_i + 1, runs)

        out["model_view"] = _summarize_rep_means(model_view_rep_means, runs=runs)
        out["model_view"]["bookkeeping"] = model_view_book

    # -------------------------
    # Mode 2: factor_view
    # -------------------------
    if mode_factor_view:

        def factor_groups(
            rows_: Sequence[Dict[str, Any]], factor: str
        ) -> Dict[str, List[Dict[str, Any]]]:
            g: Dict[str, List[Dict[str, Any]]] = {}
            for rr in rows_:
                v = str(rr.get(factor, ""))
                g.setdefault(v, []).append(rr)
            return g

        vt_groups = factor_groups(rows, "visual_type")
        sp_groups = factor_groups(rows, "sentiment_polarity")
        lt_groups = factor_groups(rows, "layout_type")

        factor_view: Dict[str, Any] = {
            "visual_type": {},
            "sentiment_polarity": {},
            "layout_type": {},
        }

        def eval_one_factor_set(
            groups: Dict[str, List[Dict[str, Any]]], factor_name: str
        ) -> Dict[str, Any]:
            result: Dict[str, Any] = {}

            for fv, grows in groups.items():
                rep_means = _init_rep_struct(models)
                book = _init_book_struct(models)

                for r_i in range(runs):
                    # per-run sample_size budget split evenly across models
                    mquotas = allocate_quota(sample_size, len(models), rng)
                    mquota_map = {m: q for m, q in zip(models, mquotas)}

                    rep_vals = _init_rep_struct(models)

                    for m in models:
                        q = mquota_map[m]
                        if q <= 0:
                            continue

                        if len(grows) >= q:
                            gbatch = rng.sample(grows, k=q)
                        else:
                            gbatch = [rng.choice(grows) for _ in range(q)]

                        for row in gbatch:
                            tgt = str(row.get("tgt_emo", ""))

                            sample = make_sample(src_dir, gen_root, m, row)
                            payload, cost_info = client.judge(
                                sample,
                                cfg,
                                use_cache=use_cache,
                                force_refresh=force_refresh,
                            )

                            _update_bookkeeping(
                                book, m, payload, cost_info, compute_cost
                            )
                            if "error" in payload:
                                continue

                            metrics = extract_metrics(payload, tgt)
                            for kk, vv in metrics.items():
                                rep_vals[m][kk].append(float(vv))

                    for m in models:
                        for kk in METRIC_KEYS:
                            rep_means[m][kk].append(mean(rep_vals[m][kk]))

                summarized = _summarize_rep_means(rep_means, runs=runs)
                summarized["bookkeeping"] = book

                # macro-average across models for quick comparison
                macro: Dict[str, Any] = {}
                for kk in METRIC_KEYS:
                    model_means = [
                        summarized["models"][m][kk]["mean_over_runs"] for m in models
                    ]
                    macro[kk] = {
                        "mean_over_models": mean(model_means),
                        "models": len(models),
                    }
                summarized["macro_over_models"] = macro

                result[fv] = summarized
                logger.info("[factor_view:{}={}] done.", factor_name, fv)

            return result

        factor_view["visual_type"] = eval_one_factor_set(vt_groups, "visual_type")
        factor_view["sentiment_polarity"] = eval_one_factor_set(
            sp_groups, "sentiment_polarity"
        )
        factor_view["layout_type"] = eval_one_factor_set(lt_groups, "layout_type")

        out["factor_view"] = factor_view

    return out


# =============================
# CLI
# =============================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate meme emotion reframing outputs (per model)."
    )

    p.add_argument(
        "--index", required=True, help="Path to index file (.json or .jsonl)."
    )
    p.add_argument(
        "--src-dir", required=True, help="Directory of original/source images."
    )
    p.add_argument(
        "--gen-root",
        required=True,
        help="Root directory of generated images. Subdirs are model names.",
    )

    p.add_argument(
        "--models",
        required=True,
        nargs="+",
        help="One or more model subdir names under --gen-root.",
    )

    p.add_argument(
        "--visual-type",
        nargs="*",
        default=None,
        help="Filter: allowed visual_type values.",
    )
    p.add_argument(
        "--sentiment-polarity",
        nargs="*",
        default=None,
        help="Filter: allowed sentiment_polarity values.",
    )
    p.add_argument(
        "--layout-type",
        nargs="*",
        default=None,
        help="Filter: allowed layout_type values.",
    )

    p.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repeated subsampling runs (default 10).",
    )
    p.add_argument(
        "--sample-size", type=int, default=50, help="Sample size per run (default 50)."
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")

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
        "--model", default="gpt-5.2", help="Judge model name (default gpt-5.2)."
    )
    p.add_argument("--max-output-tokens", type=int, default=900)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument(
        "--rationale-first",
        action="store_true",
        help="Rationales first in schema (global).",
    )
    p.add_argument(
        "--rationale-last",
        action="store_true",
        help="Rationales last in schema (global).",
    )
    p.add_argument(
        "--images-first",
        action="store_true",
        help="Images before instruction text (ablation).",
    )
    p.add_argument(
        "--images-last",
        action="store_true",
        help="Images after instruction text (ablation).",
    )

    p.add_argument(
        "--compute-cost",
        action="store_true",
        help="Include cost summary (exclude cache hits).",
    )

    # === MOD: modes (can enable both) ===
    p.add_argument(
        "--mode-model-view",
        action="store_true",
        help="Mode 1: per-model metrics with uniform sampling over (visual_type,sentiment_polarity,layout_type) combinations.",
    )
    p.add_argument(
        "--mode-factor-view",
        action="store_true",
        help="Mode 2: per-factor (visual_type / sentiment_polarity / layout_type) metrics; within each factor value, sample budget is evenly split across models.",
    )

    p.add_argument(
        "--out", default="", help="Optional output JSON path; empty => stdout."
    )
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    # config switches
    rationale_first = True
    if args.rationale_first and args.rationale_last:
        logger.error("Choose only one: --rationale-first or --rationale-last")
        return 2
    if args.rationale_last:
        rationale_first = False

    images_first = True
    if args.images_first and args.images_last:
        logger.error("Choose only one: --images-first or --images-last")
        return 2
    if args.images_last:
        images_first = False

    # === MOD: mode defaults ===
    mode_model_view = bool(args.mode_model_view)
    mode_factor_view = bool(args.mode_factor_view)
    if not mode_model_view and not mode_factor_view:
        mode_model_view = True  # default

    # load index
    rows_all = load_index(args.index)
    logger.info("Loaded index rows: {}", len(rows_all))

    # aggregate overview BEFORE missing filter
    overview_before = aggregate_counts(rows_all)

    # filter by fields (before missing check)
    rows_filtered = filter_rows(
        rows_all,
        visual_type=args.visual_type,
        sentiment_polarity=args.sentiment_polarity,
        layout_type=args.layout_type,
    )
    logger.info("After field filters: {}", len(rows_filtered))

    # verify missing for requested models
    rows_kept, missing_report = verify_missing(
        rows_filtered, args.src_dir, args.gen_root, args.models
    )
    logger.info("After missing-check keep: {}", len(rows_kept))
    if missing_report["missing_source"] > 0 or any(
        missing_report["missing_by_model"].values()
    ):
        logger.warning("Missing report: {}", missing_report)

    # aggregate overview AFTER missing filter
    overview_after = aggregate_counts(rows_kept)

    # build judge client
    client = JudgeClient(
        model=args.model,
        retry=RetryConfig(timeout_s=120),
        max_in_flight=8,
        cache=CacheConfig(
            enabled=bool(args.use_cache),
            db_path=args.db,
            return_cached_errors=False,
        ),
    )

    cfg = JudgeConfig(
        rationale_first=rationale_first,
        images_first=images_first,
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
    )

    # evaluate
    summary = evaluate_models(
        rows=rows_kept,
        models=args.models,
        src_dir=args.src_dir,
        gen_root=args.gen_root,
        client=client,
        cfg=cfg,
        runs=int(args.runs),
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        compute_cost=bool(args.compute_cost),
        use_cache=bool(args.use_cache),
        force_refresh=bool(args.force_refresh),
        mode_model_view=mode_model_view,
        mode_factor_view=mode_factor_view,
    )

    out_obj = {
        "config": {
            "index": args.index,
            "src_dir": args.src_dir,
            "gen_root": args.gen_root,
            "models": list(args.models),
            "filters": {
                "visual_type": args.visual_type,
                "sentiment_polarity": args.sentiment_polarity,
                "layout_type": args.layout_type,
            },
            "sampling": {
                "runs": args.runs,
                "sample_size": args.sample_size,
                "seed": args.seed,
            },
            "judge": {
                "judge_model": args.model,
                "rationale_first": rationale_first,
                "images_first": images_first,
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
            },
            "cache": {
                "enabled": bool(args.use_cache),
                "db": args.db,
                "force_refresh": bool(args.force_refresh),
            },
            "compute_cost": bool(args.compute_cost),
            "modes": {
                "mode_model_view": mode_model_view,
                "mode_factor_view": mode_factor_view,
            },
        },
        "overview_before_filters": overview_before,
        "missing_report": missing_report,
        "overview_after_missing_filter": overview_after,
        "results": summary,
        "notes": {
            "cost_behavior": "cost is summed ONLY for non-cache calls, using cost_info['estimated_cost_usd'] returned by JudgeClient.judge().",
            "statistics": "Each metric reports mean/std/CI95 over repeated subsampling runs (replicate means).",
            "modes": {
                "model_view": "Per-model aggregation; per-run sample is uniformly allocated over (visual_type,sentiment_polarity,layout_type) combinations.",
                "factor_view": "Per-factor aggregation (visual_type/sentiment_polarity/layout_type separately); per-run sample budget is evenly split across models within each factor value.",
            },
        },
    }

    js = json.dumps(out_obj, ensure_ascii=False, indent=2)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        logger.info("Wrote output to {}", args.out)
    else:
        print(js)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
