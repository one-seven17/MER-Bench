from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

# ---- import your judge module ----
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
# Basic helpers
# =============================


def safe_get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


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


# =============================
# Filtering / validation
# =============================


FACTORS = ("visual_type", "sentiment_polarity", "layout_type")


def unique_values(rows: Sequence[Dict[str, Any]], field: str) -> List[str]:
    s = {str(r.get(field, "")) for r in rows if str(r.get(field, ""))}
    return sorted(s)


def validate_closed_set(
    field: str, requested: Optional[Sequence[str]], allowed: Sequence[str]
) -> None:
    if not requested:
        return
    allowed_set = set(allowed)
    bad = [v for v in requested if v not in allowed_set]
    if bad:
        raise ValueError(
            f"Invalid {field} values: {bad}. Allowed (closed set from data): {allowed}"
        )


def row_matches_field(
    row: Dict[str, Any], field: str, allowed: Optional[Sequence[str]]
) -> bool:
    if not allowed:
        return True
    return str(row.get(field, "")) in set(allowed)


def filter_rows_by_all(
    rows: Sequence[Dict[str, Any]],
    visual_type: Optional[Sequence[str]],
    sentiment_polarity: Optional[Sequence[str]],
    layout_type: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if (
            row_matches_field(r, "visual_type", visual_type)
            and row_matches_field(r, "sentiment_polarity", sentiment_polarity)
            and row_matches_field(r, "layout_type", layout_type)
        ):
            out.append(r)
    return out


def aggregate_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates by each factor independently (not coupled).
    """
    out: Dict[str, Any] = {}
    for f in FACTORS:
        cnt: Dict[str, int] = {}
        for r in rows:
            v = str(r.get(f, ""))
            cnt[v] = cnt.get(v, 0) + 1
        out[f"by_{f}"] = [
            {"value": k, "count": v}
            for k, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
        ]
    return out


# =============================
# Cartesian "data category cells"
# =============================


def active_factor_spec(
    visual_type: Optional[Sequence[str]],
    sentiment_polarity: Optional[Sequence[str]],
    layout_type: Optional[Sequence[str]],
) -> List[Tuple[str, List[str]]]:
    """
    Return list of (factor_name, factor_values) for factors explicitly specified by user.
    """
    spec: List[Tuple[str, List[str]]] = []
    if visual_type:
        spec.append(("visual_type", list(visual_type)))
    if sentiment_polarity:
        spec.append(("sentiment_polarity", list(sentiment_polarity)))
    if layout_type:
        spec.append(("layout_type", list(layout_type)))
    return spec


def build_cells(spec: List[Tuple[str, List[str]]]) -> List[Dict[str, str]]:
    """
    Cartesian product of specified factor values.
    Each cell is a dict: {factor: value, ...}
    If spec is empty -> single empty cell meaning "no cell constraints".
    """
    if not spec:
        return [{}]

    names = [k for k, _ in spec]
    value_lists = [vs for _, vs in spec]
    cells: List[Dict[str, str]] = []
    for combo in itertools.product(*value_lists):
        cells.append({names[i]: combo[i] for i in range(len(names))})
    return cells


def cell_key(cell: Dict[str, str]) -> str:
    if not cell:
        return "ALL"
    parts = [f"{k}={cell[k]}" for k in sorted(cell.keys())]
    return "|".join(parts)


def filter_rows_by_cell(
    rows: Sequence[Dict[str, Any]], cell: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Only constrain fields present in cell; other factors remain free.
    """
    if not cell:
        return list(rows)
    out: List[Dict[str, Any]] = []
    for r in rows:
        ok = True
        for k, v in cell.items():
            if str(r.get(k, "")) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def sample_uniform_over_cells(
    rows: Sequence[Dict[str, Any]],
    cells: Sequence[Dict[str, str]],
    sample_size: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Uniform sampling over *cells* (Cartesian product of specified factors).
    - Allocate sample_size evenly across NON-EMPTY cells (cells with at least 1 row).
    - Within each cell, sample without replacement if enough; else with replacement.
    """
    if sample_size <= 0:
        return []
    if not cells:
        return []

    cell_rows: List[Tuple[Dict[str, str], List[Dict[str, Any]]]] = []
    for c in cells:
        cr = filter_rows_by_cell(rows, c)
        if cr:
            cell_rows.append((c, cr))

    if not cell_rows:
        return []

    quotas = allocate_quota(sample_size, len(cell_rows), rng)
    out: List[Dict[str, Any]] = []

    for (c, bucket), q in zip(cell_rows, quotas):
        if q <= 0:
            continue
        if len(bucket) >= q:
            out.extend(rng.sample(bucket, k=q))
        else:
            out.extend([rng.choice(bucket) for _ in range(q)])

    rng.shuffle(out)
    return out


# =============================
# Paths + missing checks
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
# Metric extraction
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

    def to_1_5(x: Any) -> float:
        if isinstance(x, (int, float)) and 1 <= float(x) <= 5:
            return float(x)
        return 0.0

    out["visual_generation_quality_1_5"] = to_1_5(vq)
    out["visual_emotion_accuracy_1_5"] = to_1_5(va)
    out["text_generation_quality_1_5"] = to_1_5(tq)
    out["text_emotion_accuracy_1_5"] = to_1_5(ta)
    out["perceived_emotion_shift_1_5"] = to_1_5(shift)
    out["overall_generation_quality_1_5"] = to_1_5(oq)

    out["layout_consistency_accuracy"] = 1.0 if isinstance(lc, bool) and lc else 0.0
    out["tgt_emotion_hit"] = (
        1.0 if (isinstance(cls_label, str) and cls_label == tgt_emotion) else 0.0
    )

    return out


# =============================
# Cost / cache helpers
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
    # NEW:
    cells: Sequence[Dict[str, str]],
    mode_model_view: bool,
    mode_category_view: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")
    if runs <= 0:
        raise ValueError("--runs must be > 0")

    out: Dict[str, Any] = {}

    # -------------------------
    # Mode 1: model_view
    # Per model, uniform sampling over CELLS
    # -------------------------
    if mode_model_view:
        rep_means = _init_rep_struct(models)
        book = _init_book_struct(models)

        for r_i in range(runs):
            batch = sample_uniform_over_cells(
                rows, cells=cells, sample_size=sample_size, rng=rng
            )
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

                    _update_bookkeeping(book, m, payload, cost_info, compute_cost)
                    if "error" in payload:
                        continue

                    metrics = extract_metrics(payload, tgt)
                    for k, v in metrics.items():
                        rep_vals[m][k].append(float(v))

            for m in models:
                for k in METRIC_KEYS:
                    rep_means[m][k].append(mean(rep_vals[m][k]))

            logger.info("[model_view] Run {}/{} done.", r_i + 1, runs)

        out["model_view"] = _summarize_rep_means(rep_means, runs=runs)
        out["model_view"]["bookkeeping"] = book

    # -------------------------
    # Mode 2: category_view
    # For each cell, results aggregated across models.
    # Requirement:
    #   - output has |cells| keys (e.g., 4 keys)
    #   - each val is aggregated over specified models
    #   - within each cell per run, sample_size budget is evenly split across models
    # -------------------------
    if mode_category_view:
        cat_out: Dict[str, Any] = {}

        for cell in cells:
            ck = cell_key(cell)
            cell_rows = filter_rows_by_cell(rows, cell)
            if not cell_rows:
                cat_out[ck] = {
                    "warning": "empty_cell_after_filters",
                    "cell": cell,
                }
                continue

            rep_means = _init_rep_struct(models)
            book = _init_book_struct(models)

            for r_i in range(runs):
                # split per-run budget across models
                mquotas = allocate_quota(sample_size, len(models), rng)
                mquota_map = {m: q for m, q in zip(models, mquotas)}
                rep_vals = _init_rep_struct(models)

                for m in models:
                    q = mquota_map[m]
                    if q <= 0:
                        continue

                    if len(cell_rows) >= q:
                        gbatch = rng.sample(cell_rows, k=q)
                    else:
                        gbatch = [rng.choice(cell_rows) for _ in range(q)]

                    for row in gbatch:
                        tgt = str(row.get("tgt_emo", ""))

                        sample = make_sample(src_dir, gen_root, m, row)
                        payload, cost_info = client.judge(
                            sample,
                            cfg,
                            use_cache=use_cache,
                            force_refresh=force_refresh,
                        )

                        _update_bookkeeping(book, m, payload, cost_info, compute_cost)
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

            # aggregate across models (macro)
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
            summarized["cell"] = cell
            summarized["cell_size"] = len(cell_rows)

            cat_out[ck] = summarized
            logger.info("[category_view] cell {} done.", ck)

        out["category_view"] = cat_out

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

    # Modes (can enable both)
    p.add_argument(
        "--mode-model-view",
        action="store_true",
        help="Per-model metrics with uniform sampling over Cartesian cells of specified factors.",
    )
    p.add_argument(
        "--mode-category-view",
        action="store_true",
        help="Per-cell metrics aggregated across models; sample budget evenly split across models per cell.",
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

    # modes
    mode_model_view = bool(args.mode_model_view)
    mode_category_view = bool(args.mode_category_view)
    if not mode_model_view and not mode_category_view:
        mode_model_view = True  # default

    # load index
    rows_all = load_index(args.index)
    logger.info("Loaded index rows: {}", len(rows_all))

    # ====== NEW: closed-set validation (from the data itself) ======
    allowed_vt = unique_values(rows_all, "visual_type")
    allowed_sp = unique_values(rows_all, "sentiment_polarity")
    allowed_lt = unique_values(rows_all, "layout_type")

    try:
        validate_closed_set("visual_type", args.visual_type, allowed_vt)
        validate_closed_set("sentiment_polarity", args.sentiment_polarity, allowed_sp)
        validate_closed_set("layout_type", args.layout_type, allowed_lt)
    except Exception as e:
        logger.error("Filter validation failed: {}", e)
        return 2

    # aggregate overview BEFORE missing filter
    overview_before = aggregate_counts(rows_all)

    # filter by fields (still an AND filter, but NOTE: cells are built later only from specified factors)
    rows_filtered = filter_rows_by_all(
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

    # ====== NEW: build CARTESIAN CELLS only from specified factors (decoupled) ======
    spec = active_factor_spec(
        args.visual_type, args.sentiment_polarity, args.layout_type
    )
    cells = build_cells(spec)
    logger.info("Active factors: {}", [k for k, _ in spec] if spec else ["(none)"])
    logger.info("Cartesian cell count: {}", len(cells))

    # If user specified factors but some cells are empty after missing filter, warn early
    empty_cells = [cell_key(c) for c in cells if not filter_rows_by_cell(rows_kept, c)]
    if empty_cells:
        logger.warning(
            "Empty cells after filters+missing-check (will be skipped in uniform sampling): {}",
            empty_cells[:50],
        )

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
        cells=cells,
        mode_model_view=mode_model_view,
        mode_category_view=mode_category_view,
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
                "cells": [cell_key(c) for c in cells],
                "active_factors_for_cells": [k for k, _ in spec] if spec else [],
                "note": "Cells are built from the Cartesian product of explicitly specified factors only.",
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
                "mode_category_view": mode_category_view,
            },
        },
        "closed_sets": {
            "visual_type": allowed_vt,
            "sentiment_polarity": allowed_sp,
            "layout_type": allowed_lt,
        },
        "overview_before_filters": overview_before,
        "missing_report": missing_report,
        "overview_after_missing_filter": overview_after,
        "results": summary,
        "notes": {
            "cost_behavior": "Cost is summed ONLY for non-cache calls, using cost_info['estimated_cost_usd'] returned by JudgeClient.judge().",
            "modes": {
                "model_view": "Per-model aggregation; per-run sample is uniformly allocated over Cartesian cells of specified factors.",
                "category_view": "Per-cell aggregation; within each cell and run, sample budget is evenly split across models; output has |cells| keys.",
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
