import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Helpers
# -------------------------

EPS = 1e-12


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def hm(a: float, b: float, eps: float = EPS) -> float:
    denom = a + b
    if denom <= eps:
        return 0.0
    return 2.0 * a * b / (denom + eps)


def geo_mean(vals: List[float], eps: float = EPS) -> float:
    if not vals:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= max(v, eps)
    return prod ** (1.0 / len(vals))


def norm_1_5_to_0_1(x: float) -> float:
    # map 1..5 -> 0..1
    return clamp((x - 1.0) / 4.0, 0.0, 1.0)


def percentile(sorted_xs: List[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    if p <= 0:
        return sorted_xs[0]
    if p >= 1:
        return sorted_xs[-1]
    n = len(sorted_xs)
    idx = p * (n - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    w = idx - lo
    return sorted_xs[lo] * (1 - w) + sorted_xs[hi] * w


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std_unbiased(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def bootstrap_ci(
    xs: List[float],
    *,
    rng: random.Random,
    n_boot: int = 5000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap CI on the mean.
    No normality assumption. Works on small n but CI may be wide.
    """
    if not xs:
        return (0.0, 0.0)
    n = len(xs)
    boot_means: List[float] = []
    for _ in range(n_boot):
        s = [xs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(mean(s))
    boot_means.sort()
    alpha = (1.0 - ci) / 2.0
    lo = percentile(boot_means, alpha)
    hi = percentile(boot_means, 1.0 - alpha)
    return (lo, hi)


# -------------------------
# Core: compute TA/CF/SS/RFS from metric values in [0,1]
# -------------------------


def compute_components_from_unit_metrics(
    *,
    Ev: float,  # visual emotion acc in [0,1]
    Et: float,  # text emotion acc in [0,1]
    H: float,  # hit rate in [0,1] (gate)
    L: float,  # layout consistency in [0,1] (gate)
    Qv: float,  # visual quality in [0,1]
    Qt: float,  # text quality in [0,1]
    Qo: float,  # overall quality in [0,1]
    S: float,  # shift in [0,1]
    alpha: float = 3.0,
) -> Dict[str, float]:
    # 3-block design:
    TA = H * hm(Ev, Et)
    CF = L * geo_mean([Qv, Qt, Qo])
    SS = 1.0 - math.exp(-alpha * clamp(S, 0.0, 1.0))
    RFS = TA * CF * SS
    return {"TA": TA, "CF": CF, "SS": SS, "RFS": RFS}


# -------------------------
# Extractors for your output JSON
# -------------------------


def _get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def extract_metric_replicates(node: Dict[str, Any], metric_key: str) -> List[float]:
    """
    Preferred: read per-run replicate means saved by main.py:
      node[metric_key]["replicate_means"] = [run0_mean, run1_mean, ...]
    Fallback: if only mean/std exist, synthesize a degenerate replicate list of length 1.

    Returns list of floats (length = runs) if available.
    """
    m = _get(node, metric_key)
    if not isinstance(m, dict):
        return []

    reps = m.get("replicate_means")
    if isinstance(reps, list) and reps:
        out: List[float] = []
        for x in reps:
            if isinstance(x, (int, float)):
                out.append(float(x))
        return out

    # fallback (legacy files)
    mu = m.get("mean_over_runs", 0.0)
    if isinstance(mu, (int, float)):
        return [float(mu)]
    return []


def _unitize_scalar(metric_name: str, x: float) -> float:
    """
    Convert a single scalar metric (either in 1..5 or 0..1) into unit space [0,1].
    """
    if metric_name in {
        "visual_generation_quality_1_5",
        "visual_emotion_accuracy_1_5",
        "text_generation_quality_1_5",
        "text_emotion_accuracy_1_5",
        "perceived_emotion_shift_1_5",
        "overall_generation_quality_1_5",
    }:
        return norm_1_5_to_0_1(float(x))

    if metric_name in {"layout_consistency_accuracy", "tgt_emotion_hit"}:
        return clamp(float(x), 0.0, 1.0)

    return clamp(float(x), 0.0, 1.0)


# -------------------------
# Main aggregator (run-wise, then bootstrap CI)
# -------------------------


def rfs_from_metrics_node_bootstrap(
    metrics_node: Dict[str, Any],
    *,
    alpha: float = 3.0,
    seed: int = 0,
    n_boot: int = 5000,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """
    metrics_node is either:
      - model_view["models"][<model>]  (contains metric dicts)
      - factor_view["metrics"]         (contains metric dicts)

    Requires main.py to save:
      metric["replicate_means"] = list of per-run means

    Computes per-run components -> returns mean/std + bootstrap CI (non-parametric).
    """
    rng = random.Random(seed)

    need = {
        "Ev": "visual_emotion_accuracy_1_5",
        "Et": "text_emotion_accuracy_1_5",
        "H": "tgt_emotion_hit",
        "L": "layout_consistency_accuracy",
        "Qv": "visual_generation_quality_1_5",
        "Qt": "text_generation_quality_1_5",
        "Qo": "overall_generation_quality_1_5",
        "S": "perceived_emotion_shift_1_5",
    }

    # extract replicate arrays
    rep: Dict[str, List[float]] = {}
    for sym, k in need.items():
        rep[sym] = extract_metric_replicates(metrics_node, k)

    # align runs length: use the minimum across available metrics (strict, avoids misalignment)
    lengths = [len(v) for v in rep.values() if v]
    if not lengths:
        return {
            "alpha": alpha,
            "seed": seed,
            "n_boot": n_boot,
            "ci": ci,
            "warning": "No replicate_means found. Ensure main.py writes replicate_means per metric.",
        }

    R = min(lengths)
    if R <= 0:
        return {
            "alpha": alpha,
            "seed": seed,
            "n_boot": n_boot,
            "ci": ci,
            "warning": "Replicate length is zero after alignment.",
        }

    # compute per-run components
    runs_TA: List[float] = []
    runs_CF: List[float] = []
    runs_SS: List[float] = []
    runs_RFS: List[float] = []

    for i in range(R):
        Ev = _unitize_scalar(need["Ev"], rep["Ev"][i])
        Et = _unitize_scalar(need["Et"], rep["Et"][i])
        H = _unitize_scalar(need["H"], rep["H"][i])
        L = _unitize_scalar(need["L"], rep["L"][i])
        Qv = _unitize_scalar(need["Qv"], rep["Qv"][i])
        Qt = _unitize_scalar(need["Qt"], rep["Qt"][i])
        Qo = _unitize_scalar(need["Qo"], rep["Qo"][i])
        S = _unitize_scalar(need["S"], rep["S"][i])

        comp = compute_components_from_unit_metrics(
            Ev=Ev, Et=Et, H=H, L=L, Qv=Qv, Qt=Qt, Qo=Qo, S=S, alpha=alpha
        )
        runs_TA.append(comp["TA"])
        runs_CF.append(comp["CF"])
        runs_SS.append(comp["SS"])
        runs_RFS.append(comp["RFS"])

    def pack(xs: List[float]) -> Dict[str, Any]:
        xs_sorted = sorted(xs)
        mu = mean(xs)
        sd = std_unbiased(xs)
        ci_lo, ci_hi = bootstrap_ci(xs, rng=rng, n_boot=n_boot, ci=ci)
        return {
            "mean": mu,
            "std": sd,
            "ci95": [ci_lo, ci_hi] if abs(ci - 0.95) < 1e-9 else [ci_lo, ci_hi],
            "runs": len(xs_sorted),
        }

    return {
        "alpha": alpha,
        "seed": seed,
        "bootstrap_samples": n_boot,
        "ci": ci,
        "runs_used": R,
        "TA": pack(runs_TA),
        "CF": pack(runs_CF),
        "SS": pack(runs_SS),
        "RFS": pack(runs_RFS),
        # optional: you can keep these for debugging/plots (may be large)
        # "per_run": {"TA": runs_TA, "CF": runs_CF, "SS": runs_SS, "RFS": runs_RFS},
    }


def compute_rfs_from_output_json(
    json_path: str,
    *,
    alpha: float = 3.0,
    seed: int = 0,
    n_boot: int = 5000,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """
    Input: your final output JSON file from main.py
    Output: RFS aggregates for model_view and/or factor_view (depending on what exists in file).

    Expects each metric dict to include "replicate_means" (per run).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    results = obj.get("results", {})
    out: Dict[str, Any] = {
        "input_file": json_path,
        "alpha": alpha,
        "seed": seed,
        "bootstrap_samples": n_boot,
        "ci": ci,
        "computed": {},
    }

    mv = results.get("model_view")
    if isinstance(mv, dict):
        mv_models = _get(mv, "models")
        if isinstance(mv_models, dict):
            out_mv: Dict[str, Any] = {}
            for model_name, node in mv_models.items():
                if not isinstance(node, dict):
                    continue
                out_mv[model_name] = rfs_from_metrics_node_bootstrap(
                    node, alpha=alpha, seed=seed, n_boot=n_boot, ci=ci
                )
            out["computed"]["model_view"] = out_mv

    fv = results.get("factor_view")
    if isinstance(fv, dict):
        metrics_node = fv.get("metrics")
        if isinstance(metrics_node, dict):
            out["computed"]["factor_view"] = rfs_from_metrics_node_bootstrap(
                metrics_node, alpha=alpha, seed=seed, n_boot=n_boot, ci=ci
            )

    if not out["computed"]:
        out["warning"] = "No model_view or factor_view found in JSON['results']."

    return out


# -------------------------
# Example CLI usage
# -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to main.py output JSON")
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--out", default="", help="Optional output JSON path")
    args = p.parse_args()

    agg = compute_rfs_from_output_json(
        args.input, alpha=args.alpha, seed=args.seed, n_boot=args.n_boot, ci=args.ci
    )

    s = json.dumps(agg, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
        print(f"[OK] wrote: {args.out}")
    else:
        print(s)
