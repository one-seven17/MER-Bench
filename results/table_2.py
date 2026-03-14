import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

EPS = 1e-12

# --- IMPORTANT: non-collapse floor for soft conjunction ---
# Larger => less “collapse to 0.00” after rounding; keep small to preserve ordering.
SOFT_FLOOR = 1e-3

# -------------------------
# Output column ordering (metric key -> column name)
# -------------------------
ORDER = [
    ("model", "model"),
    ("visual_generation_quality_1_5", "VQS"),
    ("visual_emotion_accuracy_1_5", "VEAS"),
    ("text_generation_quality_1_5", "TQS"),
    ("text_emotion_accuracy_1_5", "TEAS"),
    ("layout_consistency_accuracy", "LC"),
    ("overall_generation_quality_1_5", "OGQS"),
    ("tgt_emotion_hit", "OEC"),
    ("perceived_emotion_shift_1_5", "PESM"),
    ("__TA__", "TA"),
    ("__CF__", "CF"),
    ("__SS__", "SS"),
    ("__RFS__", "Overall"),
]

LIKERT_1_5 = {
    "visual_generation_quality_1_5",
    "visual_emotion_accuracy_1_5",
    "text_generation_quality_1_5",
    "text_emotion_accuracy_1_5",
    "perceived_emotion_shift_1_5",
    "overall_generation_quality_1_5",
}
UNIT_0_1 = {"layout_consistency_accuracy", "tgt_emotion_hit"}


# -------------------------
# Helpers
# -------------------------
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
    return clamp((x - 1.0) / 4.0, 0.0, 1.0)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std_unbiased(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def pct(x_unit: float) -> float:
    return 100.0 * clamp(x_unit, 0.0, 1.0)


def f2(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def cell_pm_percent(mu_unit: Any, sd_unit: Any) -> str:
    mu_s = f2(pct(float(mu_unit)))
    sd_s = f2(pct(float(sd_unit)))
    if mu_s == "" and sd_s == "":
        return ""
    return f"{mu_s}±{sd_s}"


def _get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


# -------------------------
# Non-collapsing overall: soft conjunction in log-space
# -------------------------
def soft_and_3(a: float, b: float, c: float, floor: float = SOFT_FLOOR) -> float:
    """
    A “soft product” that:
      - stays in [0,1]
      - equals 1 if all are 1
      - does NOT hit exact 0 unless components are truly ~0
      - is monotone in each argument
    Definition (log-space geometric mean with a small floor):
        soft = exp( (log(f+a)+log(f+b)+log(f+c))/3 ) - f
        normalized by (1 - f) to keep [0,1]
    """
    f = clamp(float(floor), 0.0, 0.2)  # keep sane
    a = clamp(a, 0.0, 1.0)
    b = clamp(b, 0.0, 1.0)
    c = clamp(c, 0.0, 1.0)
    # geometric mean in log domain of shifted variables
    gm = math.exp((math.log(f + a) + math.log(f + b) + math.log(f + c)) / 3.0)
    # shift back and renormalize to [0,1]
    val = (gm - f) / max(EPS, 1.0 - f)
    return clamp(val, 0.0, 1.0)


# -------------------------
# Extraction
# -------------------------
def extract_replicates(metrics_node: Dict[str, Any], metric_key: str) -> List[float]:
    """
    Preferred: metric["replicate_means"] = [run0_mean, run1_mean, ...]
    Fallback: metric["mean_over_runs"] (degenerate run list length=1)
    """
    m = _get(metrics_node, metric_key)
    if not isinstance(m, dict):
        return []

    reps = m.get("replicate_means")
    if isinstance(reps, list) and reps:
        out: List[float] = []
        for x in reps:
            if isinstance(x, (int, float)):
                out.append(float(x))
        return out

    mu = m.get("mean_over_runs")
    if isinstance(mu, (int, float)):
        return [float(mu)]
    return []


def extract_mean_std_fallback(
    metrics_node: Dict[str, Any], metric_key: str
) -> Tuple[float, float]:
    m = _get(metrics_node, metric_key)
    if not isinstance(m, dict):
        return 0.0, 0.0
    mu = m.get("mean_over_runs", 0.0)
    sd = m.get("std_over_runs", 0.0)
    return (
        float(mu) if isinstance(mu, (int, float)) else 0.0,
        float(sd) if isinstance(sd, (int, float)) else 0.0,
    )


def unitize(metric_name: str, x: float) -> float:
    if metric_name in LIKERT_1_5:
        return norm_1_5_to_0_1(float(x))
    if metric_name in UNIT_0_1:
        return clamp(float(x), 0.0, 1.0)
    return clamp(float(x), 0.0, 1.0)


def unitize_sd(metric_name: str, sd_raw: float) -> float:
    if metric_name in LIKERT_1_5:
        return abs(float(sd_raw)) / 4.0
    return abs(float(sd_raw))


def metric_unit_mean_std(
    metrics_node: Dict[str, Any], metric_key: str
) -> Tuple[float, float, int]:
    """
    Get metric mean/std in unit space.
    Preference:
      - If replicate_means exists => unitize each replicate, compute mean/std from replicates.
      - Else => unitize mean_over_runs and scale std_over_runs accordingly.
    Returns (mu_unit, sd_unit, R)
    """
    reps = extract_replicates(metrics_node, metric_key)
    if len(reps) >= 2:
        reps_u = [unitize(metric_key, x) for x in reps]
        return mean(reps_u), std_unbiased(reps_u), len(reps_u)
    if len(reps) == 1:
        mu_u = unitize(metric_key, reps[0])
        return mu_u, 0.0, 1

    mu_raw, sd_raw = extract_mean_std_fallback(metrics_node, metric_key)
    mu_u = unitize(metric_key, mu_raw)
    sd_u = unitize_sd(metric_key, sd_raw)
    # cap to not exceed boundaries too aggressively
    sd_u = min(sd_u, mu_u, 1.0 - mu_u) if sd_u > 0 else 0.0
    return mu_u, sd_u, 0


# -------------------------
# TA/CF/SS/RFS per-run in unit space
# -------------------------
def compute_components_per_run(
    metrics_node: Dict[str, Any], alpha: float = 3.0
) -> Dict[str, Dict[str, float]]:
    """
    Compute TA/CF/SS/RFS per-run (using replicate_means aligned by run index),
    then aggregate mean/std. All in unit space.
    RFS uses soft_and_3(TA, CF, SS) to avoid collapse-to-zero.
    """
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

    rep_raw: Dict[str, List[float]] = {
        sym: extract_replicates(metrics_node, k) for sym, k in need.items()
    }
    lengths = [len(v) for v in rep_raw.values() if v]
    if not lengths:
        z = {"mean": 0.0, "std": 0.0, "runs": 0.0}
        return {"TA": z, "CF": z, "SS": z, "RFS": z}

    R = min(lengths)
    if R <= 0:
        z = {"mean": 0.0, "std": 0.0, "runs": 0.0}
        return {"TA": z, "CF": z, "SS": z, "RFS": z}

    runs_TA: List[float] = []
    runs_CF: List[float] = []
    runs_SS: List[float] = []
    runs_RFS: List[float] = []

    for i in range(R):
        Ev = unitize(need["Ev"], rep_raw["Ev"][i])
        Et = unitize(need["Et"], rep_raw["Et"][i])
        H = unitize(need["H"], rep_raw["H"][i])
        L = unitize(need["L"], rep_raw["L"][i])
        Qv = unitize(need["Qv"], rep_raw["Qv"][i])
        Qt = unitize(need["Qt"], rep_raw["Qt"][i])
        Qo = unitize(need["Qo"], rep_raw["Qo"][i])
        S = unitize(need["S"], rep_raw["S"][i])

        TA = clamp(H * hm(Ev, Et), 0.0, 1.0)
        CF = clamp(L * geo_mean([Qv, Qt, Qo]), 0.0, 1.0)
        SS = clamp(1.0 - math.exp(-alpha * clamp(S, 0.0, 1.0)), 0.0, 1.0)

        # ---- non-collapsing overall ----
        RFS = soft_and_3(TA, CF, SS, floor=SOFT_FLOOR)

        runs_TA.append(TA)
        runs_CF.append(CF)
        runs_SS.append(SS)
        runs_RFS.append(RFS)

    def pack(xs: List[float]) -> Dict[str, float]:
        return {"mean": mean(xs), "std": std_unbiased(xs), "runs": float(len(xs))}

    return {
        "TA": pack(runs_TA),
        "CF": pack(runs_CF),
        "SS": pack(runs_SS),
        "RFS": pack(runs_RFS),
    }


# -------------------------
# Main
# -------------------------
def main() -> None:
    global SOFT_FLOOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to main.py output JSON")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument(
        "--alpha", type=float, default=3.0, help="SS curve alpha (default 3.0)"
    )
    ap.add_argument(
        "--soft-floor",
        type=float,
        default=SOFT_FLOOR,
        help="Soft floor for non-collapsing RFS (default 1e-3). Larger => less collapse.",
    )
    args = ap.parse_args()

    SOFT_FLOOR = float(args.soft_floor)

    obj = json.loads(Path(args.input).read_text(encoding="utf-8"))
    models = _get(obj, "results", "model_view", "models")
    if not isinstance(models, dict):
        raise SystemExit("Cannot find JSON path: results.model_view.models")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([col_name for _, col_name in ORDER])

        for model_name, metrics_node in models.items():
            if not isinstance(metrics_node, dict):
                continue

            comps = compute_components_per_run(metrics_node, alpha=float(args.alpha))

            row: List[str] = []
            for key, _col in ORDER:
                if key == "model":
                    row.append(model_name)
                    continue

                if key == "__TA__":
                    row.append(cell_pm_percent(comps["TA"]["mean"], comps["TA"]["std"]))
                    continue
                if key == "__CF__":
                    row.append(cell_pm_percent(comps["CF"]["mean"], comps["CF"]["std"]))
                    continue
                if key == "__SS__":
                    row.append(cell_pm_percent(comps["SS"]["mean"], comps["SS"]["std"]))
                    continue
                if key == "__RFS__":
                    row.append(
                        cell_pm_percent(comps["RFS"]["mean"], comps["RFS"]["std"])
                    )
                    continue

                mu_u, sd_u, _R = metric_unit_mean_std(metrics_node, key)
                row.append(cell_pm_percent(mu_u, sd_u))

            w.writerow(row)


if __name__ == "__main__":
    main()
