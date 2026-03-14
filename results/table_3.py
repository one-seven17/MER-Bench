#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

EPS = 1e-12
SOFT_FLOOR = 1e-3

# =========================
# 输出列定义（每文件一行）
# =========================
BASE_METRICS = [
    ("visual_generation_quality_1_5", "VQS"),
    ("visual_emotion_accuracy_1_5", "VEAS"),
    ("text_generation_quality_1_5", "TQS"),
    ("text_emotion_accuracy_1_5", "TEAS"),
    ("layout_consistency_accuracy", "LC"),
    ("overall_generation_quality_1_5", "OGQS"),
    ("tgt_emotion_hit", "OEC"),
    ("perceived_emotion_shift_1_5", "PESM"),
]
COMP_COLS = ["TA", "CF", "SS", "RFS"]


# =========================
# 基础函数
# =========================
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


def soft_and_3(a: float, b: float, c: float, floor: float = SOFT_FLOOR) -> float:
    f = clamp(float(floor), 0.0, 0.2)
    a = clamp(a, 0.0, 1.0)
    b = clamp(b, 0.0, 1.0)
    c = clamp(c, 0.0, 1.0)

    gm = math.exp((math.log(f + a) + math.log(f + b) + math.log(f + c)) / 3.0)
    val = (gm - f) / max(EPS, 1.0 - f)
    return clamp(val, 0.0, 1.0)


def pct(x_unit: float) -> float:
    return 100.0 * clamp(x_unit, 0.0, 1.0)


def f2(x: float) -> str:
    return f"{x:.2f}"


def cell(mu: float, sd: float) -> str:
    return f"{f2(mu)}±{f2(sd)}"


def _get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


# =========================
# 指标处理
# =========================
LIKERT_1_5 = {
    "visual_generation_quality_1_5",
    "visual_emotion_accuracy_1_5",
    "text_generation_quality_1_5",
    "text_emotion_accuracy_1_5",
    "perceived_emotion_shift_1_5",
    "overall_generation_quality_1_5",
}


def unitize(metric_name: str, x: float) -> float:
    if metric_name in LIKERT_1_5:
        return norm_1_5_to_0_1(float(x))
    return clamp(float(x), 0.0, 1.0)


def extract_replicates(metrics_node: Dict[str, Any], metric_key: str) -> List[float]:
    """
    Preferred: metric["replicate_means"] = [run0_mean, run1_mean, ...]
    Fallback: metric["mean_over_runs"] (degenerate length=1)
    """
    m = metrics_node.get(metric_key)
    if not isinstance(m, dict):
        return []
    reps = m.get("replicate_means")
    if isinstance(reps, list) and reps:
        return [float(x) for x in reps if isinstance(x, (int, float))]
    mu = m.get("mean_over_runs")
    if isinstance(mu, (int, float)):
        return [float(mu)]
    return []


def metric_mean_std_cell(metrics_node: Dict[str, Any], metric_key: str) -> str:
    """
    原始指标也输出百分制：
      - 先将 mean/std 视作原尺度（1..5 或 0..1）
      - unitize 到 [0,1]
      - 再乘 100
    注意：这等价于把 1..5 线性映射到 0..100。
    """
    m = metrics_node.get(metric_key)
    if not isinstance(m, dict):
        return ""

    mu = m.get("mean_over_runs")
    sd = m.get("std_over_runs")
    if not isinstance(mu, (int, float)):
        return ""

    mu_u = unitize(metric_key, float(mu))
    sd_u = unitize(metric_key, float(sd)) if isinstance(sd, (int, float)) else 0.0

    return cell(pct(mu_u), pct(sd_u))


def compute_components_mean_std_pct(
    metrics_node: Dict[str, Any], alpha: float = 3.0
) -> Dict[str, Tuple[float, float]]:
    """
    从 factor_view.metrics 计算 TA/CF/SS/RFS 的 mean±std（百分制）。
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

    rep_raw = {k: extract_replicates(metrics_node, v) for k, v in need.items()}
    lengths = [len(v) for v in rep_raw.values() if v]
    if not lengths:
        return {"TA": (0.0, 0.0), "CF": (0.0, 0.0), "SS": (0.0, 0.0), "RFS": (0.0, 0.0)}

    R = min(lengths)
    if R <= 0:
        return {"TA": (0.0, 0.0), "CF": (0.0, 0.0), "SS": (0.0, 0.0), "RFS": (0.0, 0.0)}

    runs_ta: List[float] = []
    runs_cf: List[float] = []
    runs_ss: List[float] = []
    runs_rfs: List[float] = []

    for i in range(R):
        Ev = unitize(need["Ev"], rep_raw["Ev"][i])
        Et = unitize(need["Et"], rep_raw["Et"][i])
        H = unitize(need["H"], rep_raw["H"][i])
        L = unitize(need["L"], rep_raw["L"][i])
        Qv = unitize(need["Qv"], rep_raw["Qv"][i])
        Qt = unitize(need["Qt"], rep_raw["Qt"][i])
        Qo = unitize(need["Qo"], rep_raw["Qo"][i])
        S = unitize(need["S"], rep_raw["S"][i])

        TA = H * hm(Ev, Et)
        CF = L * geo_mean([Qv, Qt, Qo])
        SS = 1.0 - math.exp(-alpha * clamp(S, 0.0, 1.0))
        RFS = soft_and_3(TA, CF, SS)

        runs_ta.append(TA)
        runs_cf.append(CF)
        runs_ss.append(SS)
        runs_rfs.append(RFS)

    def pack(xs: List[float]) -> Tuple[float, float]:
        return (pct(mean(xs)), pct(std_unbiased(xs)))

    return {
        "TA": pack(runs_ta),
        "CF": pack(runs_cf),
        "SS": pack(runs_ss),
        "RFS": pack(runs_rfs),
    }


# =========================
# 主逻辑：每个文件一行
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit("No JSON files found.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        header = ["file"] + [abbr for _k, abbr in BASE_METRICS] + COMP_COLS
        w.writerow(header)

        for jf in json_files:
            obj = json.loads(jf.read_text(encoding="utf-8"))
            metrics_node = _get(obj, "results", "factor_view", "metrics")
            if not isinstance(metrics_node, dict):
                continue

            comps = compute_components_mean_std_pct(metrics_node, alpha=3.0)

            row: List[str] = [jf.stem]

            # 原始指标：也输出百分制
            for key, _abbr in BASE_METRICS:
                row.append(metric_mean_std_cell(metrics_node, key))

            # 组件指标：TA/CF/SS/RFS（百分制）
            row.append(cell(*comps["TA"]))
            row.append(cell(*comps["CF"]))
            row.append(cell(*comps["SS"]))
            row.append(cell(*comps["RFS"]))

            w.writerow(row)

    print(f"[OK] CSV written to: {out_path}")


if __name__ == "__main__":
    main()
