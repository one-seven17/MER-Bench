#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

EPS = 1e-12
SOFT_FLOOR = 1e-3


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
    cur = d
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
        return norm_1_5_to_0_1(x)
    return clamp(x, 0.0, 1.0)


def extract_replicates(metrics_node: Dict[str, Any], metric_key: str) -> List[float]:
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


def compute_rfs_mean_std(metrics_node: Dict[str, Any], alpha: float = 3.0):
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
        return 0.0, 0.0

    R = min(lengths)

    runs = []

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

        runs.append(RFS)

    mu = mean(runs)
    sd = std_unbiased(runs)

    return pct(mu), pct(sd)


# =========================
# 主逻辑
# =========================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    json_files = sorted(list(input_dir.glob("*.json")))

    if not json_files:
        raise SystemExit("No JSON files found.")

    table: Dict[str, Dict[str, str]] = {}

    for jf in json_files:
        obj = json.loads(jf.read_text(encoding="utf-8"))
        models = _get(obj, "results", "model_view", "models")
        if not isinstance(models, dict):
            continue

        col_name = jf.stem

        for model_name, metrics_node in models.items():
            if not isinstance(metrics_node, dict):
                continue

            mu, sd = compute_rfs_mean_std(metrics_node)

            if model_name not in table:
                table[model_name] = {}

            table[model_name][col_name] = cell(mu, sd)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    col_names = [jf.stem for jf in json_files]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["model"] + col_names)

        for model_name in sorted(table.keys()):
            row = [model_name]
            for col in col_names:
                row.append(table[model_name].get(col, ""))
            w.writerow(row)

    print(f"[OK] CSV written to: {out_path}")


if __name__ == "__main__":
    main()
