#!/usr/bin/env python3
"""
Score fixed-duration CSI windows from a Nexmon UDP/5500 PCAP.

Default policy:
  - start scanning after 15 s
  - use 60 s physical-time windows
  - move the candidate start by 1 s each step
  - rank windows by gap, coverage, burst, density, and packet-rate stability

This script is intentionally read-only with respect to the CSI data. It does
not trim, deburst, interpolate, or overwrite the original capture. It only
reports which physical 60 s interval looks best.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clean_nexmon_csi_v2 import (  # noqa: E402
    detect_extreme_static_subcarriers,
    known_80mhz_null_and_pilot_indices,
    save_heatmap_compare,
    suppress_frame_bursts,
)
from select_clean_60s_from_tail import (  # noqa: E402
    candidate_metrics,
    clean_jsonable,
    read_pcap_records_and_csi,
    sort_parsed_by_timestamp,
)


def robust_cv(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - med)))
    return mad / max(abs(med), 1e-6)


def raw_1ms_density_metrics(ts_w: np.ndarray, start_s: float, duration_s: float, target_hz: float) -> dict:
    n = int(round(duration_s * target_hz))
    rel = ts_w - start_s
    slots = np.floor(rel * target_hz).astype(np.int64)
    slots = slots[(slots >= 0) & (slots < n)]
    density = np.bincount(slots, minlength=n).astype(np.int32) if slots.size else np.zeros(n, dtype=np.int32)
    raw_valid = density > 0

    # Per-second packet density is a useful burstiness proxy that stays easy
    # to interpret beside packet-rate and gap metrics.
    sec_n = int(math.ceil(duration_s))
    sec_slots = np.clip(np.floor(rel).astype(np.int64), 0, sec_n - 1)
    sec_density = np.bincount(sec_slots, minlength=sec_n).astype(np.float32) if sec_slots.size else np.zeros(sec_n)

    return {
        "raw_1ms_coverage": float(np.mean(raw_valid)),
        "empty_1ms_slots": int(np.sum(~raw_valid)),
        "multi_packet_1ms_slots": int(np.sum(density > 1)),
        "max_packets_in_1ms": int(np.max(density)) if density.size else 0,
        "packet_density_per_second_cv": robust_cv(sec_density),
        "packet_density_per_second_min": float(np.min(sec_density)) if sec_density.size else 0.0,
        "packet_density_per_second_max": float(np.max(sec_density)) if sec_density.size else 0.0,
    }


def bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = mask.astype(bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts, ends))


def run_stats_from_mask(mask: np.ndarray, target_hz: float, prefix: str) -> dict:
    runs = bool_runs(mask)
    durations = np.asarray([(e - s) / target_hz for s, e in runs], dtype=np.float32)
    total_s = float(np.sum(durations)) if durations.size else 0.0
    out = {
        f"{prefix}_run_count": int(len(runs)),
        f"{prefix}_total_s": total_s,
        f"{prefix}_ratio": float(total_s / max(mask.size / target_hz, 1e-6)),
        f"{prefix}_longest_s": float(np.max(durations)) if durations.size else 0.0,
        f"{prefix}_p95_ms": float(np.percentile(durations, 95) * 1000.0) if durations.size else 0.0,
    }
    return out


def enhanced_window_metrics(row: dict, ts_w: np.ndarray, args: argparse.Namespace) -> dict:
    """Add metrics that describe structure, not just scalar maxima."""
    out: dict[str, float | int] = {}
    dt = np.diff(ts_w.astype(np.float64))
    duration = float(args.window_seconds)

    if dt.size:
        gap50 = dt[dt > 0.050]
        gap100 = dt[dt > 0.100]
        gap150 = dt[dt > 0.150]
        gap250 = dt[dt > 0.250]
        out.update(
            {
                "gaps_gt_50ms": int(gap50.size),
                "gaps_gt_150ms": int(gap150.size),
                "total_gap_excess_gt50_s": float(np.sum(gap50 - 0.050)) if gap50.size else 0.0,
                "total_gap_excess_gt100_s": float(np.sum(gap100 - 0.100)) if gap100.size else 0.0,
                "total_gap_excess_gt150_s": float(np.sum(gap150 - 0.150)) if gap150.size else 0.0,
                "total_gap_excess_gt250_s": float(np.sum(gap250 - 0.250)) if gap250.size else 0.0,
                "total_gap_duration_gt100_s": float(np.sum(gap100)) if gap100.size else 0.0,
                "gap_dt_cv": robust_cv(dt.astype(np.float32)),
            }
        )
    else:
        out.update(
            {
                "gaps_gt_50ms": 0,
                "gaps_gt_150ms": 0,
                "total_gap_excess_gt50_s": 0.0,
                "total_gap_excess_gt100_s": 0.0,
                "total_gap_excess_gt150_s": 0.0,
                "total_gap_excess_gt250_s": 0.0,
                "total_gap_duration_gt100_s": 0.0,
                "gap_dt_cv": float("nan"),
            }
        )

    valid = row.get("_valid")
    burst_mask = row.get("_burst_mask")
    if isinstance(valid, np.ndarray):
        invalid = ~valid.astype(bool)
        out.update(run_stats_from_mask(invalid, args.target_hz, "invalid"))
        out.update(run_stats_from_mask(valid.astype(bool), args.target_hz, "valid"))

        edge_n = int(round(args.edge_seconds * args.target_hz))
        edge_n = min(edge_n, valid.size // 2) if valid.size else 0
        if edge_n > 0:
            out["start_invalid_ratio"] = float(np.mean(invalid[:edge_n]))
            out["end_invalid_ratio"] = float(np.mean(invalid[-edge_n:]))
            out["edge_invalid_ratio"] = float(0.5 * (out["start_invalid_ratio"] + out["end_invalid_ratio"]))
        else:
            out["start_invalid_ratio"] = 0.0
            out["end_invalid_ratio"] = 0.0
            out["edge_invalid_ratio"] = 0.0

    if isinstance(burst_mask, np.ndarray):
        out.update(run_stats_from_mask(burst_mask.astype(bool), args.target_hz, "burst"))

    out["packet_rate_deficit_ratio"] = float(max(0.0, args.expected_min_rate_hz - row.get("avg_rate_hz", 0.0)) / max(args.expected_min_rate_hz, 1e-6))
    out["window_fragmentation_index"] = float(out.get("invalid_run_count", 0) / max(duration, 1e-6))
    out["effective_good_time_s"] = float(duration - out.get("invalid_total_s", 0.0) - out.get("burst_total_s", 0.0))
    out["effective_good_time_ratio"] = float(out["effective_good_time_s"] / max(duration, 1e-6))
    return out


def score_window(row: dict, args: argparse.Namespace) -> float:
    """Lower is better, with anti-specialization safeguards."""
    penalties = penalty_breakdown(row, args)
    weighted_mean = 0.0
    weight_sum = 0.0
    for name, penalty in penalties.items():
        weight = getattr(args, f"w_{name}")
        weighted_mean += weight * penalty
        weight_sum += weight
    fail_count = int(sum(p >= 1.0 for p in penalties.values()))
    worst_penalty = float(max(penalties.values())) if penalties else 0.0
    score = weighted_mean / max(weight_sum, 1e-6)
    score += args.w_worst_penalty * worst_penalty
    score += args.w_fail_count * fail_count
    return float(score)


def penalty_breakdown(row: dict, args: argparse.Namespace) -> dict[str, float]:
    """Return normalized penalties. About 0 is good; >=1 crosses a soft limit."""
    max_gap_ms = row.get("max_gap_s", 0.0) * 1000.0
    longest_invalid_ms = row.get("longest_invalid_gap_s", 0.0) * 1000.0
    burst_ratio = row.get("burst_ratio", 0.0)
    coverage = row.get("valid_coverage", 0.0)
    raw_cov = row.get("raw_1ms_coverage", 0.0)
    density_cv = row.get("packet_density_per_second_cv", 0.0)
    density_cv = 0.0 if not np.isfinite(density_cv) else density_cv

    return {
        "max_gap": max_gap_ms / max(args.max_gap_ms, 1e-6),
        "long_gap": longest_invalid_ms / max(args.max_invalid_gap_ms, 1e-6),
        "total_gap": row.get("total_gap_excess_gt100_s", 0.0) / max(args.max_total_gap_excess_gt100_s, 1e-6),
        "gap50": row.get("gaps_gt_50ms", 0) / max(args.max_gaps_gt_50ms, 1),
        "gap100": row.get("gaps_gt_100ms", 0) / max(args.max_gaps_gt_100ms, 1),
        "gap150": row.get("gaps_gt_150ms", 0) / max(args.max_gaps_gt_150ms, 1),
        "gap250": row.get("gaps_gt_250ms", 0) / max(args.max_gaps_gt_250ms, 1),
        "gap500": row.get("gaps_gt_500ms", 0) / max(args.max_gaps_gt_500ms, 1),
        "burst": burst_ratio / max(args.max_burst_ratio, 1e-6),
        "burst_run": row.get("burst_longest_s", 0.0) * 1000.0 / max(args.max_burst_run_ms, 1e-6),
        "coverage": max(0.0, args.min_coverage - coverage) / max(1.0 - args.min_coverage, 1e-6),
        "raw_coverage": max(0.0, args.min_raw_1ms_coverage - raw_cov) / max(args.min_raw_1ms_coverage, 1e-6),
        "density_cv": density_cv / max(args.max_density_cv, 1e-6),
        "dt_cv": row.get("gap_dt_cv", 0.0) / max(args.max_dt_cv, 1e-6),
        "edge_invalid": row.get("edge_invalid_ratio", 0.0) / max(args.max_edge_invalid_ratio, 1e-6),
        "fragmentation": row.get("window_fragmentation_index", 0.0) / max(args.max_fragmentation_index, 1e-6),
        "good_time": max(0.0, args.min_effective_good_time_ratio - row.get("effective_good_time_ratio", 0.0))
        / max(1.0 - args.min_effective_good_time_ratio, 1e-6),
        "rate_deficit": row.get("packet_rate_deficit_ratio", 0.0),
    }


def quality_grade(row: dict, args: argparse.Namespace) -> str:
    p = penalty_breakdown(row, args)
    fails = sum(v >= 1.0 for v in p.values())
    worst = max(p.values()) if p else 0.0
    if row["strict_pass"]:
        return "A"
    if row.get("gaps_gt_500ms", 0) > 0 or worst >= 2.0 or fails >= 4:
        return "D"
    if row.get("gaps_gt_250ms", 0) > 0 or fails >= 2:
        return "C"
    return "B"


def strict_check(row: dict, args: argparse.Namespace) -> tuple[bool, str]:
    checks = [
        (row["packet_count"] >= args.min_packets, f"packet_count<{args.min_packets}"),
        (row["max_gap_s"] * 1000.0 <= args.max_gap_ms, f"max_gap>{args.max_gap_ms:g}ms"),
        (row["gaps_gt_50ms"] <= args.max_gaps_gt_50ms, f"gaps_gt_50ms>{args.max_gaps_gt_50ms}"),
        (row["gaps_gt_100ms"] <= args.max_gaps_gt_100ms, f"gaps_gt_100ms>{args.max_gaps_gt_100ms}"),
        (row["gaps_gt_150ms"] <= args.max_gaps_gt_150ms, f"gaps_gt_150ms>{args.max_gaps_gt_150ms}"),
        (row["gaps_gt_250ms"] <= args.max_gaps_gt_250ms, f"gaps_gt_250ms>{args.max_gaps_gt_250ms}"),
        (row["gaps_gt_500ms"] <= args.max_gaps_gt_500ms, f"gaps_gt_500ms>{args.max_gaps_gt_500ms}"),
        (
            row["total_gap_excess_gt100_s"] <= args.max_total_gap_excess_gt100_s,
            f"total_gap_excess_gt100>{args.max_total_gap_excess_gt100_s:g}s",
        ),
        (
            row["longest_invalid_gap_s"] * 1000.0 <= args.max_invalid_gap_ms,
            f"longest_invalid_gap>{args.max_invalid_gap_ms:g}ms",
        ),
        (row["invalid_run_count"] <= args.max_invalid_runs, f"invalid_runs>{args.max_invalid_runs}"),
        (row["edge_invalid_ratio"] <= args.max_edge_invalid_ratio, f"edge_invalid_ratio>{args.max_edge_invalid_ratio:g}"),
        (row["valid_coverage"] >= args.min_coverage, f"coverage<{args.min_coverage:g}"),
        (row["raw_1ms_coverage"] >= args.min_raw_1ms_coverage, f"raw_1ms_coverage<{args.min_raw_1ms_coverage:g}"),
        (row["burst_ratio"] <= args.max_burst_ratio, f"burst_ratio>{args.max_burst_ratio:g}"),
        (row["burst_longest_s"] * 1000.0 <= args.max_burst_run_ms, f"burst_run>{args.max_burst_run_ms:g}ms"),
        (
            row["effective_good_time_ratio"] >= args.min_effective_good_time_ratio,
            f"effective_good_time_ratio<{args.min_effective_good_time_ratio:g}",
        ),
        (
            row["packet_density_per_second_cv"] <= args.max_density_cv
            or not np.isfinite(row["packet_density_per_second_cv"]),
            f"density_cv>{args.max_density_cv:g}",
        ),
        (
            row["gap_dt_cv"] <= args.max_dt_cv or not np.isfinite(row["gap_dt_cv"]),
            f"dt_cv>{args.max_dt_cv:g}",
        ),
    ]
    for ok, reason in checks:
        if not ok:
            return False, reason
    return True, "pass"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_score_plot(path: Path, rows: list[dict], selected_rank: int) -> None:
    if not rows:
        return
    w, h = 1400, 520
    margin_l, margin_r, margin_t, margin_b = 90, 30, 45, 80
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    starts = np.asarray([r["start_s"] for r in rows], dtype=np.float32)
    scores = np.asarray([r["score"] for r in rows], dtype=np.float32)
    max_gaps = np.asarray([r["max_gap_s"] * 1000.0 for r in rows], dtype=np.float32)
    bursts = np.asarray([r["burst_ratio"] * 100.0 for r in rows], dtype=np.float32)
    total_gap = np.asarray([r.get("total_gap_excess_gt100_s", 0.0) for r in rows], dtype=np.float32)
    good_time = np.asarray([r.get("effective_good_time_ratio", 0.0) * 100.0 for r in rows], dtype=np.float32)
    pass_mask = np.asarray([bool(r["strict_pass"]) for r in rows])

    def xmap(v: float) -> int:
        if starts.size <= 1 or float(starts[-1]) == float(starts[0]):
            return margin_l + plot_w // 2
        return int(margin_l + (v - float(starts[0])) / (float(starts[-1]) - float(starts[0])) * plot_w)

    def ymap(v: float, vmax: float) -> int:
        vmax = max(vmax, 1e-6)
        return int(margin_t + plot_h - min(max(v / vmax, 0.0), 1.0) * plot_h)

    score_vmax = float(np.nanpercentile(scores, 95)) if np.any(np.isfinite(scores)) else 1.0
    gap_vmax = max(float(np.nanpercentile(max_gaps, 95)), 250.0)
    burst_vmax = max(float(np.nanpercentile(bursts, 95)), 1.0)
    total_gap_vmax = max(float(np.nanpercentile(total_gap, 95)), 1.0)
    good_time_vmax = 100.0

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((margin_l, 15), "Candidate 60s window scores (lower is better)", fill=(20, 20, 20))
    d.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill=(120, 120, 120))
    d.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(120, 120, 120))

    for yfrac in (0.25, 0.5, 0.75):
        y = int(margin_t + plot_h * yfrac)
        d.line((margin_l, y, margin_l + plot_w, y), fill=(230, 230, 230))

    def draw_series(vals: np.ndarray, vmax: float, color: tuple[int, int, int], width: int = 3) -> None:
        pts = [(xmap(float(s)), ymap(float(v), vmax)) for s, v in zip(starts, vals)]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=width)

    draw_series(scores, score_vmax, (28, 83, 185), 3)
    draw_series(max_gaps, gap_vmax, (210, 80, 50), 2)
    draw_series(bursts, burst_vmax, (40, 150, 95), 2)
    draw_series(total_gap, total_gap_vmax, (130, 80, 180), 2)
    draw_series(good_time, good_time_vmax, (90, 90, 90), 2)

    for i, row in enumerate(rows):
        x = xmap(float(row["start_s"]))
        y = ymap(float(row["score"]), score_vmax)
        color = (30, 150, 80) if pass_mask[i] else (170, 170, 170)
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    if 0 <= selected_rank < len(rows):
        sel = rows[selected_rank]
        x = xmap(float(sel["start_s"]))
        d.line((x, margin_t, x, margin_t + plot_h), fill=(255, 170, 0), width=4)
        d.text((x + 6, margin_t + 8), f"selected {sel['start_s']:.1f}-{sel['end_s']:.1f}s", fill=(80, 60, 0))

    d.text((margin_l, h - 55), "blue: score | red: max gap | green: burst | purple: total gap excess | gray: good-time ratio | orange: selected", fill=(30, 30, 30))
    d.text((margin_l, h - 32), f"x-axis: candidate start time from {starts[0]:.1f}s to {starts[-1]:.1f}s", fill=(60, 60, 60))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score 60s CSI windows from 15s onward.")
    parser.add_argument("pcap", type=Path)
    parser.add_argument("-o", "--outdir", type=Path, default=None)
    parser.add_argument("--start-after-s", type=float, default=15.0)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--target-hz", type=float, default=1000.0)
    parser.add_argument("--max-interp-gap-ms", type=float, default=120.0)
    parser.add_argument("--burst-qc-hz", type=float, default=100.0)
    parser.add_argument("--burst-window-seconds", type=float, default=1.0)
    parser.add_argument("--burst-z-threshold", type=float, default=4.0)
    parser.add_argument("--burst-dilate-frames", type=int, default=2)
    parser.add_argument("--remove-pilots", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--min-packets", type=int, default=1000)
    parser.add_argument("--max-gap-ms", type=float, default=250.0)
    parser.add_argument("--max-gaps-gt-50ms", type=int, default=400)
    parser.add_argument("--max-gaps-gt-100ms", type=int, default=20)
    parser.add_argument("--max-gaps-gt-150ms", type=int, default=20)
    parser.add_argument("--max-gaps-gt-250ms", type=int, default=0)
    parser.add_argument("--max-gaps-gt-500ms", type=int, default=0)
    parser.add_argument("--max-total-gap-excess-gt100-s", type=float, default=3.0)
    parser.add_argument("--max-invalid-gap-ms", type=float, default=250.0)
    parser.add_argument("--max-invalid-runs", type=int, default=300)
    parser.add_argument("--min-coverage", type=float, default=0.97)
    parser.add_argument("--min-raw-1ms-coverage", type=float, default=0.35)
    parser.add_argument("--max-burst-ratio", type=float, default=0.035)
    parser.add_argument("--max-burst-run-ms", type=float, default=250.0)
    parser.add_argument("--max-density-cv", type=float, default=0.80)
    parser.add_argument("--max-dt-cv", type=float, default=1.50)
    parser.add_argument("--edge-seconds", type=float, default=5.0)
    parser.add_argument("--max-edge-invalid-ratio", type=float, default=0.08)
    parser.add_argument("--max-fragmentation-index", type=float, default=5.0)
    parser.add_argument("--min-effective-good-time-ratio", type=float, default=0.90)
    parser.add_argument("--expected-min-rate-hz", type=float, default=500.0)

    parser.add_argument("--w-max-gap", type=float, default=1.2)
    parser.add_argument("--w-long-gap", type=float, default=1.0)
    parser.add_argument("--w-total-gap", type=float, default=1.5)
    parser.add_argument("--w-gap50", type=float, default=0.6)
    parser.add_argument("--w-gap100", type=float, default=1.4)
    parser.add_argument("--w-gap150", type=float, default=1.8)
    parser.add_argument("--w-gap250", type=float, default=2.5)
    parser.add_argument("--w-gap500", type=float, default=4.0)
    parser.add_argument("--w-burst", type=float, default=1.4)
    parser.add_argument("--w-burst-run", type=float, default=1.1)
    parser.add_argument("--w-coverage", type=float, default=1.4)
    parser.add_argument("--w-raw-coverage", type=float, default=0.8)
    parser.add_argument("--w-density-cv", type=float, default=1.0)
    parser.add_argument("--w-dt-cv", type=float, default=0.8)
    parser.add_argument("--w-edge-invalid", type=float, default=1.0)
    parser.add_argument("--w-fragmentation", type=float, default=0.8)
    parser.add_argument("--w-good-time", type=float, default=1.6)
    parser.add_argument("--w-rate-deficit", type=float, default=0.8)
    parser.add_argument("--w-worst-penalty", type=float, default=1.8)
    parser.add_argument("--w-fail-count", type=float, default=1.0)
    args = parser.parse_args()

    outdir = args.outdir or (args.pcap.parent / f"{args.pcap.stem}_window_scores")
    outdir.mkdir(parents=True, exist_ok=True)

    _global_header, _endian, _records, parsed = read_pcap_records_and_csi(args.pcap)
    parsed = sort_parsed_by_timestamp(parsed)
    abs_ts = parsed["packet_time_abs_s"].astype(np.float64)
    rel_ts = abs_ts - abs_ts[0]
    amp = parsed["amplitude"].astype(np.float32)

    base_bad = known_80mhz_null_and_pilot_indices(remove_pilots=args.remove_pilots)
    auto_bad = detect_extreme_static_subcarriers(amp, already_bad=base_bad)
    removed = np.unique(np.concatenate([base_bad, auto_bad])).astype(np.int64)
    keep = np.setdiff1d(np.arange(amp.shape[1]), removed)
    log_keep = np.log1p(amp[:, keep]).astype(np.float32)

    total_duration_s = float(rel_ts[-1] - rel_ts[0])
    last_start = total_duration_s - args.window_seconds
    if last_start < args.start_after_s:
        raise SystemExit(
            f"Capture duration {total_duration_s:.3f}s is too short for a {args.window_seconds:g}s "
            f"window starting after {args.start_after_s:g}s"
        )

    starts = np.arange(args.start_after_s, last_start + 1e-9, args.stride_seconds, dtype=np.float64)
    full_rows: list[dict] = []
    rows: list[dict] = []
    for start_s in starts:
        end_s = float(start_s + args.window_seconds)
        row = candidate_metrics(
            rel_ts,
            log_keep,
            float(start_s),
            end_s,
            target_hz=args.target_hz,
            burst_qc_hz=args.burst_qc_hz,
            max_interp_gap_s=args.max_interp_gap_ms / 1000.0,
            burst_window_seconds=args.burst_window_seconds,
            burst_z_threshold=args.burst_z_threshold,
            burst_dilate_frames=args.burst_dilate_frames,
        )
        in_win = (rel_ts >= start_s) & (rel_ts <= end_s)
        ts_w = rel_ts[in_win]
        row.update(raw_1ms_density_metrics(ts_w, float(start_s), args.window_seconds, args.target_hz))
        row.update(enhanced_window_metrics(row, ts_w, args))
        ok, reason = strict_check(row, args)
        row["strict_pass"] = bool(ok)
        row["reason"] = reason
        row["score"] = score_window(row, args)
        penalties = penalty_breakdown(row, args)
        row["worst_penalty"] = float(max(penalties.values())) if penalties else 0.0
        row["fail_count"] = int(sum(v >= 1.0 for v in penalties.values()))
        row["quality_grade"] = quality_grade(row, args)
        for name, value in penalties.items():
            row[f"penalty_{name}"] = float(value)
        full_rows.append(row)
        rows.append(clean_jsonable(row))

    strict_rows = [r for r in rows if r["strict_pass"]]
    selected = min(strict_rows, key=lambda r: r["score"]) if strict_rows else min(rows, key=lambda r: r["score"])
    selected_rank = rows.index(selected)
    selected_full = full_rows[selected_rank]

    csv_path = outdir / "candidate_60s_window_scores.csv"
    json_path = outdir / "best_window_report.json"
    plot_path = outdir / "candidate_60s_window_scores.png"
    write_csv(csv_path, rows)
    draw_score_plot(plot_path, rows, selected_rank)

    in_sel = (rel_ts >= selected["start_s"]) & (rel_ts <= selected["end_s"])
    packet_x = log_keep[in_sel].astype(np.float32)
    packet_t = (rel_ts[in_sel] - selected["start_s"]).astype(np.float64)
    effective_packet_hz = packet_x.shape[0] / max(float(packet_t[-1] - packet_t[0]), 1e-6)
    packet_after_x, packet_burst_mask, _packet_burst_score, _packet_burst_z = suppress_frame_bursts(
        packet_x,
        target_hz=effective_packet_hz,
        window_seconds=args.burst_window_seconds,
        z_threshold=args.burst_z_threshold,
        dilate_frames=args.burst_dilate_frames,
        time_axis=packet_t,
    )
    packet_before = packet_x.T
    packet_after = packet_after_x.T
    save_heatmap_compare(
        packet_before,
        packet_after,
        outdir / "selected_window_packet_axis.png",
        "Selected 60s window, packet-axis view",
        "Raw log amplitude, x-axis = CSI packet index",
        "Packet-level deburst, same CSI packet index",
    )
    np.savez_compressed(
        outdir / "selected_window_packet_level_for_visual_check.npz",
        packet_time_s=packet_t.astype(np.float32),
        packet_raw_log_amp=packet_x.astype(np.float32),
        packet_deburst_log_amp=packet_after_x.astype(np.float32),
        packet_burst_mask=packet_burst_mask.astype(bool),
        selected_start_s=np.float32(selected["start_s"]),
        selected_end_s=np.float32(selected["end_s"]),
        kept_subcarrier_indices=keep.astype(np.int16),
        removed_subcarrier_indices=removed.astype(np.int16),
    )
    save_heatmap_compare(
        selected_full["_grid_x"].T,
        selected_full["_burst_clean"].T,
        outdir / "selected_window_time_axis_before_after.png",
        "Selected 60s window, 1kHz physical-time QC view",
        "Before deburst, x-axis = physical time",
        "After deburst, x-axis = physical time",
    )

    report = {
        "source_pcap": str(args.pcap),
        "policy": {
            "start_after_s": args.start_after_s,
            "window_seconds": args.window_seconds,
            "stride_seconds": args.stride_seconds,
            "target_hz": args.target_hz,
            "max_interp_gap_ms_for_qc_grid": args.max_interp_gap_ms,
            "selection": "best strict-pass window by balanced penalty score; if none pass, best score overall",
            "score_design": "normalized multi-metric penalties plus worst-penalty and fail-count safeguards",
        },
        "capture": {
            "packet_count": int(rel_ts.size),
            "duration_s": total_duration_s,
            "kept_subcarriers": int(keep.size),
            "removed_subcarriers": removed.astype(int).tolist(),
        },
        "candidate_count": len(rows),
        "strict_pass_count": len(strict_rows),
        "selected_window": selected,
        "top_10_windows": sorted(rows, key=lambda r: r["score"])[:10],
        "outputs": {
            "csv": str(csv_path),
            "plot": str(plot_path),
            "packet_axis_heatmap": str(outdir / "selected_window_packet_axis.png"),
            "time_axis_before_after_heatmap": str(outdir / "selected_window_time_axis_before_after.png"),
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Parsed CSI packets: {rel_ts.size}")
    print(f"Capture duration: {total_duration_s:.3f}s")
    print(f"Candidate windows: {len(rows)}")
    print(f"Strict-pass windows: {len(strict_rows)}")
    print(
        "Selected window: "
        f"{selected['start_s']:.3f}s -> {selected['end_s']:.3f}s | "
        f"score={selected['score']:.3f} | strict_pass={selected['strict_pass']} | reason={selected['reason']}"
    )
    print(
        "Selected metrics: "
        f"packets={selected['packet_count']} avg_rate={selected.get('avg_rate_hz', float('nan')):.1f}Hz "
        f"max_gap={selected.get('max_gap_s', float('nan'))*1000:.1f}ms "
        f"gaps>100ms={selected.get('gaps_gt_100ms')} gaps>250ms={selected.get('gaps_gt_250ms')} "
        f"coverage={selected.get('valid_coverage', float('nan')):.4f} "
        f"raw_1ms_coverage={selected.get('raw_1ms_coverage', float('nan')):.4f} "
        f"burst_ratio={selected.get('burst_ratio', float('nan')):.4f}"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Plot: {plot_path}")
    print(f"Packet-axis heatmap: {outdir / 'selected_window_packet_axis.png'}")
    print(f"Time-axis heatmap: {outdir / 'selected_window_time_axis_before_after.png'}")


if __name__ == "__main__":
    main()
