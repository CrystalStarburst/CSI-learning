#!/usr/bin/env python3
"""
Build fixed-packet CSI segments for deep-learning baselines.

Default protocol:
  - discard the first 10000 CSI packets as warm-up
  - remove known 80 MHz null/pilot subcarriers
  - convert amplitude to log1p amplitude
  - do NOT deburst by default
  - cut packet-index sliding windows:
        window = 4000 packets
        stride = 2000 packets

This follows a traditional "fixed CSI matrix" style more closely than the
irregular-timestamp pipeline. The original packet timestamps are still saved
for audit, so each fixed-packet segment can be mapped back to physical time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clean_nexmon_csi_v2 import (  # noqa: E402
    detect_extreme_static_subcarriers,
    known_80mhz_null_and_pilot_indices,
    suppress_frame_bursts,
)
from score_60s_windows import read_pcap_records_and_csi, sort_parsed_by_timestamp  # noqa: E402


def infer_label_from_name(path: Path) -> str:
    stem = path.stem.lower()
    if "slow" in stem and "walking" in stem:
        return "slow_walking"
    elif "normal" in stem and "walking" in stem:
        return "normal_walking"
    elif "sitting" in stem and "down" in stem:
        return "sitting_down"
    elif "sitting" in stem:
        return "sitting"
    elif "standing" in stem and "up" in stem:
        return "standing_up"
    elif "standing" in stem:
        return "standing"
    elif "bend" in stem:
        return "bending"
    elif "get" in stem:
        return "getting_up"
    elif "lying" in stem:
        return "lying_down"
    elif "reach" in stem: 
        return "reaching"
    elif "fall" in stem:
        return "falling"
    elif "empty" in stem:
        return "empty"
    return "unknown"


def contiguous_segment_count(n_packets: int, discard: int, window: int, stride: int) -> int:
    usable = n_packets - discard
    if usable < window:
        return 0
    return ((usable - window) // stride) + 1


def build_segments(
    packet_time_s: np.ndarray,
    log_amp: np.ndarray,
    *,
    discard_packets: int,
    window_packets: int,
    stride_packets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_segments = contiguous_segment_count(log_amp.shape[0], discard_packets, window_packets, stride_packets)
    if n_segments == 0:
        raise ValueError(
            f"Not enough packets after discard: packets={log_amp.shape[0]}, "
            f"discard={discard_packets}, window={window_packets}"
        )

    x = np.empty((n_segments, window_packets, log_amp.shape[1]), dtype=np.float32)
    t = np.empty((n_segments, window_packets), dtype=np.float32)
    start_packet = np.empty(n_segments, dtype=np.int64)
    end_packet = np.empty(n_segments, dtype=np.int64)
    duration_s = np.empty(n_segments, dtype=np.float32)

    for i in range(n_segments):
        start = discard_packets + i * stride_packets
        end = start + window_packets
        x[i] = log_amp[start:end]
        # Store segment-local physical time. t[i, 0] is always approximately 0.
        ts = packet_time_s[start:end].astype(np.float64)
        t[i] = (ts - ts[0]).astype(np.float32)
        start_packet[i] = start
        end_packet[i] = end - 1
        duration_s[i] = float(ts[-1] - ts[0])

    return x, t, start_packet, end_packet, duration_s


def summarize_segment_gaps(segment_time_s: np.ndarray) -> dict[str, float | int]:
    dt = np.diff(segment_time_s.astype(np.float64), axis=1)
    return {
        "segment_duration_mean_s": float(np.mean(segment_time_s[:, -1] - segment_time_s[:, 0])),
        "segment_duration_min_s": float(np.min(segment_time_s[:, -1] - segment_time_s[:, 0])),
        "segment_duration_max_s": float(np.max(segment_time_s[:, -1] - segment_time_s[:, 0])),
        "dt_p50_ms": float(np.percentile(dt, 50) * 1000.0),
        "dt_p95_ms": float(np.percentile(dt, 95) * 1000.0),
        "dt_p99_ms": float(np.percentile(dt, 99) * 1000.0),
        "dt_max_ms": float(np.max(dt) * 1000.0),
        "gaps_gt_50ms": int(np.sum(dt > 0.050)),
        "gaps_gt_100ms": int(np.sum(dt > 0.100)),
        "gaps_gt_250ms": int(np.sum(dt > 0.250)),
    }


def process_one_pcap(path: Path, outdir: Path, args: argparse.Namespace) -> dict:
    _global_header, _endian, _records, parsed = read_pcap_records_and_csi(path)
    parsed = sort_parsed_by_timestamp(parsed)

    abs_ts = parsed["packet_time_abs_s"].astype(np.float64)
    rel_ts = abs_ts - abs_ts[0]
    amp = parsed["amplitude"].astype(np.float32)

    known_bad = known_80mhz_null_and_pilot_indices(remove_pilots=args.remove_pilots)
    auto_bad = (
        detect_extreme_static_subcarriers(amp, already_bad=known_bad)
        if args.auto_remove_static
        else np.array([], dtype=np.int64)
    )
    removed = np.unique(np.concatenate([known_bad, auto_bad])).astype(np.int64)
    kept = np.setdiff1d(np.arange(amp.shape[1]), removed).astype(np.int64)

    log_amp = np.log1p(amp[:, kept]).astype(np.float32)
    source_key = "minimal_log_amp"

    burst_mask = np.zeros(log_amp.shape[0], dtype=bool)
    if args.deburst:
        duration = float(rel_ts[-1] - rel_ts[0]) if rel_ts.size > 1 else 0.0
        effective_hz = log_amp.shape[0] / max(duration, 1e-6)
        log_amp, burst_mask, _score, _z = suppress_frame_bursts(
            log_amp,
            target_hz=effective_hz,
            window_seconds=args.burst_window_seconds,
            z_threshold=args.burst_z_threshold,
            dilate_frames=args.burst_dilate_frames,
            time_axis=rel_ts,
        )
        source_key = "deburst_log_amp"

    x, t, start_packet, end_packet, duration_s = build_segments(
        rel_ts,
        log_amp,
        discard_packets=args.discard_packets,
        window_packets=args.window_packets,
        stride_packets=args.stride_packets,
    )

    label = args.label if args.label is not None else infer_label_from_name(path)
    y_str = np.asarray([label] * x.shape[0])

    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / f"{path.stem}_fixedpkt_w{args.window_packets}_s{args.stride_packets}.npz"
    np.savez_compressed(
        out_npz,
        X=x,
        packet_time_s=t,
        label_str=y_str,
        source_file=np.asarray([path.name] * x.shape[0]),
        source_key=np.asarray(source_key),
        segment_start_packet=start_packet,
        segment_end_packet=end_packet,
        segment_duration_s=duration_s,
        discard_packets=np.int64(args.discard_packets),
        window_packets=np.int64(args.window_packets),
        stride_packets=np.int64(args.stride_packets),
        kept_subcarrier_indices=kept.astype(np.int16),
        removed_subcarrier_indices=removed.astype(np.int16),
        auto_static_subcarrier_indices=auto_bad.astype(np.int16),
        packet_burst_mask=burst_mask.astype(bool),
    )

    raw_dt = np.diff(rel_ts)
    summary = {
        "source_file": path.name,
        "output_npz": str(out_npz),
        "label": label,
        "source_key": source_key,
        "raw_packets": int(rel_ts.size),
        "discard_packets": int(args.discard_packets),
        "segments": int(x.shape[0]),
        "window_packets": int(args.window_packets),
        "stride_packets": int(args.stride_packets),
        "kept_subcarriers": int(kept.size),
        "removed_subcarriers": int(removed.size),
        "capture_duration_s": float(rel_ts[-1] - rel_ts[0]) if rel_ts.size > 1 else 0.0,
        "capture_avg_rate_hz": float(rel_ts.size / max(rel_ts[-1] - rel_ts[0], 1e-6)) if rel_ts.size > 1 else 0.0,
        "raw_dt_p50_ms": float(np.percentile(raw_dt, 50) * 1000.0) if raw_dt.size else 0.0,
        "raw_dt_p99_ms": float(np.percentile(raw_dt, 99) * 1000.0) if raw_dt.size else 0.0,
        "raw_dt_max_ms": float(np.max(raw_dt) * 1000.0) if raw_dt.size else 0.0,
        "burst_packets_marked": int(np.sum(burst_mask)),
        **summarize_segment_gaps(t),
    }
    (outdir / f"{path.stem}_fixedpkt_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def expand_inputs(inputs: list[Path]) -> list[Path]:
    pcaps: list[Path] = []
    for item in inputs:
        if item.is_dir():
            pcaps.extend(sorted(item.glob("*.pcap")))
        else:
            pcaps.append(item)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in pcaps:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-packet CSI segment NPZ files from Nexmon PCAPs.")
    parser.add_argument("inputs", nargs="+", type=Path, help="PCAP file(s) or folder(s) containing PCAP files.")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--discard-packets", type=int, default=10000)
    parser.add_argument("--window-packets", type=int, default=4000)
    parser.add_argument("--stride-packets", type=int, default=2000)
    parser.add_argument("--label", default=None, help="Optional label override for all input files.")
    parser.add_argument("--remove-pilots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--auto-remove-static",
        action="store_true",
        help="Optionally remove file-specific extreme static subcarriers. Off by default to keep a consistent feature dimension.",
    )
    parser.add_argument("--deburst", action="store_true", help="Optional ablation: apply packet-level deburst before segmentation.")
    parser.add_argument("--burst-window-seconds", type=float, default=1.0)
    parser.add_argument("--burst-z-threshold", type=float, default=4.0)
    parser.add_argument("--burst-dilate-frames", type=int, default=2)
    args = parser.parse_args()

    pcaps = expand_inputs(args.inputs)
    if not pcaps:
        raise SystemExit("No PCAP files found.")

    if len(args.inputs) == 1:
        outdir = args.outdir or (args.inputs[0].parent / (args.inputs[0].stem + "_fixed_packet_segments_w4000_s2000"))
    else:
        outdir = Path("csi/fixed_packet_segments_w4000_s2000")
        outdir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for p in pcaps:
        print(f"Processing {p}")
        try:
            summary = process_one_pcap(p, outdir, args)
            summaries.append(summary)
            print(
                f"  segments={summary['segments']} X=({summary['segments']}, "
                f"{summary['window_packets']}, {summary['kept_subcarriers']}) "
                f"duration_mean={summary['segment_duration_mean_s']:.2f}s "
                f"label={summary['label']}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            summaries.append({"source_file": p.name, "error": str(exc)})

    write_summary_csv(outdir / "fixed_packet_segments_summary.csv", summaries)
    (outdir / "fixed_packet_segments_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary: {outdir / 'fixed_packet_segments_summary.csv'}")


if __name__ == "__main__":
    main()
