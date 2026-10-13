#!/usr/bin/env python3
"""
Select a clean fixed-duration CSI clip by scanning backward from the tail.

Why this exists:
  - Warm-up / idle periods often appear near the beginning of a Nexmon capture.
  - The trial should be defined by physical time, not by packet count.
  - We want to preserve the original packet-level CSI and timestamps while
    using a 1 kHz reference grid only for QC.

The script searches candidate windows from the end of the capture backward.
The first window satisfying the QC criteria is selected. If no window fully
passes, the best-scoring 60 s window is still exported, but the report clearly
marks it as "strict_pass: False".

Outputs:
  - a trimmed PCAP containing packets in the selected absolute time interval
  - packet-level NPZ preserving original irregular timestamps
  - optional 1 kHz QC grid NPZ
  - before/after packet-axis and time-axis heatmaps
  - CSV of candidate windows and a text QC report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clean_nexmon_csi_v2 import (  # noqa: E402
    detect_extreme_static_subcarriers,
    interpolate_bad_time_rows,
    known_80mhz_null_and_pilot_indices,
    resample_with_gap_mask,
    rolling_nanmedian_1d,
    save_heatmap_compare,
    sort_and_deduplicate,
)


@dataclass
class PcapRecord:
    sec: int
    frac: int
    caplen: int
    origlen: int
    raw_header: bytes
    raw_packet: bytes
    timestamp: float
    is_csi: bool
    csi_index: int | None


def pcap_endian_and_scale(magic: bytes) -> tuple[str, float]:
    if magic == b"\xd4\xc3\xb2\xa1":
        return "<", 1e-6
    if magic == b"\xa1\xb2\xc3\xd4":
        return ">", 1e-6
    if magic == b"\x4d\x3c\xb2\xa1":
        return "<", 1e-9
    if magic == b"\xa1\xb2\x3c\x4d":
        return ">", 1e-9
    raise ValueError(f"Unsupported PCAP magic: {magic!r}")


def read_pcap_records_and_csi(
    path: Path,
    csi_header_bytes: int = 18,
    n_subcarriers: int = 256,
) -> tuple[bytes, str, list[PcapRecord], dict[str, np.ndarray]]:
    """Read PCAP records and parse Nexmon UDP/5500 CSI and metadata.

    Returns:
      global_header, endian, records, parsed arrays
    """
    records: list[PcapRecord] = []
    csi_ts: list[float] = []
    complex_csi: list[np.ndarray] = []
    amps: list[np.ndarray] = []
    rssi: list[int] = []
    src_mac: list[np.ndarray] = []
    seq_no: list[int] = []
    csi_config: list[int] = []
    chanspec: list[int] = []
    chip: list[int] = []
    payload_lens: list[int] = []
    csi_record_indices: list[int] = []

    with path.open("rb") as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            raise ValueError(f"{path} is too small to be a PCAP file")
        endian, frac_scale = pcap_endian_and_scale(global_header[:4])
        pkt_header = struct.Struct(endian + "IIII")

        expected_csi_bytes = n_subcarriers * 2 * 2
        min_payload = csi_header_bytes + expected_csi_bytes

        while True:
            raw_header = f.read(16)
            if len(raw_header) == 0:
                break
            if len(raw_header) < 16:
                break
            sec, frac, caplen, origlen = pkt_header.unpack(raw_header)
            pkt = f.read(caplen)
            if len(pkt) < caplen:
                break

            timestamp = sec + frac * frac_scale
            is_csi = False
            csi_index: int | None = None

            if len(pkt) >= 42 and pkt[12:14] == b"\x08\x00":
                ip_start = 14
                ihl = (pkt[ip_start] & 0x0F) * 4
                if ihl >= 20 and ip_start + ihl + 8 <= len(pkt) and pkt[ip_start + 9] == 17:
                    udp_start = ip_start + ihl
                    sport, dport, udp_len, _udp_sum = struct.unpack("!HHHH", pkt[udp_start : udp_start + 8])
                    if sport == 5500 and dport == 5500:
                        payload = pkt[udp_start + 8 : udp_start + udp_len]
                        if len(payload) >= min_payload:
                            raw = payload[csi_header_bytes : csi_header_bytes + expected_csi_bytes]
                            vals = np.frombuffer(raw, dtype="<i2")
                            if vals.size == n_subcarriers * 2:
                                iq = vals.reshape(n_subcarriers, 2)
                                csi = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
                                complex_csi.append(csi.astype(np.complex64))
                                amps.append(np.abs(csi).astype(np.float32))
                                csi_ts.append(timestamp)
                                # Nexmon CSI UDP header, commonly:
                                # [0:2] magic, [2] RSSI int8, [3] FC, [4:10] source MAC,
                                # [10:12] seq, [12:14] core/spatial config,
                                # [14:16] chanspec, [16:18] chip.
                                rssi.append(int(np.frombuffer(payload[2:3], dtype=np.int8)[0]))
                                src_mac.append(np.frombuffer(payload[4:10], dtype=np.uint8).copy())
                                seq_no.append(int.from_bytes(payload[10:12], "little", signed=False))
                                csi_config.append(int.from_bytes(payload[12:14], "little", signed=False))
                                chanspec.append(int.from_bytes(payload[14:16], "little", signed=False))
                                chip.append(int.from_bytes(payload[16:18], "little", signed=False))
                                payload_lens.append(len(payload))
                                csi_record_indices.append(len(records))
                                is_csi = True
                                csi_index = len(csi_ts) - 1

            records.append(
                PcapRecord(
                    sec=sec,
                    frac=frac,
                    caplen=caplen,
                    origlen=origlen,
                    raw_header=raw_header,
                    raw_packet=pkt,
                    timestamp=timestamp,
                    is_csi=is_csi,
                    csi_index=csi_index,
                )
            )

    if not amps:
        raise ValueError(f"No Nexmon UDP/5500 CSI packets were parsed from {path}")

    parsed = {
        "packet_time_abs_s": np.asarray(csi_ts, dtype=np.float64),
        "complex_csi": np.vstack(complex_csi).astype(np.complex64),
        "amplitude": np.vstack(amps).astype(np.float32),
        "rssi_dbm": np.asarray(rssi, dtype=np.int16),
        "source_mac": np.vstack(src_mac).astype(np.uint8),
        "sequence_number": np.asarray(seq_no, dtype=np.uint16),
        "csi_config": np.asarray(csi_config, dtype=np.uint16),
        "chanspec": np.asarray(chanspec, dtype=np.uint16),
        "chip": np.asarray(chip, dtype=np.uint16),
        "payload_length": np.asarray(payload_lens, dtype=np.uint16),
        "pcap_record_index": np.asarray(csi_record_indices, dtype=np.int64),
    }
    return (
        global_header,
        endian,
        records,
        parsed,
    )


def write_trimmed_pcap(global_header: bytes, records: Iterable[PcapRecord], start_abs: float, end_abs: float, out_pcap: Path) -> int:
    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_pcap.open("wb") as f:
        f.write(global_header)
        for rec in records:
            if start_abs <= rec.timestamp <= end_abs:
                f.write(rec.raw_header)
                f.write(rec.raw_packet)
                n += 1
    return n


def longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def sort_parsed_by_timestamp(parsed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ts = parsed["packet_time_abs_s"]
    order = np.argsort(ts)
    ts_sorted = ts[order]
    keep = np.concatenate([[True], np.diff(ts_sorted) > 0])
    order = order[keep]
    return {k: (v[order] if isinstance(v, np.ndarray) and v.shape[:1] == ts.shape[:1] else v) for k, v in parsed.items()}


def make_grid_mask(packet_time_s: np.ndarray, duration_s: float, target_hz: float) -> dict[str, np.ndarray]:
    """Build raw 1 kHz observation masks from packet timestamps."""
    step = 1.0 / target_hz
    n = int(round(duration_s * target_hz))
    grid = np.arange(n, dtype=np.float64) * step
    ts = np.sort(packet_time_s.astype(np.float64))

    density = np.zeros(n, dtype=np.int16)
    if ts.size:
        slot = np.floor(ts * target_hz).astype(np.int64)
        slot = slot[(slot >= 0) & (slot < n)]
        if slot.size:
            density[:] = np.bincount(slot, minlength=n).astype(np.int16)
    raw_valid = density > 0

    pos = np.searchsorted(ts, grid, side="right") - 1
    has_prev = pos >= 0
    age = np.full(n, np.nan, dtype=np.float32)
    age[has_prev] = (grid[has_prev] - ts[pos[has_prev]]).astype(np.float32)

    nearest = np.full(n, -1, dtype=np.int64)
    if ts.size:
        right = np.searchsorted(ts, grid, side="left")
        left = np.clip(right - 1, 0, ts.size - 1)
        right_clip = np.clip(right, 0, ts.size - 1)
        choose_right = np.abs(ts[right_clip] - grid) < np.abs(grid - ts[left])
        nearest[:] = np.where(choose_right, right_clip, left).astype(np.int64)

    return {
        "time_grid_s": grid.astype(np.float32),
        "raw_valid_mask": raw_valid.astype(bool),
        "packet_density_1ms": density,
        "age_since_last_packet_s": age,
        "nearest_packet_index": nearest,
    }


def resample_on_fixed_grid(
    packet_time_s: np.ndarray,
    x: np.ndarray,
    *,
    duration_s: float,
    target_hz: float,
    max_interp_gap_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample packet data onto an exact [0, duration_s) grid."""
    order = np.argsort(packet_time_s)
    ts = packet_time_s[order].astype(np.float64)
    x = x[order].astype(np.float32)
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts = ts[keep]
    x = x[keep]

    n = int(round(duration_s * target_hz))
    grid = (np.arange(n, dtype=np.float64) / target_hz).astype(np.float64)
    pos = np.searchsorted(ts, grid, side="left")
    left = np.clip(pos - 1, 0, len(ts) - 1)
    right = np.clip(pos, 0, len(ts) - 1)
    surrounded = (pos > 0) & (pos < len(ts))
    source_gap = ts[right] - ts[left]
    valid = surrounded & (source_gap <= max_interp_gap_s)

    # Mark slots that actually contain at least one packet as valid, even near
    # the beginning/end of the fixed window.
    density = make_grid_mask(ts, duration_s=duration_s, target_hz=target_hz)["packet_density_1ms"]
    valid |= density > 0

    out = np.empty((grid.size, x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(grid, ts, x[:, j]).astype(np.float32)
    out[~valid, :] = np.nan
    return grid.astype(np.float32), out, valid.astype(bool)


def save_raw_packet_and_grid_npz(
    outdir: Path,
    *,
    source_pcap: Path,
    parsed: dict[str, np.ndarray],
    rel_time_s: np.ndarray,
    keep_subcarriers: np.ndarray,
    removed_subcarriers: np.ndarray,
    target_hz: float,
    duration_s: float,
    prefix: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    amp = parsed["amplitude"].astype(np.float32)
    complex_csi = parsed["complex_csi"].astype(np.complex64)
    grid_mask = make_grid_mask(rel_time_s, duration_s=duration_s, target_hz=target_hz)

    np.savez_compressed(
        outdir / f"{prefix}_raw_packet_level.npz",
        source_pcap=str(source_pcap),
        packet_time_s=rel_time_s.astype(np.float32),
        packet_time_abs_s=parsed["packet_time_abs_s"].astype(np.float64),
        packet_index=np.arange(rel_time_s.size, dtype=np.int64),
        complex_csi=complex_csi,
        amplitude=amp,
        phase=np.angle(complex_csi).astype(np.float32),
        rssi_dbm=parsed["rssi_dbm"].astype(np.int16),
        source_mac=parsed["source_mac"].astype(np.uint8),
        sequence_number=parsed["sequence_number"].astype(np.uint16),
        csi_config=parsed["csi_config"].astype(np.uint16),
        chanspec=parsed["chanspec"].astype(np.uint16),
        chip=parsed["chip"].astype(np.uint16),
        payload_length=parsed["payload_length"].astype(np.uint16),
        pcap_record_index=parsed["pcap_record_index"].astype(np.int64),
        kept_subcarrier_indices=keep_subcarriers.astype(np.int16),
        removed_subcarrier_indices=removed_subcarriers.astype(np.int16),
    )
    np.savez_compressed(
        outdir / f"{prefix}_raw_grid_mask_{int(round(target_hz))}hz.npz",
        source_pcap=str(source_pcap),
        target_hz=np.float32(target_hz),
        duration_s=np.float32(duration_s),
        **grid_mask,
    )


def contiguous_runs(mask: np.ndarray):
    mask = mask.astype(bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts, ends))


def interpolate_short_invalid_runs(x: np.ndarray, valid: np.ndarray, time_s: np.ndarray, max_gap_s: float):
    out = x.copy().astype(np.float32)
    valid_out = valid.copy().astype(bool)
    fill_mask = np.zeros(valid.shape[0], dtype=bool)
    keep_missing_mask = np.zeros(valid.shape[0], dtype=bool)
    step_s = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.001

    for start, end in contiguous_runs(~valid_out):
        run_duration = float((end - start) * step_s)
        left, right = start - 1, end
        can_fill = (
            run_duration <= max_gap_s
            and left >= 0
            and right < len(valid_out)
            and valid_out[left]
            and valid_out[right]
        )
        if can_fill:
            fill_mask[start:end] = True
        else:
            keep_missing_mask[start:end] = True

    if np.any(fill_mask):
        good = valid_out & np.any(np.isfinite(out), axis=1)
        for j in range(out.shape[1]):
            finite_good = good & np.isfinite(out[:, j])
            if np.sum(finite_good) >= 2:
                out[fill_mask, j] = np.interp(time_s[fill_mask], time_s[finite_good], out[finite_good, j]).astype(np.float32)
        valid_out[fill_mask] = True
    out[~valid_out, :] = np.nan
    return out, valid_out, fill_mask, keep_missing_mask


def save_interpolated_selected_grid(
    outdir: Path,
    *,
    source_pcap: Path,
    selected_pcap: Path,
    time_s: np.ndarray,
    deburst_log_amp: np.ndarray,
    valid_mask: np.ndarray,
    raw_grid_mask: dict[str, np.ndarray],
    burst_mask: np.ndarray,
    burst_score: np.ndarray,
    burst_z: np.ndarray,
    keep_subcarriers: np.ndarray,
    removed_subcarriers: np.ndarray,
    max_gap_ms: float,
) -> None:
    interp_dir = outdir / f"interp_le{int(round(max_gap_ms))}ms"
    interp_dir.mkdir(parents=True, exist_ok=True)
    x_interp, valid_interp, fill_mask, remain_mask = interpolate_short_invalid_runs(
        deburst_log_amp, valid_mask, time_s.astype(np.float64), max_gap_ms / 1000.0
    )
    step_s = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.001
    qc = {
        "source_pcap": str(source_pcap),
        "selected_pcap": str(selected_pcap),
        "max_gap_ms": max_gap_ms,
        "grid_frames": int(time_s.size),
        "original_valid_coverage": float(np.mean(valid_mask)),
        "filled_frames": int(np.sum(fill_mask)),
        "filled_runs": int(len(contiguous_runs(fill_mask))),
        "remaining_valid_coverage": float(np.mean(valid_interp)),
        "remaining_invalid_frames": int(np.sum(~valid_interp)),
        "remaining_longest_invalid_ms": float(longest_true_run(~valid_interp) * step_s * 1000.0),
    }
    np.savez_compressed(
        interp_dir / "processed_csi_1000hz_interp.npz",
        source_pcap=str(source_pcap),
        selected_pcap=str(selected_pcap),
        time_s=time_s.astype(np.float32),
        X=x_interp.astype(np.float32),
        csi_log_amp_interp=x_interp.astype(np.float32),
        original_deburst_log_amp=deburst_log_amp.astype(np.float32),
        original_valid_time_mask=valid_mask.astype(bool),
        valid_time_mask_interp=valid_interp.astype(bool),
        short_gap_fill_mask=fill_mask.astype(bool),
        long_gap_remaining_mask=(~valid_interp).astype(bool),
        raw_valid_mask=raw_grid_mask["raw_valid_mask"].astype(bool),
        packet_density_1ms=raw_grid_mask["packet_density_1ms"].astype(np.int16),
        age_since_last_packet_s=raw_grid_mask["age_since_last_packet_s"].astype(np.float32),
        nearest_packet_index=raw_grid_mask["nearest_packet_index"].astype(np.int64),
        burst_time_mask=burst_mask.astype(bool),
        burst_score=burst_score.astype(np.float32),
        burst_z=burst_z.astype(np.float32),
        kept_subcarrier_indices=keep_subcarriers.astype(np.int16),
        removed_subcarrier_indices=removed_subcarriers.astype(np.int16),
    )
    with (interp_dir / "qc_report.json").open("w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)
    save_heatmap_compare(
        deburst_log_amp.T,
        x_interp.T,
        interp_dir / "01_before_after_interp_logamp.png",
        f"Selected 60s post-deburst interpolation <= {max_gap_ms:g} ms",
        "Before interpolation: debursted 1kHz CSI",
        "After interpolation: short gaps filled, long gaps white",
    )


def suppress_frame_bursts_fast(
    log_amp_grid: np.ndarray,
    *,
    target_hz: float,
    burst_qc_hz: float,
    window_seconds: float,
    z_threshold: float,
    dilate_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast physical-time burst detection.

    The CSI grid itself is preserved at target_hz. For speed, only the 1-D
    frame-level burst score is block-aggregated to burst_qc_hz for robust
    baseline estimation. The resulting block mask is mapped back to target_hz.
    """
    x = log_amp_grid.astype(np.float32)
    valid_time = np.any(np.isfinite(x), axis=1)
    if x.shape[0] == 0:
        return x.copy(), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    score = np.full(x.shape[0], np.nan, dtype=np.float32)
    if np.any(valid_time):
        # Missing grid rows are all-NaN rows. Compute ordinary row statistics
        # only on valid rows to avoid NumPy's slow masked-array nanmedian path.
        xv = x[valid_time]
        frame_med = np.median(xv, axis=1)
        frame_p90 = np.percentile(xv, 90, axis=1)
        score[valid_time] = (0.5 * frame_med + 0.5 * frame_p90).astype(np.float32)

    decim = max(1, int(round(target_hz / max(burst_qc_hz, 1e-6))))
    n_blocks = int(math.ceil(score.size / decim))
    pad = n_blocks * decim - score.size
    score_pad = np.pad(score, (0, pad), constant_values=np.nan)
    valid_pad = np.pad(valid_time, (0, pad), constant_values=False)
    score_blocks = score_pad.reshape(n_blocks, decim)
    valid_blocks = valid_pad.reshape(n_blocks, decim)

    # Use block p90 rather than max. Max is too sensitive and tends to mark
    # ordinary small fluctuations as bursts; p90 still catches vertical bright
    # stripes while staying closer to the original v2 detector behavior.
    block_score = np.full(n_blocks, np.nan, dtype=np.float32)
    any_valid = np.any(valid_blocks, axis=1)
    if np.any(any_valid):
        block_score[any_valid] = np.nanpercentile(score_blocks[any_valid], 90, axis=1).astype(np.float32)

    qc_hz_actual = target_hz / decim
    window = max(5, int(round(window_seconds * qc_hz_actual)))
    if window % 2 == 0:
        window += 1
    baseline = rolling_nanmedian_1d(block_score, window)
    resid = block_score - baseline
    local_abs = rolling_nanmedian_1d(np.abs(resid), window)
    global_abs = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    scale = 1.4826 * np.maximum(local_abs, global_abs + 1e-6)
    block_z = resid / np.maximum(scale, 1e-6)
    block_bad = any_valid & (block_z > z_threshold)

    if dilate_frames > 0 and np.any(block_bad):
        # Convert target-grid dilation frames into block units.
        dilate_blocks = max(1, int(math.ceil(dilate_frames / decim)))
        kernel = np.ones(2 * dilate_blocks + 1, dtype=np.int16)
        block_bad = np.convolve(block_bad.astype(np.int16), kernel, mode="same") > 0
        block_bad &= any_valid

    bad = np.repeat(block_bad, decim)[: score.size]
    bad &= valid_time
    z = np.repeat(block_z, decim)[: score.size].astype(np.float32)
    cleaned = interpolate_bad_time_rows(x, bad)
    return cleaned.astype(np.float32), bad.astype(bool), score.astype(np.float32), z


def candidate_metrics(
    rel_ts: np.ndarray,
    log_keep: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    target_hz: float,
    burst_qc_hz: float,
    max_interp_gap_s: float,
    burst_window_seconds: float,
    burst_z_threshold: float,
    burst_dilate_frames: int,
) -> dict:
    in_win = (rel_ts >= start_s) & (rel_ts <= end_s)
    ts_w = rel_ts[in_win]
    x_w = log_keep[in_win]

    if ts_w.size < 2:
        return {
            "start_s": start_s,
            "end_s": end_s,
            "packet_count": int(ts_w.size),
            "score": float("inf"),
            "strict_pass": False,
            "reason": "too_few_packets",
        }

    dt = np.diff(ts_w)
    duration = float(end_s - start_s)
    avg_rate = float(ts_w.size / max(duration, 1e-6))
    max_gap = float(np.max(dt)) if dt.size else 0.0

    # Shift the candidate to zero time for interpolation/QC.
    grid, grid_x, valid = resample_with_gap_mask(
        ts_w,
        x_w,
        target_hz=target_hz,
        max_interp_gap_s=max_interp_gap_s,
    )
    burst_clean, burst_mask, burst_score, burst_z = suppress_frame_bursts_fast(
        grid_x,
        target_hz=target_hz,
        burst_qc_hz=burst_qc_hz,
        window_seconds=burst_window_seconds,
        z_threshold=burst_z_threshold,
        dilate_frames=burst_dilate_frames,
    )

    valid_coverage = float(np.mean(valid))
    burst_ratio = float(np.mean(burst_mask))
    longest_invalid_s = float(longest_true_run(~valid) / target_hz)
    longest_burst_s = float(longest_true_run(burst_mask) / target_hz)

    # A compact score for fallback ranking when no window strictly passes.
    score = (
        20.0 * max(0.0, 0.97 - valid_coverage)
        + 6.0 * burst_ratio
        + 4.0 * max(0.0, max_gap - 0.5)
        + 2.0 * longest_invalid_s
        + 1.5 * longest_burst_s
    )

    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "packet_count": int(ts_w.size),
        "avg_rate_hz": avg_rate,
        "dt_p50_ms": float(np.percentile(dt, 50) * 1000.0),
        "dt_p95_ms": float(np.percentile(dt, 95) * 1000.0),
        "dt_p99_ms": float(np.percentile(dt, 99) * 1000.0),
        "max_gap_s": max_gap,
        "gaps_gt_100ms": int(np.sum(dt > 0.100)),
        "gaps_gt_250ms": int(np.sum(dt > 0.250)),
        "gaps_gt_500ms": int(np.sum(dt > 0.500)),
        "valid_coverage": valid_coverage,
        "longest_invalid_gap_s": longest_invalid_s,
        "burst_ratio": burst_ratio,
        "burst_frames": int(np.sum(burst_mask)),
        "longest_burst_s": longest_burst_s,
        "burst_z_max": float(np.nanmax(burst_z)) if np.any(np.isfinite(burst_z)) else float("nan"),
        "score": float(score),
        "strict_pass": False,
        "reason": "",
        "_grid": grid,
        "_grid_x": grid_x,
        "_valid": valid,
        "_burst_clean": burst_clean,
        "_burst_mask": burst_mask,
        "_burst_score": burst_score,
        "_burst_z": burst_z,
    }


def strict_check(m: dict, args: argparse.Namespace) -> tuple[bool, str]:
    if m["packet_count"] < args.min_packets:
        return False, f"packet_count<{args.min_packets}"
    if m["start_s"] < args.exclude_first_seconds:
        return False, f"inside_first_{args.exclude_first_seconds:g}s_warmup_zone"
    if m["max_gap_s"] > args.max_gap_ms / 1000.0:
        return False, f"max_gap>{args.max_gap_ms:g}ms"
    if m["gaps_gt_500ms"] > args.max_gaps_gt_500ms:
        return False, f"gaps_gt_500ms>{args.max_gaps_gt_500ms}"
    if m["valid_coverage"] < args.min_coverage:
        return False, f"coverage<{args.min_coverage:g}"
    if m["longest_invalid_gap_s"] > args.max_invalid_gap_ms / 1000.0:
        return False, f"longest_invalid_gap>{args.max_invalid_gap_ms:g}ms"
    if m["burst_ratio"] > args.max_burst_ratio:
        return False, f"burst_ratio>{args.max_burst_ratio:g}"
    if m["longest_burst_s"] > args.max_burst_run_ms / 1000.0:
        return False, f"longest_burst>{args.max_burst_run_ms:g}ms"
    return True, "pass"


def clean_jsonable(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan backward from the tail and export a clean fixed-duration Nexmon CSI clip."
    )
    parser.add_argument("pcap", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--scan-step-seconds", type=float, default=0.5)
    parser.add_argument(
        "--max-scan-windows",
        type=int,
        default=1,
        help="How many candidate windows to test from the tail backward. Default 1 = the last 60s only.",
    )
    parser.add_argument("--exclude-first-seconds", type=float, default=10.0)
    parser.add_argument("--target-hz", type=float, default=1000.0, help="Reference grid used for QC only.")
    parser.add_argument("--max-interp-gap-ms", type=float, default=120.0)
    parser.add_argument("--max-gap-ms", type=float, default=500.0)
    parser.add_argument("--max-gaps-gt-500ms", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--max-invalid-gap-ms", type=float, default=250.0)
    parser.add_argument("--min-packets", type=int, default=10000)
    parser.add_argument("--burst-window-seconds", type=float, default=1.0)
    parser.add_argument("--burst-qc-hz", type=float, default=100.0, help="QC-only block rate for fast burst scoring; data are not downsampled.")
    parser.add_argument("--burst-z-threshold", type=float, default=4.0)
    parser.add_argument("--burst-dilate-frames", type=int, default=2)
    parser.add_argument("--max-burst-ratio", type=float, default=0.10)
    parser.add_argument("--max-burst-run-ms", type=float, default=300.0)
    parser.add_argument("--keep-pilots", action="store_true")
    parser.add_argument("--save-qc-grid", action="store_true", default=True)
    parser.add_argument(
        "--final-interp-gap-ms",
        default="100,250",
        help="Comma-separated post-selection interpolation thresholds, e.g. '100,250'. Use empty string to disable.",
    )
    args = parser.parse_args()

    outdir = args.outdir or (args.pcap.parent / f"{args.pcap.stem}_tail60s_selected")
    outdir.mkdir(parents=True, exist_ok=True)

    global_header, _endian, records, parsed = read_pcap_records_and_csi(args.pcap)
    parsed = sort_parsed_by_timestamp(parsed)
    ts_abs = parsed["packet_time_abs_s"]
    amp = parsed["amplitude"]
    rel_ts = ts_abs - ts_abs[0]
    total_duration_s = float(rel_ts[-1])

    bad_known = known_80mhz_null_and_pilot_indices(remove_pilots=not args.keep_pilots)
    bad_auto = detect_extreme_static_subcarriers(amp, already_bad=bad_known)
    bad_all = np.array(sorted(set(bad_known.tolist()) | set(bad_auto.tolist())), dtype=np.int64)
    bad_set = set(bad_all.tolist())
    keep = np.array([i for i in range(amp.shape[1]) if i not in bad_set], dtype=np.int64)

    log_keep = np.log1p(amp[:, keep]).astype(np.float32)

    raw_full_dir = outdir / "raw_full_capture"
    save_raw_packet_and_grid_npz(
        raw_full_dir,
        source_pcap=args.pcap,
        parsed=parsed,
        rel_time_s=rel_ts,
        keep_subcarriers=keep,
        removed_subcarriers=bad_all,
        target_hz=args.target_hz,
        duration_s=total_duration_s,
        prefix="full",
    )

    if total_duration_s < args.window_seconds:
        raise ValueError(
            f"Capture duration {total_duration_s:.3f}s is shorter than requested window {args.window_seconds:.3f}s"
        )

    latest_end = total_duration_s
    earliest_end = args.window_seconds
    candidate_rows: list[dict] = []
    selected: dict | None = None

    n_steps = int(math.floor((latest_end - earliest_end) / args.scan_step_seconds)) + 1
    if args.max_scan_windows > 0:
        n_steps = min(n_steps, args.max_scan_windows - 1)
    for i in range(n_steps + 1):
        end_s = latest_end - i * args.scan_step_seconds
        start_s = end_s - args.window_seconds
        if start_s < 0:
            continue
        m = candidate_metrics(
            rel_ts,
            log_keep,
            start_s,
            end_s,
            target_hz=args.target_hz,
            burst_qc_hz=args.burst_qc_hz,
            max_interp_gap_s=args.max_interp_gap_ms / 1000.0,
            burst_window_seconds=args.burst_window_seconds,
            burst_z_threshold=args.burst_z_threshold,
            burst_dilate_frames=args.burst_dilate_frames,
        )
        ok, reason = strict_check(m, args)
        m["strict_pass"] = bool(ok)
        m["reason"] = reason
        candidate_rows.append(m)
        if ok:
            selected = m
            break

    if selected is None:
        selected = min(candidate_rows, key=lambda x: x["score"])
        original_reason = selected.get("reason", "")
        selected["strict_pass"] = False
        selected["reason"] = f"no_strict_window_found_best_score_exported; original_reason={original_reason}"

    selected_start_abs = float(ts_abs[0] + selected["start_s"])
    selected_end_abs = float(ts_abs[0] + selected["end_s"])
    trimmed_pcap = outdir / f"{args.pcap.stem}_selected_{int(round(args.window_seconds))}s.pcap"
    written_records = write_trimmed_pcap(global_header, records, selected_start_abs, selected_end_abs, trimmed_pcap)

    win_mask = (rel_ts >= selected["start_s"]) & (rel_ts <= selected["end_s"])
    win_rel = (rel_ts[win_mask] - selected["start_s"]).astype(np.float32)
    win_log = log_keep[win_mask].astype(np.float32)
    exact_grid, exact_grid_x, exact_valid = resample_on_fixed_grid(
        win_rel.astype(np.float64),
        win_log,
        duration_s=args.window_seconds,
        target_hz=args.target_hz,
        max_interp_gap_s=args.max_interp_gap_ms / 1000.0,
    )
    exact_burst_clean, exact_burst_mask, exact_burst_score, exact_burst_z = suppress_frame_bursts_fast(
        exact_grid_x,
        target_hz=args.target_hz,
        burst_qc_hz=args.burst_qc_hz,
        window_seconds=args.burst_window_seconds,
        z_threshold=args.burst_z_threshold,
        dilate_frames=args.burst_dilate_frames,
    )
    selected_parsed = {
        k: (v[win_mask] if isinstance(v, np.ndarray) and v.shape[:1] == rel_ts.shape[:1] else v)
        for k, v in parsed.items()
    }
    selected_parsed["packet_time_abs_s"] = selected_parsed["packet_time_abs_s"].astype(np.float64)
    selected_raw_dir = outdir / "raw_selected_60s"
    save_raw_packet_and_grid_npz(
        selected_raw_dir,
        source_pcap=args.pcap,
        parsed=selected_parsed,
        rel_time_s=win_rel.astype(np.float64),
        keep_subcarriers=keep,
        removed_subcarriers=bad_all,
        target_hz=args.target_hz,
        duration_s=args.window_seconds,
        prefix="selected",
    )
    # Avoid running the expensive rolling-median detector a second time on the
    # irregular packet timeline. The selected window has already been judged on
    # a physical-time 1 kHz QC grid. Here we map that QC burst mask back to the
    # original packet timestamps, then interpolate only those packet rows.
    grid_idx = np.clip(np.rint(win_rel.astype(np.float64) * args.target_hz).astype(np.int64), 0, exact_burst_mask.size - 1)
    packet_burst_mask = exact_burst_mask[grid_idx].astype(bool)
    packet_burst_score = exact_burst_score[grid_idx].astype(np.float32)
    packet_burst_z = exact_burst_z[grid_idx].astype(np.float32)
    packet_deburst = interpolate_bad_time_rows(win_log, packet_burst_mask, time_axis=win_rel).astype(np.float32)

    np.savez_compressed(
        outdir / "selected_packet_level_60s.npz",
        source_pcap=str(args.pcap),
        selected_pcap=str(trimmed_pcap),
        selected_start_s=np.float32(selected["start_s"]),
        selected_end_s=np.float32(selected["end_s"]),
        packet_time_s=win_rel,
        kept_subcarrier_indices=keep.astype(np.int16),
        removed_subcarrier_indices=bad_all.astype(np.int16),
        packet_log_amp=win_log,
        packet_deburst_log_amp=packet_deburst.astype(np.float32),
        packet_burst_mask=packet_burst_mask.astype(bool),
        packet_burst_score=packet_burst_score.astype(np.float32),
        packet_burst_z=packet_burst_z.astype(np.float32),
    )

    save_heatmap_compare(
        win_log.T,
        packet_deburst.T,
        outdir / "01_selected_packet_axis_before_after.png",
        "Selected 60s packet-level CSI before/after deburst",
        "Before, x-axis = original CSI packet index inside selected 60s",
        "After, x-axis = original CSI packet index inside selected 60s",
    )

    if args.save_qc_grid:
        # Mask aligned to the exact selected 60s grid. This grid starts at the
        # selected physical-window boundary, not at the first packet.
        packet_time_for_qc_grid = win_rel.astype(np.float64)
        qc_grid_duration_s = args.window_seconds
        selected_raw_grid_mask_for_qc = make_grid_mask(
            packet_time_for_qc_grid,
            duration_s=qc_grid_duration_s,
            target_hz=args.target_hz,
        )
        np.savez_compressed(
            outdir / "selected_qc_grid_1000hz_60s.npz",
            time_s=exact_grid.astype(np.float32),
            clean_log_amp=exact_grid_x.astype(np.float32),
            deburst_log_amp=exact_burst_clean.astype(np.float32),
            valid_time_mask=exact_valid.astype(bool),
            raw_valid_mask=selected_raw_grid_mask_for_qc["raw_valid_mask"].astype(bool),
            packet_density_1ms=selected_raw_grid_mask_for_qc["packet_density_1ms"].astype(np.int16),
            age_since_last_packet_s=selected_raw_grid_mask_for_qc["age_since_last_packet_s"].astype(np.float32),
            nearest_packet_index=selected_raw_grid_mask_for_qc["nearest_packet_index"].astype(np.int64),
            burst_time_mask=exact_burst_mask.astype(bool),
            burst_score=exact_burst_score.astype(np.float32),
            burst_z=exact_burst_z.astype(np.float32),
            kept_subcarrier_indices=keep.astype(np.int16),
            removed_subcarrier_indices=bad_all.astype(np.int16),
        )
        save_heatmap_compare(
            exact_grid_x.T,
            exact_burst_clean.T,
            outdir / "02_selected_time_axis_qc_before_after.png",
            f"Selected 60s true-time QC grid before/after deburst ({args.target_hz:g} Hz reference)",
            "Before, x-axis = physical time in selected 60s",
            "After, x-axis = physical time in selected 60s",
        )
        interp_thresholds = []
        if args.final_interp_gap_ms.strip():
            interp_thresholds = [float(x.strip()) for x in args.final_interp_gap_ms.split(",") if x.strip()]
        for gap_ms in interp_thresholds:
            save_interpolated_selected_grid(
                outdir,
                source_pcap=args.pcap,
                selected_pcap=trimmed_pcap,
                time_s=exact_grid.astype(np.float32),
                deburst_log_amp=exact_burst_clean.astype(np.float32),
                valid_mask=exact_valid.astype(bool),
                raw_grid_mask=selected_raw_grid_mask_for_qc,
                burst_mask=exact_burst_mask.astype(bool),
                burst_score=exact_burst_score.astype(np.float32),
                burst_z=exact_burst_z.astype(np.float32),
                keep_subcarriers=keep,
                removed_subcarriers=bad_all,
                max_gap_ms=gap_ms,
            )

    csv_path = outdir / "candidate_windows_tail_scan.csv"
    public_rows = [clean_jsonable(r) for r in candidate_rows]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(public_rows[0].keys()))
        writer.writeheader()
        writer.writerows(public_rows)

    report = {
        "input_pcap": str(args.pcap),
        "output_dir": str(outdir),
        "selected_pcap": str(trimmed_pcap),
        "trimmed_pcap_records_written": int(written_records),
        "total_csi_packets": int(ts_abs.size),
        "total_duration_s": total_duration_s,
        "search_policy": "scan candidate 60s windows from capture tail backward; export first strict pass, otherwise best score",
        "qc_thresholds": {
            "window_seconds": args.window_seconds,
            "exclude_first_seconds": args.exclude_first_seconds,
            "target_hz_qc_only": args.target_hz,
            "max_interp_gap_ms": args.max_interp_gap_ms,
            "max_gap_ms": args.max_gap_ms,
            "max_gaps_gt_500ms": args.max_gaps_gt_500ms,
            "min_coverage": args.min_coverage,
            "max_invalid_gap_ms": args.max_invalid_gap_ms,
            "min_packets": args.min_packets,
            "burst_z_threshold": args.burst_z_threshold,
            "burst_dilate_frames": args.burst_dilate_frames,
            "max_burst_ratio": args.max_burst_ratio,
            "max_burst_run_ms": args.max_burst_run_ms,
        },
        "selected_window": clean_jsonable(selected),
        "removed_known_null_pilot_indices": bad_known.tolist(),
        "removed_auto_static_indices": bad_auto.tolist(),
        "kept_subcarriers": int(keep.size),
        "notes": [
            "raw_full_capture contains full-capture packet timestamps, complex CSI, RSSI, metadata, and raw 1kHz grid masks.",
            "raw_selected_60s contains the same raw packet-level fields restricted to the selected physical 60s interval.",
            "interp_leXXXms folders contain post-selection interpolated 1kHz arrays plus masks for deep-learning use.",
            "The selected packet-level NPZ preserves original irregular CSI packet timestamps within the 60s clip.",
            "The 1kHz grid is saved for QC/reference; it is not a mandatory downsampling step.",
            "If strict_pass is false, the exported clip is the least-bad candidate and should not be treated as clean without review.",
        ],
    }
    with (outdir / "selection_qc_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (outdir / "selection_qc_report.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))
        f.write("\n")

    print(f"Output directory: {outdir}")
    print(f"Selected PCAP: {trimmed_pcap}")
    print(
        f"Selected window: {selected['start_s']:.3f}s -> {selected['end_s']:.3f}s "
        f"strict_pass={selected['strict_pass']} reason={selected['reason']}"
    )
    print(
        f"Packets={selected['packet_count']} avg_rate={selected.get('avg_rate_hz', float('nan')):.2f}Hz "
        f"max_gap={selected.get('max_gap_s', float('nan'))*1000:.1f}ms "
        f"coverage={selected.get('valid_coverage', float('nan'))*100:.2f}% "
        f"burst_ratio={selected.get('burst_ratio', float('nan'))*100:.2f}%"
    )


if __name__ == "__main__":
    main()
