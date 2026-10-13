#!/usr/bin/env python3
"""
Clean and visualize Nexmon CSI exported as UDP/5500 PCAP.

Designed for 80 MHz Nexmon CSI packets whose UDP payload is:
  18 bytes Nexmon metadata + 256 complex int16 CSI samples = 1042 bytes.

Outputs:
  - raw packet-index heatmap
  - raw true-time heatmap with missing slots shown in white
  - cleaned/resampled log-amplitude heatmap
  - burst-suppressed resampled heatmap
  - trainable motion-feature heatmap
  - processed .npz and QC text report

The important design choice is conservative gap handling:
short gaps are interpolated, but long gaps remain masked instead of being
silently hallucinated. Bright burst frames are detected after true-time
resampling and replaced using temporal interpolation.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def read_nexmon_pcap(path: Path, csi_header_bytes: int = 18, n_subcarriers: int = 256):
    """Return timestamps and CSI amplitude matrix with shape [packets, subcarriers]."""
    ts = []
    amps = []
    payload_lens = []

    with path.open("rb") as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            raise ValueError(f"{path} is too small to be a PCAP file")

        magic = global_header[:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            endian = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            endian = ">"
        else:
            raise ValueError(f"Unsupported PCAP magic: {magic!r}")

        pkt_header = struct.Struct(endian + "IIII")
        expected_csi_bytes = n_subcarriers * 2 * 2
        min_payload = csi_header_bytes + expected_csi_bytes

        while True:
            header = f.read(16)
            if len(header) < 16:
                break
            sec, frac, caplen, _origlen = pkt_header.unpack(header)
            pkt = f.read(caplen)
            if len(pkt) < caplen or len(pkt) < 42:
                continue

            # Ethernet II + IPv4 + UDP only.
            if pkt[12:14] != b"\x08\x00":
                continue
            ip_start = 14
            ihl = (pkt[ip_start] & 0x0F) * 4
            if ihl < 20 or pkt[ip_start + 9] != 17:
                continue
            udp_start = ip_start + ihl
            if udp_start + 8 > len(pkt):
                continue
            sport, dport, udp_len, _udp_sum = struct.unpack("!HHHH", pkt[udp_start : udp_start + 8])
            if sport != 5500 and dport != 5500:
                continue

            payload = pkt[udp_start + 8 : udp_start + udp_len]
            payload_lens.append(len(payload))
            if len(payload) < min_payload:
                continue

            raw = payload[csi_header_bytes : csi_header_bytes + expected_csi_bytes]
            vals = np.frombuffer(raw, dtype="<i2")
            if vals.size != n_subcarriers * 2:
                continue
            iq = vals.reshape(n_subcarriers, 2)
            csi = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
            amps.append(np.abs(csi).astype(np.float32))
            ts.append(sec + frac * 1e-6)

    if not amps:
        raise ValueError("No Nexmon UDP/5500 CSI packets were parsed")

    return np.asarray(ts, dtype=np.float64), np.vstack(amps), np.asarray(payload_lens)


def known_80mhz_null_and_pilot_indices(remove_pilots: bool = True):
    # Raw 256 bins are interpreted as shifted indices [-128, ..., 127] -> [0, ..., 255].
    null_bins = [-128, -127, -126, -125, -124, -123, -1, 0, 1, 123, 124, 125, 126, 127]
    pilot_bins = [-103, -75, -39, -11, 11, 39, 75, 103]
    bad = {b + 128 for b in null_bins}
    if remove_pilots:
        bad.update(b + 128 for b in pilot_bins)
    return np.array(sorted(i for i in bad if 0 <= i < 256), dtype=np.int64)


def robust_mad(x: np.ndarray, axis=None):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    return med, 1.4826 * mad


def detect_extreme_static_subcarriers(amp: np.ndarray, already_bad: Iterable[int]):
    """Detect fixed, extreme-amplitude hardware stripes without removing normal CSI bands."""
    already_bad = set(int(i) for i in already_bad)
    mean = np.nanmean(amp, axis=0)
    std = np.nanstd(amp, axis=0)
    cv = std / (mean + 1e-6)

    med = np.median(mean)
    mad = 1.4826 * np.median(np.abs(mean - med)) + 1e-6
    extreme_level = med + 12.0 * mad

    auto = []
    for i, (m, c) in enumerate(zip(mean, cv)):
        if i in already_bad:
            continue
        # Extremely bright and almost time-invariant: typical center/null/hardware stripe.
        if m > extreme_level and c < 0.02:
            auto.append(i)
    return np.array(auto, dtype=np.int64)


def sort_and_deduplicate(ts: np.ndarray, x: np.ndarray):
    order = np.argsort(ts)
    ts = ts[order]
    x = x[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    return ts[keep], x[keep]


def resample_with_gap_mask(ts: np.ndarray, x: np.ndarray, target_hz: float, max_interp_gap_s: float):
    """Linear interpolation on a uniform grid; mark targets inside long source gaps as NaN."""
    ts, x = sort_and_deduplicate(ts, x)
    rel = ts - ts[0]
    step = 1.0 / target_hz
    grid = np.arange(0.0, rel[-1] + 0.5 * step, step, dtype=np.float64)

    pos = np.searchsorted(rel, grid, side="left")
    left = np.clip(pos - 1, 0, len(rel) - 1)
    right = np.clip(pos, 0, len(rel) - 1)
    surrounded = (pos > 0) & (pos < len(rel))
    small_gap = (rel[right] - rel[left]) <= max_interp_gap_s
    valid = surrounded & small_gap
    valid[0] = True
    valid[-1] = True

    out = np.empty((grid.size, x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(grid, rel, x[:, j]).astype(np.float32)
    out[~valid, :] = np.nan
    return grid, out, valid


def moving_nanmean(x: np.ndarray, window: int):
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float32)
    finite = np.isfinite(x).astype(np.float32)
    filled = np.nan_to_num(x, nan=0.0).astype(np.float32)
    out = np.empty_like(filled)
    for j in range(x.shape[1]):
        num = np.convolve(filled[:, j], kernel, mode="same")
        den = np.convolve(finite[:, j], kernel, mode="same")
        out[:, j] = num / np.maximum(den, 1e-6)
    out[finite == 0] = np.nan
    return out


def rolling_nanmedian_1d(x: np.ndarray, window: int):
    """Small, dependency-free rolling median for QC/despiking signals."""
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window)
    return np.nanmedian(view, axis=1).astype(np.float32)


def interpolate_bad_time_rows(x: np.ndarray, bad_rows: np.ndarray, time_axis: np.ndarray | None = None):
    """Replace bad time rows using linear interpolation along time for each subcarrier."""
    if not np.any(bad_rows):
        return x.copy()
    out = x.copy()
    if time_axis is None:
        t = np.arange(x.shape[0], dtype=np.float32)
    else:
        t = time_axis.astype(np.float32)
    good_rows = (~bad_rows) & np.any(np.isfinite(x), axis=1)
    if np.sum(good_rows) < 2:
        return out
    for j in range(x.shape[1]):
        finite_good = good_rows & np.isfinite(x[:, j])
        if np.sum(finite_good) < 2:
            continue
        out[bad_rows, j] = np.interp(t[bad_rows], t[finite_good], x[finite_good, j]).astype(np.float32)
    return out


def suppress_frame_bursts(
    log_amp_resampled: np.ndarray,
    target_hz: float,
    window_seconds: float,
    z_threshold: float,
    dilate_frames: int,
    time_axis: np.ndarray | None = None,
):
    """Detect and replace column-wise gain bursts after true-time resampling.

    Bright vertical stripes are treated as packet/frame-level artifacts when the
    robust per-frame amplitude score jumps far above its local baseline.
    """
    x = log_amp_resampled.astype(np.float32)
    valid_time = np.any(np.isfinite(x), axis=1)

    frame_med = np.nanmedian(x, axis=1)
    frame_p90 = np.nanpercentile(x, 90, axis=1)
    # p90 is sensitive to column-wise bright bursts; median keeps it robust.
    score = 0.5 * frame_med + 0.5 * frame_p90

    window = max(5, int(round(window_seconds * target_hz)))
    if window % 2 == 0:
        window += 1
    baseline = rolling_nanmedian_1d(score, window)
    resid = score - baseline

    local_abs = rolling_nanmedian_1d(np.abs(resid), window)
    global_abs = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    scale = 1.4826 * np.maximum(local_abs, global_abs + 1e-6)
    z = resid / np.maximum(scale, 1e-6)

    bad = valid_time & (z > z_threshold)
    if dilate_frames > 0 and np.any(bad):
        kernel = np.ones(2 * dilate_frames + 1, dtype=np.int16)
        bad = np.convolve(bad.astype(np.int16), kernel, mode="same") > 0
        bad &= valid_time

    cleaned = interpolate_bad_time_rows(x, bad, time_axis=time_axis)
    return cleaned, bad.astype(bool), score.astype(np.float32), z.astype(np.float32)


def make_trainable_feature(log_amp_resampled: np.ndarray, highpass_seconds: float, target_hz: float):
    # Suppress packet-level/global gain first.
    frame_med = np.nanmedian(log_amp_resampled, axis=1, keepdims=True)
    x = log_amp_resampled - frame_med

    # Remove static subcarrier profile.
    sub_med = np.nanmedian(x, axis=0, keepdims=True)
    x = x - sub_med

    # Remove slow drift, keep motion-scale fluctuations.
    window = max(3, int(round(highpass_seconds * target_hz)))
    if window % 2 == 0:
        window += 1
    baseline = moving_nanmean(x, window)
    x = x - baseline

    # Per-subcarrier robust z normalization after valid-window processing.
    med, scale = robust_mad(x, axis=0)
    z = (x - med) / np.maximum(scale, 1e-3)
    return np.clip(z, -4.0, 4.0).astype(np.float32)


def remove_top_pca_components(feature: np.ndarray, n_components: int):
    """Remove dominant low-rank components; NaN/missing slots are preserved."""
    if n_components <= 0:
        return feature
    valid = np.isfinite(feature)
    x = np.nan_to_num(feature, nan=0.0).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(n_components, vt.shape[0])
    cleaned = x - (u[:, :k] * s[:k]) @ vt[:k, :]
    cleaned[~valid] = np.nan
    return np.clip(cleaned, -4.0, 4.0).astype(np.float32)


def valid_segments_from_mask(valid_mask: np.ndarray, time_s: np.ndarray, min_duration_s: float):
    """Return continuous valid intervals as [start_idx, end_idx, start_s, end_s]."""
    valid = np.asarray(valid_mask, dtype=bool)
    segments = []
    i = 0
    while i < valid.size:
        while i < valid.size and not valid[i]:
            i += 1
        if i >= valid.size:
            break
        start = i
        while i < valid.size and valid[i]:
            i += 1
        end = i
        duration = float(time_s[end - 1] - time_s[start]) if end > start else 0.0
        if duration >= min_duration_s:
            segments.append((start, end, float(time_s[start]), float(time_s[end - 1])))
    return segments


def concatenate_valid_segments(x: np.ndarray, segments, separator_frames: int = 5):
    """Concatenate valid segments with NaN separators for compact visualization/training QA."""
    if not segments:
        return x[:0].copy()
    parts = []
    sep = np.full((separator_frames, x.shape[1]), np.nan, dtype=np.float32)
    for idx, (start, end, _start_s, _end_s) in enumerate(segments):
        if idx:
            parts.append(sep)
        parts.append(x[start:end])
    return np.vstack(parts).astype(np.float32)


def downsample_for_image(arr_s_by_t: np.ndarray, max_width: int = 1800):
    h, w = arr_s_by_t.shape
    if w <= max_width:
        return arr_s_by_t
    edges = np.linspace(0, w, max_width + 1).astype(int)
    out = np.empty((h, max_width), dtype=np.float32)
    for i in range(max_width):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        out[:, i] = np.nanmean(arr_s_by_t[:, a:b], axis=1)
    return out


def turbo_colormap(v: np.ndarray):
    # Compact polynomial approximation of Google's Turbo colormap.
    v = np.clip(v, 0.0, 1.0)
    r = 0.13572138 + 4.61539260*v - 42.66032258*v**2 + 132.13108234*v**3 - 152.94239396*v**4 + 59.28637943*v**5
    g = 0.09140261 + 2.19418839*v + 4.84296658*v**2 - 14.18503333*v**3 + 4.27729857*v**4 + 2.82956604*v**5
    b = 0.10667330 + 12.64194608*v - 60.58204836*v**2 + 110.36276771*v**3 - 89.90310912*v**4 + 27.34824973*v**5
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def save_heatmap(arr_s_by_t: np.ndarray, path: Path, title: str, robust: bool = True, max_width: int = 1800):
    data = downsample_for_image(arr_s_by_t.astype(np.float32), max_width=max_width)
    finite = np.isfinite(data)
    if finite.any():
        if robust:
            vmin, vmax = np.nanpercentile(data, [2, 98])
        else:
            vmin, vmax = np.nanmin(data), np.nanmax(data)
        if not np.isfinite(vmax - vmin) or abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        norm = (data - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(data)
        vmin, vmax = 0.0, 1.0

    rgb = turbo_colormap(norm)
    rgb[~finite] = np.array([255, 255, 255], dtype=np.uint8)
    # Flip vertically so subcarrier index increases upward, matching common CSI heatmaps.
    rgb = rgb[::-1, :, :]
    img = Image.fromarray(rgb, mode="RGB").resize((rgb.shape[1], max(360, rgb.shape[0] * 3)))

    top = 50
    canvas = Image.new("RGB", (img.width, img.height + top), "white")
    canvas.paste(img, (0, top))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"{title}  |  color range [{vmin:.3g}, {vmax:.3g}], white = missing", fill=(0, 0, 0))
    canvas.save(path)


def heatmap_rgb_with_range(arr_s_by_t: np.ndarray, vmin: float, vmax: float, max_width: int = 900):
    data = downsample_for_image(arr_s_by_t.astype(np.float32), max_width=max_width)
    finite = np.isfinite(data)
    if not np.isfinite(vmax - vmin) or abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    norm = (data - vmin) / (vmax - vmin)
    rgb = turbo_colormap(norm)
    rgb[~finite] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb[::-1, :, :]


def save_heatmap_compare(
    before_s_by_t: np.ndarray,
    after_s_by_t: np.ndarray,
    path: Path,
    title: str,
    before_label: str,
    after_label: str,
    robust: bool = True,
    panel_width: int = 900,
):
    """Save before/after heatmaps using one shared color range."""
    before_ds = downsample_for_image(before_s_by_t.astype(np.float32), max_width=panel_width)
    after_ds = downsample_for_image(after_s_by_t.astype(np.float32), max_width=panel_width)
    combined = np.concatenate([before_ds.ravel(), after_ds.ravel()])
    combined = combined[np.isfinite(combined)]
    if combined.size:
        if robust:
            vmin, vmax = np.percentile(combined, [2, 98])
        else:
            vmin, vmax = np.min(combined), np.max(combined)
    else:
        vmin, vmax = 0.0, 1.0

    before_rgb = heatmap_rgb_with_range(before_s_by_t, vmin, vmax, max_width=panel_width)
    after_rgb = heatmap_rgb_with_range(after_s_by_t, vmin, vmax, max_width=panel_width)
    height = max(360, before_rgb.shape[0] * 3)
    before_img = Image.fromarray(before_rgb, mode="RGB").resize((before_rgb.shape[1], height))
    after_img = Image.fromarray(after_rgb, mode="RGB").resize((after_rgb.shape[1], height))

    gap = 20
    top = 70
    canvas = Image.new("RGB", (before_img.width + after_img.width + gap, height + top), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{title}  |  shared color range [{vmin:.3g}, {vmax:.3g}], white = missing", fill=(0, 0, 0))
    draw.text((12, 42), before_label, fill=(0, 0, 0))
    draw.text((before_img.width + gap + 12, 42), after_label, fill=(0, 0, 0))
    canvas.paste(before_img, (0, top))
    canvas.paste(after_img, (before_img.width + gap, top))
    canvas.save(path)


def summarize_qc(ts: np.ndarray, payload_lens: np.ndarray, valid_grid: np.ndarray, target_hz: float):
    dt = np.diff(ts)
    sec = np.floor(ts - ts[0]).astype(int)
    counts = np.bincount(sec)
    unique_lens, len_counts = np.unique(payload_lens, return_counts=True)
    return {
        "packets": int(len(ts)),
        "duration_s": float(ts[-1] - ts[0]),
        "avg_raw_rate_hz": float(len(ts) / (ts[-1] - ts[0])),
        "payload_lengths": {int(k): int(v) for k, v in zip(unique_lens, len_counts)},
        "dt_p50_s": float(np.percentile(dt, 50)),
        "dt_p95_s": float(np.percentile(dt, 95)),
        "dt_p99_s": float(np.percentile(dt, 99)),
        "dt_p99_5_s": float(np.percentile(dt, 99.5)),
        "dt_max_s": float(np.max(dt)),
        "gaps_gt_20ms": int(np.sum(dt > 0.020)),
        "gaps_gt_50ms": int(np.sum(dt > 0.050)),
        "gaps_gt_100ms": int(np.sum(dt > 0.100)),
        "gaps_gt_500ms": int(np.sum(dt > 0.500)),
        "per_second_count_min": int(np.min(counts)),
        "per_second_count_median": float(np.median(counts)),
        "per_second_count_mean": float(np.mean(counts)),
        "per_second_count_p95": float(np.percentile(counts, 95)),
        "per_second_count_max": int(np.max(counts)),
        "resample_target_hz": float(target_hz),
        "resample_valid_coverage": float(np.mean(valid_grid)),
    }


def summarize_packet_qc(ts: np.ndarray, payload_lens: np.ndarray):
    dt = np.diff(ts)
    sec = np.floor(ts - ts[0]).astype(int)
    counts = np.bincount(sec)
    unique_lens, len_counts = np.unique(payload_lens, return_counts=True)
    return {
        "packets": int(len(ts)),
        "duration_s": float(ts[-1] - ts[0]),
        "avg_raw_rate_hz": float(len(ts) / (ts[-1] - ts[0])),
        "payload_lengths": {int(k): int(v) for k, v in zip(unique_lens, len_counts)},
        "dt_p50_s": float(np.percentile(dt, 50)),
        "dt_p95_s": float(np.percentile(dt, 95)),
        "dt_p99_s": float(np.percentile(dt, 99)),
        "dt_p99_5_s": float(np.percentile(dt, 99.5)),
        "dt_max_s": float(np.max(dt)),
        "gaps_gt_20ms": int(np.sum(dt > 0.020)),
        "gaps_gt_50ms": int(np.sum(dt > 0.050)),
        "gaps_gt_100ms": int(np.sum(dt > 0.100)),
        "gaps_gt_500ms": int(np.sum(dt > 0.500)),
        "per_second_count_min": int(np.min(counts)),
        "per_second_count_median": float(np.median(counts)),
        "per_second_count_mean": float(np.mean(counts)),
        "per_second_count_p95": float(np.percentile(counts, 95)),
        "per_second_count_max": int(np.max(counts)),
    }

def crop_secs(x,segments,secs):
    segments = segments[::-1]
    dur = 0
    i=0
    for start, end, start_s, end_s in segments:
        dur = dur + end_s - start_s
        i=i+1
        if dur >secs:
            break
    x = x[segments[i][0]:segments[0][1]]
    print(x.shape)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--target-hz", type=float, default=100.0)
    parser.add_argument("--max-interp-gap-ms", type=float, default=120.0)
    parser.add_argument("--highpass-seconds", type=float, default=1.0)
    parser.add_argument("--burst-window-seconds", type=float, default=1.0)
    parser.add_argument("--burst-z-threshold", type=float, default=6.0)
    parser.add_argument("--burst-dilate-frames", type=int, default=1)
    parser.add_argument("--disable-burst-suppression", action="store_true")
    parser.add_argument("--remove-pca-components", type=int, default=2)
    parser.add_argument("--min-usable-segment-seconds", type=float, default=2.0)
    parser.add_argument("--save-packet-level-sidecar", action="store_true", default=True)
    parser.add_argument("--keep-pilots", action="store_true", help="Do not remove 80MHz pilot subcarriers")
    args = parser.parse_args()

    outdir = args.outdir or (args.pcap.parent / (args.pcap.stem + "_clean_outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    ts, amp, payload_lens = read_nexmon_pcap(args.pcap)
    ts, amp = sort_and_deduplicate(ts, amp)

    bad_known = known_80mhz_null_and_pilot_indices(remove_pilots=not args.keep_pilots)
    bad_auto = detect_extreme_static_subcarriers(amp, already_bad=bad_known)
    bad_all = np.array(sorted(set(bad_known.tolist()) | set(bad_auto.tolist())), dtype=np.int64)
    keep = np.array([i for i in range(amp.shape[1]) if i not in set(bad_all.tolist())], dtype=np.int64)

    raw_log = np.log1p(amp)
    rel_ts = ts - ts[0]
    grid, raw_grid, raw_valid = resample_with_gap_mask(
        ts, raw_log, target_hz=args.target_hz, max_interp_gap_s=args.max_interp_gap_ms / 1000.0
    )

    clean_log = raw_log[:, keep]
    if args.save_packet_level_sidecar:
        effective_packet_hz = len(ts) / max(rel_ts[-1], 1e-6)
        if args.disable_burst_suppression:
            packet_burst_clean_log = clean_log
            packet_burst_mask = np.zeros(clean_log.shape[0], dtype=bool)
            packet_burst_score = np.full(clean_log.shape[0], np.nan, dtype=np.float32)
            packet_burst_z = np.full(clean_log.shape[0], np.nan, dtype=np.float32)
        else:
            packet_burst_clean_log, packet_burst_mask, packet_burst_score, packet_burst_z = suppress_frame_bursts(
                clean_log,
                target_hz=effective_packet_hz,
                window_seconds=args.burst_window_seconds,
                z_threshold=args.burst_z_threshold,
                dilate_frames=args.burst_dilate_frames,
                time_axis=rel_ts,
            )
        np.savez_compressed(
            outdir / "processed_csi_packet_level_deburst_sidecar.npz",
            packet_time_s=rel_ts.astype(np.float32),
            kept_subcarrier_indices=keep.astype(np.int16),
            removed_subcarrier_indices=bad_all.astype(np.int16),
            packet_clean_log_amp=clean_log.astype(np.float32),
            packet_burst_suppressed_log_amp=packet_burst_clean_log.astype(np.float32),
            packet_burst_mask=packet_burst_mask.astype(bool),
            packet_burst_score=packet_burst_score.astype(np.float32),
            packet_burst_z=packet_burst_z.astype(np.float32),
        )

    grid, clean_grid, valid_grid = resample_with_gap_mask(
        ts, clean_log, target_hz=args.target_hz, max_interp_gap_s=args.max_interp_gap_ms / 1000.0
    )

    if args.disable_burst_suppression:
        burst_clean_grid = clean_grid
        burst_mask = np.zeros(clean_grid.shape[0], dtype=bool)
        burst_score = np.full(clean_grid.shape[0], np.nan, dtype=np.float32)
        burst_z = np.full(clean_grid.shape[0], np.nan, dtype=np.float32)
    else:
        burst_clean_grid, burst_mask, burst_score, burst_z = suppress_frame_bursts(
            clean_grid,
            target_hz=args.target_hz,
            window_seconds=args.burst_window_seconds,
            z_threshold=args.burst_z_threshold,
            dilate_frames=args.burst_dilate_frames,
        )
    feature = make_trainable_feature(burst_clean_grid, args.highpass_seconds, args.target_hz)
    feature_pca = remove_top_pca_components(feature, args.remove_pca_components)
    usable_segments = valid_segments_from_mask(valid_grid, grid, args.min_usable_segment_seconds)
    usable_segments_arr = np.asarray(usable_segments, dtype=np.float32) if usable_segments else np.empty((0, 4), dtype=np.float32)

    #burst_clean_grid = crop_secs(burst_clean_grid, usable_segments,60)

    if args.save_packet_level_sidecar:
        save_heatmap_compare(
            clean_log.T,
            packet_burst_clean_log.T,
            outdir / "01_packet_axis_before_after_logamp.png",
            "Packet-axis log amplitude before/after deburst",
            "Before deburst, x-axis = CSI packet index",
            "After deburst, x-axis = CSI packet index",
        )
    save_heatmap_compare(
        clean_grid.T,
        burst_clean_grid.T,
        outdir / "02_time_axis_before_after_logamp.png",
        f"Time-axis log amplitude before/after deburst ({args.target_hz:g} Hz grid)",
        "Before deburst, x-axis = elapsed time grid",
        "After deburst, x-axis = elapsed time grid",
    )
    if args.remove_pca_components > 0:
        compact_feature_pca = concatenate_valid_segments(feature_pca, usable_segments)
        save_heatmap(
            compact_feature_pca.T,
            outdir / f"03_motion_feature_pca{args.remove_pca_components}_valid_segments_only.png",
            f"Trainable motion feature PCA top-{args.remove_pca_components}, valid segments only",
            robust=False,
        )

    qc = summarize_qc(ts, payload_lens, valid_grid, args.target_hz)
    qc["removed_known_null_pilot_indices"] = bad_known.tolist()
    qc["removed_auto_static_indices"] = bad_auto.tolist()
    qc["kept_subcarriers"] = int(keep.size)
    qc["burst_suppression_enabled"] = not args.disable_burst_suppression
    qc["burst_frames_detected"] = int(np.sum(burst_mask))
    qc["burst_frame_ratio"] = float(np.mean(burst_mask))
    qc["burst_z_threshold"] = float(args.burst_z_threshold)
    qc["usable_segments_min_seconds"] = float(args.min_usable_segment_seconds)
    qc["usable_segments_count"] = int(len(usable_segments))
    qc["usable_segments_total_seconds"] = float(sum(end_s - start_s for _start, _end, start_s, end_s in usable_segments))
    qc["packet_level_sidecar_saved"] = bool(args.save_packet_level_sidecar)

    report = outdir / "qc_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"Input: {args.pcap}\n")
        for k, v in qc.items():
            f.write(f"{k}: {v}\n")
        f.write("\nInterpretation:\n")
        f.write("- Horizontal raw stripes are expected to be dominated by null/pilot/static hardware bins and static channel profile.\n")
        f.write("- Bright vertical burst stripes are detected from robust per-frame amplitude scores after true-time resampling and replaced by temporal interpolation.\n")
        f.write("- The trainable feature removes packet-level gain, static subcarrier profile, and slow drift.\n")
        f.write("- White columns after resampling are real timestamp gaps longer than max_interp_gap; do not fill them blindly for training.\n")
        f.write("- For model training, cut windows only from usable_segments_s.csv intervals, or keep valid_time_mask as a mask channel.\n")

    segments_csv = outdir / "usable_segments_s.csv"
    with segments_csv.open("w", encoding="utf-8") as f:
        f.write("start_idx,end_idx,start_s,end_s,duration_s\n")
        for start, end, start_s, end_s in usable_segments:
            f.write(f"{start},{end},{start_s:.6f},{end_s:.6f},{end_s - start_s:.6f}\n")

    np.savez_compressed(
        outdir / f"processed_csi_{int(round(args.target_hz))}hz.npz",
        time_s=grid.astype(np.float32),
        kept_subcarrier_indices=keep.astype(np.int16),
        removed_subcarrier_indices=bad_all.astype(np.int16),
        clean_log_amp=clean_grid.astype(np.float32),
        burst_suppressed_log_amp=burst_clean_grid.astype(np.float32),
        motion_feature=feature.astype(np.float32),
        motion_feature_pca=feature_pca.astype(np.float32),
        burst_time_mask=burst_mask.astype(bool),
        burst_score=burst_score.astype(np.float32),
        burst_z=burst_z.astype(np.float32),
        valid_time_mask=valid_grid.astype(bool),
        usable_segments_s=usable_segments_arr,
    )

    print(f"Output directory: {outdir}")
    print(f"Packets: {qc['packets']}  duration={qc['duration_s']:.3f}s  avg_raw_rate={qc['avg_raw_rate_hz']:.2f}Hz")
    print(f"Raw dt p50={qc['dt_p50_s']*1000:.2f}ms p99={qc['dt_p99_s']*1000:.2f}ms max={qc['dt_max_s']*1000:.2f}ms")
    print(f"Gaps >100ms: {qc['gaps_gt_100ms']}  >500ms: {qc['gaps_gt_500ms']}")
    print(f"Removed known null/pilot: {len(bad_known)}  auto-static: {bad_auto.tolist()}  kept={keep.size}")
    print(f"Burst frames suppressed: {int(np.sum(burst_mask))} ({np.mean(burst_mask)*100:.2f}%)")
    print(f"Resampling coverage at {args.target_hz:g}Hz: {qc['resample_valid_coverage']*100:.2f}%")
    print(f"Usable segments >= {args.min_usable_segment_seconds:g}s: {len(usable_segments)}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
