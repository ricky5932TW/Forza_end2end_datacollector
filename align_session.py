"""Create frame-level and clip-level alignment files for a recorded session.

Examples:
    python align_session.py data/sessions/20260520_170000
    python align_session.py data/sessions/20260520_170000 --clip-frames 120 --stride-frames 60

Alignment is intentionally simple in v1: each video frame gets the nearest UDP
packet and the audio sample index predicted from the first audio block.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Any


TELEMETRY_COLUMNS = [
    "TimestampMS",
    "Speed",
    "CurrentEngineRpm",
    "Gear",
    "Accel",
    "Brake",
    "Steer",
    "AccelerationX",
    "AccelerationY",
    "AccelerationZ",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "AngularVelocityX",
    "AngularVelocityY",
    "AngularVelocityZ",
    "Yaw",
    "Pitch",
    "Roll",
    "NormalizedSuspensionTravelFrontLeft",
    "NormalizedSuspensionTravelFrontRight",
    "NormalizedSuspensionTravelRearLeft",
    "NormalizedSuspensionTravelRearRight",
    "TireSlipRatioFrontLeft",
    "TireSlipRatioFrontRight",
    "TireSlipRatioRearLeft",
    "TireSlipRatioRearRight",
    "WheelRotationSpeedFrontLeft",
    "WheelRotationSpeedFrontRight",
    "WheelRotationSpeedRearLeft",
    "WheelRotationSpeedRearRight",
    "Power",
    "Torque",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a small sidecar CSV into dictionaries."""

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def int_or_none(value: str) -> int | None:
    """Convert a CSV integer cell while preserving empty cells as ``None``."""

    if value == "":
        return None
    return int(value)


def load_manifest(session_dir: Path) -> dict[str, Any]:
    with (session_dir / "manifest.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def nearest_packet(packet_times: list[int], frame_time: int) -> int | None:
    """Return the index of the packet timestamp closest to a frame time."""

    if not packet_times:
        return None
    idx = bisect.bisect_left(packet_times, frame_time)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(packet_times):
        candidates.append(idx)
    return min(candidates, key=lambda item: abs(packet_times[item] - frame_time))


def first_audio_anchor(audio_blocks: list[dict[str, str]]) -> tuple[int, int] | None:
    """Return the first usable ``(sample_index, perf_time_ns)`` audio anchor."""

    for block in audio_blocks:
        sample = int_or_none(block.get("first_sample", ""))
        timestamp = int_or_none(block.get("t_audio_start_perf_ns", ""))
        if sample is not None and timestamp is not None:
            return sample, timestamp
    return None


def sample_for_time(frame_time_ns: int, anchor_sample: int, anchor_time_ns: int, sample_rate: int) -> int:
    """Map a perf timestamp to an audio sample index.

    Example:
        sample_for_time(1_500_000_000, 0, 1_000_000_000, 48_000) == 24_000
    """

    delta_ns = frame_time_ns - anchor_time_ns
    return anchor_sample + round(delta_ns * sample_rate / 1_000_000_000)


def write_aligned_frames(session_dir: Path, max_packet_gap_ms: float) -> list[dict[str, Any]]:
    """Write ``aligned_frames.csv`` and return rows for clip indexing."""

    manifest = load_manifest(session_dir)
    sample_rate = int(manifest.get("summary", {}).get("audio", {}).get("sample_rate", 0))
    frames = read_csv(session_dir / "frames.csv")
    packets = read_csv(session_dir / "packets.csv")
    audio_blocks = read_csv(session_dir / "audio_blocks.csv")

    packet_rows = [row for row in packets if row.get("t_recv_perf_ns")]
    packet_times = [int(row["t_recv_perf_ns"]) for row in packet_rows]
    anchor = first_audio_anchor(audio_blocks)
    aligned: list[dict[str, Any]] = []

    output_fields = [
        "frame_id",
        "t_present_perf_ns",
        "audio_sample_center",
        "packet_id",
        "packet_dt_ms",
        "is_valid",
        *TELEMETRY_COLUMNS,
    ]
    with (session_dir / "aligned_frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        for frame in frames:
            frame_time = int(frame["t_present_perf_ns"])
            packet_idx = nearest_packet(packet_times, frame_time)
            packet_row = packet_rows[packet_idx] if packet_idx is not None else {}
            packet_dt_ms = ""
            packet_ok = False
            if packet_idx is not None:
                packet_dt_ms_float = (int(packet_row["t_recv_perf_ns"]) - frame_time) / 1_000_000
                packet_dt_ms = f"{packet_dt_ms_float:.3f}"
                # A frame is valid only when the nearest parsed packet is close
                # enough for the current offline analysis tolerance.
                packet_ok = abs(packet_dt_ms_float) <= max_packet_gap_ms and not packet_row.get("parse_error")

            audio_sample = ""
            audio_ok = False
            if anchor is not None and sample_rate > 0:
                audio_sample_int = sample_for_time(frame_time, anchor[0], anchor[1], sample_rate)
                audio_sample = str(audio_sample_int)
                audio_ok = audio_sample_int >= 0

            row: dict[str, Any] = {
                "frame_id": frame["frame_id"],
                "t_present_perf_ns": frame["t_present_perf_ns"],
                "audio_sample_center": audio_sample,
                "packet_id": packet_row.get("packet_id", ""),
                "packet_dt_ms": packet_dt_ms,
                "is_valid": int(packet_ok and audio_ok),
            }
            for column in TELEMETRY_COLUMNS:
                row[column] = packet_row.get(column, "")
            writer.writerow(row)
            aligned.append(row)
    return aligned


def write_clips(session_dir: Path, aligned: list[dict[str, Any]], clip_frames: int, stride_frames: int) -> None:
    """Write fixed-length clip windows from aligned frame rows."""

    fields = [
        "clip_id",
        "start_frame_id",
        "end_frame_id",
        "start_t_perf_ns",
        "end_t_perf_ns",
        "start_audio_sample",
        "end_audio_sample",
        "valid_frame_count",
        "is_valid",
    ]
    with (session_dir / "clips.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        clip_id = 0
        for start in range(0, max(0, len(aligned) - clip_frames + 1), stride_frames):
            clip = aligned[start : start + clip_frames]
            valid_count = sum(1 for row in clip if str(row["is_valid"]) == "1")
            writer.writerow(
                {
                    "clip_id": clip_id,
                    "start_frame_id": clip[0]["frame_id"],
                    "end_frame_id": clip[-1]["frame_id"],
                    "start_t_perf_ns": clip[0]["t_present_perf_ns"],
                    "end_t_perf_ns": clip[-1]["t_present_perf_ns"],
                    "start_audio_sample": clip[0]["audio_sample_center"],
                    "end_audio_sample": clip[-1]["audio_sample_center"],
                    "valid_frame_count": valid_count,
                    "is_valid": int(valid_count == clip_frames),
                }
            )
            clip_id += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a recorded Forza session")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--max-packet-gap-ms", type=float, default=25.0)
    parser.add_argument("--clip-frames", type=int, default=120)
    parser.add_argument("--stride-frames", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aligned = write_aligned_frames(args.session_dir, args.max_packet_gap_ms)
    write_clips(args.session_dir, aligned, args.clip_frames, args.stride_frames)
    print(args.session_dir / "aligned_frames.csv")
    print(args.session_dir / "clips.csv")


if __name__ == "__main__":
    main()
