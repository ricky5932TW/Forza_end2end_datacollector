"""Diagnose whether a recorded session looks dropped or just hard to play.

Examples:
    python diagnose_session.py data/sessions/20260520_165701
    python diagnose_session.py --latest
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def latest_session(root: Path) -> Path:
    sessions = [path for path in root.iterdir() if path.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"no session folders in {root}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def load_manifest(session_dir: Path) -> dict[str, Any]:
    with (session_dir / "manifest.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def video_metadata(video_path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {"available": False, "reason": "opencv-python is not installed"}

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return {"available": False, "reason": "OpenCV could not open video"}
        return {
            "available": True,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def diagnose(session_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(session_dir)
    frames = read_csv(session_dir / "frames.csv")
    frame_times = [int(row["t_present_perf_ns"]) for row in frames if row.get("t_present_perf_ns")]
    deltas_ms = [(b - a) / 1_000_000 for a, b in zip(frame_times, frame_times[1:])]
    deltas_sorted = sorted(deltas_ms)

    video_path = session_dir / "video.avi"
    video_size_bytes = video_path.stat().st_size if video_path.exists() else 0
    duration_s = (frame_times[-1] - frame_times[0]) / 1_000_000_000 if len(frame_times) > 1 else 0.0
    fps = (len(frame_times) - 1) / duration_s if duration_s > 0 else 0.0
    mb_per_s = video_size_bytes / 1_000_000 / duration_s if duration_s > 0 else 0.0

    video_summary = manifest.get("summary", {}).get("video", {})
    capture_dropped = int(video_summary.get("dropped", 0) or 0)
    gt_33 = sum(delta > 33.4 for delta in deltas_ms)
    gt_25 = sum(delta > 25.0 for delta in deltas_ms)
    lt_10 = sum(delta < 10.0 for delta in deltas_ms)

    if capture_dropped > 0 or gt_33 > 0:
        verdict = "capture_drop_or_gap"
    elif gt_25 > 0 and lt_10 > 0:
        verdict = "capture_cadence_jitter"
    elif mb_per_s > 15:
        verdict = "large_file_playback_risk"
    else:
        verdict = "looks_ok"

    return {
        "session_dir": str(session_dir),
        "frames": len(frame_times),
        "duration_s": round(duration_s, 3),
        "fps_from_timestamps": round(fps, 3),
        "capture_dropped": capture_dropped,
        "video_size_mb": round(video_size_bytes / 1_000_000, 1),
        "video_write_rate_mb_s": round(mb_per_s, 2),
        "delta_ms": {
            "min": round(min(deltas_ms), 3) if deltas_ms else 0.0,
            "p50": round(statistics.median(deltas_ms), 3) if deltas_ms else 0.0,
            "p95": round(percentile(deltas_sorted, 0.95), 3),
            "p99": round(percentile(deltas_sorted, 0.99), 3),
            "max": round(max(deltas_ms), 3) if deltas_ms else 0.0,
            "gt_25ms": gt_25,
            "gt_33ms": gt_33,
            "lt_10ms": lt_10,
        },
        "video_metadata": video_metadata(video_path),
        "verdict": verdict,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Session: {report['session_dir']}")
    print(f"Frames: {report['frames']}  Duration: {report['duration_s']}s  FPS: {report['fps_from_timestamps']}")
    print(f"Capture dropped: {report['capture_dropped']}")
    print(f"Video size: {report['video_size_mb']} MB  Write/read rate: {report['video_write_rate_mb_s']} MB/s")
    print("Frame delta ms:", report["delta_ms"])
    print("Video metadata:", report["video_metadata"])
    print("Verdict:", report["verdict"])
    if report["verdict"] == "capture_drop_or_gap":
        print("Meaning: recording likely had real frame gaps. Try smaller region, lower fps, or faster codec/disk.")
    elif report["verdict"] == "capture_cadence_jitter":
        print("Meaning: no hard drops, but frame cadence is uneven. Playback may look stuttery even when data exists.")
    elif report["verdict"] == "large_file_playback_risk":
        print("Meaning: capture looks okay; the MJPG file may simply be heavy for the player.")
    else:
        print("Meaning: timing and file size look reasonable.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a Forza capture session")
    parser.add_argument("session_dir", type=Path, nargs="?")
    parser.add_argument("--latest", action="store_true", help="use latest data/sessions folder")
    parser.add_argument("--root", type=Path, default=Path("data/sessions"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = latest_session(args.root) if args.latest else args.session_dir
    if session_dir is None:
        raise SystemExit("pass a session_dir or --latest")
    print_report(diagnose(session_dir))


if __name__ == "__main__":
    main()
