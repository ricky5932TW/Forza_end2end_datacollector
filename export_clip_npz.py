"""Export one aligned clip to a compressed NumPy ``.npz`` file.

Examples:
    python export_clip_npz.py data/sessions/20260520_165701 --clip-id 0
    python export_clip_npz.py data/sessions/20260520_165701 --clip-id 0 --output clip0.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from session_data import ForzaSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one Forza clip as NPZ")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--clip-id", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = ForzaSession(args.session_dir)
    clip = session.load_clip(args.clip_id)
    output = args.output or (args.session_dir / f"clip_{args.clip_id:06d}.npz")

    frame_ids = np.array([int(row["frame_id"]) for row in clip["aligned_rows"]], dtype=np.int64)
    packet_ids = np.array([int(row["packet_id"]) for row in clip["aligned_rows"]], dtype=np.int64)
    audio_samples = np.array([int(row["audio_sample_center"]) for row in clip["aligned_rows"]], dtype=np.int64)

    np.savez_compressed(
        output,
        video_bgr=clip["video_bgr"],
        audio_pcm=clip["audio_pcm"],
        actions=clip["actions"],
        telemetry=clip["telemetry"],
        frame_ids=frame_ids,
        packet_ids=packet_ids,
        audio_samples=audio_samples,
    )
    print(output)
    print("video_bgr", clip["video_bgr"].shape, clip["video_bgr"].dtype)
    print("audio_pcm", clip["audio_pcm"].shape, clip["audio_pcm"].dtype)
    print("actions", clip["actions"].shape, clip["actions"].dtype)
    print("telemetry", clip["telemetry"].shape, clip["telemetry"].dtype)


if __name__ == "__main__":
    main()
