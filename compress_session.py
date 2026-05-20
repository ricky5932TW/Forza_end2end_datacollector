"""Compress a recorded session video after capture.

Examples:
    python compress_session.py data/sessions/20260520_165701
    python compress_session.py --latest --codec libx264 --crf 24

This script never edits the source ``video.avi``. It creates a preview MP4 next
to the raw recording.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def latest_session(root: Path) -> Path:
    sessions = [path for path in root.iterdir() if path.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"no session folders in {root}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    env_path = Path(sys.executable).resolve().parent / "Library" / "bin" / "ffmpeg.exe"
    if env_path.exists():
        return str(env_path)
    raise FileNotFoundError(
        "ffmpeg is not in PATH. Install it with winget install Gyan.FFmpeg "
        "or conda install -n forza -c conda-forge ffmpeg."
    )


def build_command(
    ffmpeg: str,
    session_dir: Path,
    output_path: Path,
    codec: str,
    crf: int,
    preset: str,
    audio: bool,
) -> list[str]:
    video_path = session_dir / "video.avi"
    audio_path = session_dir / "audio.wav"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    command = [ffmpeg, "-y", "-i", str(video_path)]
    include_audio = audio and audio_path.exists()
    if include_audio:
        command += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"]

    if codec.endswith("_nvenc"):
        command += ["-c:v", codec, "-preset", preset, "-cq:v", str(crf), "-b:v", "0"]
    else:
        command += ["-c:v", codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]

    if include_audio:
        command += ["-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-an"]
    command += [str(output_path)]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress session video with ffmpeg")
    parser.add_argument("session_dir", type=Path, nargs="?")
    parser.add_argument("--latest", action="store_true", help="use latest data/sessions folder")
    parser.add_argument("--root", type=Path, default=Path("data/sessions"))
    parser.add_argument("--codec", default="libx264", help="libx264, libx265, h264_nvenc, or hevc_nvenc")
    parser.add_argument("--crf", type=int, default=23, help="quality value; lower is larger/better")
    parser.add_argument("--preset", default="veryfast", help="CPU preset or NVENC preset such as p4")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = latest_session(args.root) if args.latest else args.session_dir
    if session_dir is None:
        raise SystemExit("pass a session_dir or --latest")

    output = args.output or (session_dir / f"video_{args.codec}.mp4")
    command = build_command(
        ffmpeg_path(),
        session_dir,
        output,
        args.codec,
        args.crf,
        args.preset,
        not args.no_audio,
    )
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
