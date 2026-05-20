"""Convenience loader for recorded Forza sessions.

Examples:
    from session_data import ForzaSession

    session = ForzaSession("data/sessions/20260520_165701")
    clip = session.load_clip(0, valid_only=True)
    print(clip["video_bgr"].shape, clip["audio_pcm"].shape)
    print(clip["actions"][:3])
"""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_COLUMNS = ("Steer", "Accel", "Brake")
DEFAULT_TELEMETRY_COLUMNS = (
    "TimestampMS",
    "Speed",
    "CurrentEngineRpm",
    "Gear",
    "Accel",
    "Brake",
    "Steer",
    "IsRaceOn",
    "PositionX",
    "PositionY",
    "PositionZ",
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
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(value: str, default: float = 0.0) -> float:
    return default if value == "" else float(value)


def as_int(value: str, default: int = 0) -> int:
    return default if value == "" else int(float(value))


def is_valid_row(row: dict[str, Any]) -> bool:
    """Return whether a CSV row's ``is_valid`` flag is truthy."""

    value = str(row.get("is_valid", "")).strip().lower()
    if value in {"true", "yes"}:
        return True
    try:
        return as_int(value) == 1
    except ValueError:
        return False


class ForzaSession:
    """Read one session folder produced by ``capture_session.py``."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        if not self.session_dir.exists():
            raise FileNotFoundError(self.session_dir)

    def manifest(self) -> dict[str, Any]:
        """Load ``manifest.json``."""

        with (self.session_dir / "manifest.json").open("r", encoding="utf-8") as file:
            return json.load(file)

    def frames(self) -> list[dict[str, str]]:
        """Load ``frames.csv`` rows."""

        return read_csv_rows(self.session_dir / "frames.csv")

    def packets(self) -> list[dict[str, str]]:
        """Load ``packets.csv`` rows."""

        return read_csv_rows(self.session_dir / "packets.csv")

    def aligned_frames(self, valid_only: bool = False) -> list[dict[str, str]]:
        """Load ``aligned_frames.csv`` rows."""

        rows = read_csv_rows(self.session_dir / "aligned_frames.csv")
        return [row for row in rows if is_valid_row(row)] if valid_only else rows

    def clips(self, valid_only: bool = False) -> list[dict[str, str]]:
        """Load ``clips.csv`` rows."""

        rows = read_csv_rows(self.session_dir / "clips.csv")
        return [row for row in rows if is_valid_row(row)] if valid_only else rows

    def telemetry_records(self) -> Iterable[dict[str, Any]]:
        """Yield parsed full telemetry records from ``telemetry.jsonl``."""

        with (self.session_dir / "telemetry.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)

    def raw_packet(self, packet_id: int) -> bytes:
        """Recover one UDP payload from ``udp_payloads.bin`` by packet id."""

        packets = self.packets()
        row = packets[packet_id]
        offset = as_int(row["payload_offset"])
        size = as_int(row["packet_size"])
        with (self.session_dir / "udp_payloads.bin").open("rb") as file:
            file.seek(offset)
            return file.read(size)

    def video_frame_index(self) -> dict[int, int]:
        """Map recorded ``frame_id`` values to positions inside ``video.avi``."""

        return {as_int(row["frame_id"]): index for index, row in enumerate(self.frames())}

    def _aligned_rows_for_frame_span(self, start_frame: int, end_frame: int) -> list[dict[str, str]]:
        aligned = self.aligned_frames()
        aligned_index = {as_int(row["frame_id"]): index for index, row in enumerate(aligned)}
        if start_frame not in aligned_index:
            raise KeyError(f"missing aligned start frame row: {start_frame}")
        if end_frame not in aligned_index:
            raise KeyError(f"missing aligned end frame row: {end_frame}")

        start_index = aligned_index[start_frame]
        end_index = aligned_index[end_frame]
        if end_index < start_index:
            raise ValueError(f"end frame {end_frame} appears before start frame {start_frame}")
        return aligned[start_index : end_index + 1]

    def load_video_frames(self, start_frame: int, end_frame: int) -> np.ndarray:
        """Load inclusive recorded frame-id span as BGR uint8 array: ``T,H,W,3``."""

        import cv2

        frame_index = self.video_frame_index()
        if start_frame not in frame_index:
            raise KeyError(f"missing video start frame row: {start_frame}")
        if end_frame not in frame_index:
            raise KeyError(f"missing video end frame row: {end_frame}")
        start_video_index = frame_index[start_frame]
        end_video_index = frame_index[end_frame]
        if end_video_index < start_video_index:
            raise ValueError(f"end frame {end_frame} appears before start frame {start_frame}")

        video_path = self.session_dir / "video.avi"
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"could not open {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_video_index)
            frames = []
            for _frame_id in range(start_video_index, end_video_index + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                raise RuntimeError(f"no video frames read from {start_frame} to {end_frame}")
            return np.stack(frames, axis=0)
        finally:
            cap.release()

    def load_audio_samples(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Load inclusive audio sample range as int16 array: ``N,channels``."""

        if end_sample < start_sample:
            return np.zeros((0, 0), dtype=np.int16)

        with wave.open(str(self.session_dir / "audio.wav"), "rb") as wav_file:
            channels = wav_file.getnchannels()
            total_frames = wav_file.getnframes()
            start = max(0, min(start_sample, total_frames))
            stop = max(start, min(end_sample + 1, total_frames))
            wav_file.setpos(start)
            raw = wav_file.readframes(stop - start)

        audio = np.frombuffer(raw, dtype=np.int16)
        if audio.size == 0:
            return np.zeros((0, channels), dtype=np.int16)
        return audio.reshape(-1, channels)

    def rows_to_actions(self, rows: list[dict[str, str]], normalize: bool = True) -> np.ndarray:
        """Convert aligned rows to ``Steer,Accel,Brake`` action array."""

        actions = np.array([[as_float(row[column]) for column in ACTION_COLUMNS] for row in rows], dtype=np.float32)
        if normalize:
            # Forza Data Out uses Steer [-127,127], Accel/Brake [0,255].
            actions[:, 0] /= 127.0
            actions[:, 1] /= 255.0
            actions[:, 2] /= 255.0
        return actions

    def rows_to_telemetry(
        self,
        rows: list[dict[str, str]],
        columns: tuple[str, ...] = DEFAULT_TELEMETRY_COLUMNS,
    ) -> np.ndarray:
        """Convert selected aligned telemetry columns to float32 array."""

        return np.array([[as_float(row.get(column, "")) for column in columns] for row in rows], dtype=np.float32)

    def load_valid_clip(self, clip_index: int) -> dict[str, Any]:
        """Load one clip by index from ``clips(valid_only=True)``."""

        return self.load_clip(clip_index, valid_only=True)

    def load_clip(self, clip_id: int, valid_only: bool = False) -> dict[str, Any]:
        """Load one clip as video, audio, aligned rows, actions, and telemetry.

        The video frame range and audio sample range come from ``clips.csv``.
        Telemetry/actions come from matching rows in ``aligned_frames.csv``.
        When ``valid_only`` is true, ``clip_id`` is an index into
        ``clips(valid_only=True)`` instead of the unfiltered clip table.
        """

        clip_row = self.clips(valid_only=valid_only)[clip_id]
        start_frame = as_int(clip_row["start_frame_id"])
        end_frame = as_int(clip_row["end_frame_id"])
        start_audio = as_int(clip_row["start_audio_sample"])
        end_audio = as_int(clip_row["end_audio_sample"])

        rows = self._aligned_rows_for_frame_span(start_frame, end_frame)
        if valid_only:
            invalid_frame_ids = [row["frame_id"] for row in rows if not is_valid_row(row)]
            if invalid_frame_ids:
                raise ValueError(f"clip contains invalid aligned frames: {invalid_frame_ids[:5]}")
        return {
            "clip_row": clip_row,
            "aligned_rows": rows,
            "video_bgr": self.load_video_frames(start_frame, end_frame),
            "audio_pcm": self.load_audio_samples(start_audio, end_audio),
            "actions": self.rows_to_actions(rows),
            "telemetry": self.rows_to_telemetry(rows),
        }
