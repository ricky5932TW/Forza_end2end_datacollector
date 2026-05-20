import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from session_data import ForzaSession


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FakeMediaSession(ForzaSession):
    def load_video_frames(self, start_frame: int, end_frame: int) -> np.ndarray:
        frame_count = len(self._aligned_rows_for_frame_span(start_frame, end_frame))
        return np.full((frame_count, 1, 1, 3), start_frame, dtype=np.uint8)

    def load_audio_samples(self, start_sample: int, end_sample: int) -> np.ndarray:
        sample_count = max(0, end_sample - start_sample + 1)
        return np.zeros((sample_count, 1), dtype=np.int16)


class SessionDataTests(unittest.TestCase):
    def test_load_clip_valid_only_uses_filtered_clip_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            write_rows(
                session_dir / "clips.csv",
                [
                    "clip_id",
                    "start_frame_id",
                    "end_frame_id",
                    "start_audio_sample",
                    "end_audio_sample",
                    "valid_frame_count",
                    "is_valid",
                ],
                [
                    {
                        "clip_id": 0,
                        "start_frame_id": 0,
                        "end_frame_id": 1,
                        "start_audio_sample": 0,
                        "end_audio_sample": 9,
                        "valid_frame_count": 1,
                        "is_valid": 0,
                    },
                    {
                        "clip_id": 1,
                        "start_frame_id": 2,
                        "end_frame_id": 3,
                        "start_audio_sample": 10,
                        "end_audio_sample": 19,
                        "valid_frame_count": 2,
                        "is_valid": 1,
                    },
                    {
                        "clip_id": 2,
                        "start_frame_id": 4,
                        "end_frame_id": 5,
                        "start_audio_sample": 20,
                        "end_audio_sample": 29,
                        "valid_frame_count": 2,
                        "is_valid": 1,
                    },
                ],
            )
            write_rows(
                session_dir / "aligned_frames.csv",
                ["frame_id", "is_valid", "Steer", "Accel", "Brake"],
                [
                    {"frame_id": 0, "is_valid": 0, "Steer": 0, "Accel": 0, "Brake": 0},
                    {"frame_id": 1, "is_valid": 1, "Steer": 0, "Accel": 0, "Brake": 0},
                    {"frame_id": 2, "is_valid": 1, "Steer": 127, "Accel": 255, "Brake": 0},
                    {"frame_id": 3, "is_valid": 1, "Steer": -127, "Accel": 0, "Brake": 255},
                    {"frame_id": 4, "is_valid": 1, "Steer": 0, "Accel": 128, "Brake": 0},
                    {"frame_id": 5, "is_valid": 1, "Steer": 0, "Accel": 0, "Brake": 128},
                ],
            )

            session = FakeMediaSession(session_dir)

            self.assertEqual([row["clip_id"] for row in session.clips(valid_only=True)], ["1", "2"])
            raw_clip = session.load_clip(0)
            valid_clip = session.load_clip(0, valid_only=True)

            self.assertEqual(raw_clip["clip_row"]["clip_id"], "0")
            self.assertEqual(valid_clip["clip_row"]["clip_id"], "1")
            self.assertEqual([row["frame_id"] for row in valid_clip["aligned_rows"]], ["2", "3"])
            np.testing.assert_allclose(
                valid_clip["actions"],
                np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]], dtype=np.float32),
            )

    def test_load_clip_valid_only_rejects_stale_invalid_frame_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            write_rows(
                session_dir / "clips.csv",
                [
                    "clip_id",
                    "start_frame_id",
                    "end_frame_id",
                    "start_audio_sample",
                    "end_audio_sample",
                    "valid_frame_count",
                    "is_valid",
                ],
                [
                    {
                        "clip_id": 0,
                        "start_frame_id": 0,
                        "end_frame_id": 1,
                        "start_audio_sample": 0,
                        "end_audio_sample": 9,
                        "valid_frame_count": 2,
                        "is_valid": 1,
                    }
                ],
            )
            write_rows(
                session_dir / "aligned_frames.csv",
                ["frame_id", "is_valid", "Steer", "Accel", "Brake"],
                [
                    {"frame_id": 0, "is_valid": 1, "Steer": 0, "Accel": 0, "Brake": 0},
                    {"frame_id": 1, "is_valid": 0, "Steer": 0, "Accel": 0, "Brake": 0},
                ],
            )

            session = FakeMediaSession(session_dir)

            with self.assertRaisesRegex(ValueError, "invalid aligned frames"):
                session.load_clip(0, valid_only=True)

    def test_load_clip_uses_aligned_row_span_for_non_contiguous_frame_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            write_rows(
                session_dir / "clips.csv",
                [
                    "clip_id",
                    "start_frame_id",
                    "end_frame_id",
                    "start_audio_sample",
                    "end_audio_sample",
                    "valid_frame_count",
                    "is_valid",
                ],
                [
                    {
                        "clip_id": 0,
                        "start_frame_id": 10,
                        "end_frame_id": 14,
                        "start_audio_sample": 0,
                        "end_audio_sample": 9,
                        "valid_frame_count": 3,
                        "is_valid": 1,
                    }
                ],
            )
            write_rows(
                session_dir / "aligned_frames.csv",
                ["frame_id", "is_valid", "Steer", "Accel", "Brake"],
                [
                    {"frame_id": 10, "is_valid": 1, "Steer": 0, "Accel": 0, "Brake": 0},
                    {"frame_id": 12, "is_valid": 1, "Steer": 0, "Accel": 0, "Brake": 0},
                    {"frame_id": 14, "is_valid": 1, "Steer": 0, "Accel": 0, "Brake": 0},
                ],
            )

            session = FakeMediaSession(session_dir)
            clip = session.load_clip(0, valid_only=True)

            self.assertEqual([row["frame_id"] for row in clip["aligned_rows"]], ["10", "12", "14"])
            self.assertEqual(clip["video_bgr"].shape[0], 3)
            self.assertEqual(clip["actions"].shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
