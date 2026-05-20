import csv
import json
import tempfile
import unittest
from pathlib import Path

from align_session import sample_for_time, write_aligned_frames, write_clips


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class AlignmentTests(unittest.TestCase):
    def test_sample_for_time(self) -> None:
        self.assertEqual(sample_for_time(1_500_000_000, 0, 1_000_000_000, 48_000), 24_000)

    def test_write_alignment_outputs_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            (session / "manifest.json").write_text(
                json.dumps({"summary": {"audio": {"sample_rate": 48_000}}}),
                encoding="utf-8",
            )
            write_rows(
                session / "frames.csv",
                ["frame_id", "t_present_perf_ns"],
                [
                    {"frame_id": 0, "t_present_perf_ns": 1_000_000_000},
                    {"frame_id": 1, "t_present_perf_ns": 1_016_666_667},
                ],
            )
            write_rows(
                session / "audio_blocks.csv",
                ["block_id", "first_sample", "frame_count", "t_audio_start_perf_ns"],
                [{"block_id": 0, "first_sample": 0, "frame_count": 1024, "t_audio_start_perf_ns": 1_000_000_000}],
            )
            write_rows(
                session / "packets.csv",
                [
                    "packet_id",
                    "t_recv_perf_ns",
                    "parse_error",
                    "TimestampMS",
                    "Speed",
                    "CurrentEngineRpm",
                    "Gear",
                    "Accel",
                    "Brake",
                    "Steer",
                ],
                [
                    {
                        "packet_id": 7,
                        "t_recv_perf_ns": 1_017_000_000,
                        "parse_error": "",
                        "TimestampMS": 123,
                        "Speed": 40.0,
                        "CurrentEngineRpm": 3000.0,
                        "Gear": 3,
                        "Accel": 180,
                        "Brake": 0,
                        "Steer": -4,
                    }
                ],
            )

            aligned = write_aligned_frames(session, max_packet_gap_ms=25)
            write_clips(session, aligned, clip_frames=2, stride_frames=1)

            with (session / "aligned_frames.csv").open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[1]["packet_id"], "7")
            self.assertEqual(rows[1]["is_valid"], "1")
            self.assertEqual(rows[1]["audio_sample_center"], "800")

            with (session / "clips.csv").open("r", newline="", encoding="utf-8") as file:
                clips = list(csv.DictReader(file))
            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0]["is_valid"], "1")


if __name__ == "__main__":
    unittest.main()
