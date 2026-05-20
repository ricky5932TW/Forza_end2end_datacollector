"""Default settings for local Forza data collection.

Keep this file boring on purpose: it is meant to be edited from VSCode before
running the scripts, with command-line flags available for one-off overrides.

Example:
    from dataclasses import replace
    from config import DEFAULT_CONFIG

    config = replace(DEFAULT_CONFIG, video_fps=60, udp_port=9999)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CaptureConfig:
    """Settings shared by capture workers.

    Edit this dataclass for persistent defaults. Use command-line flags for
    one-off changes such as ``--duration`` or ``--region``.
    """

    output_dir: str = "data/sessions"

    # Forza Data Out.
    udp_host: str = "0.0.0.0"
    udp_port: int = 9999
    packet_format: str = "horizon"

    # DXcam capture. Use a 1920x1080 game/display mode for the cleanest
    # 1080p60 path; optional resizing is intentionally left to offline work.
    video_fps: int = 60
    video_output_idx: int = 0
    video_region: tuple[int, int, int, int] | None = None
    video_fourcc: str = "MJPG"
    video_queue_size: int = 180

    # WASAPI loopback audio.
    audio_frames_per_buffer: int = 1024
    audio_queue_size: int = 512
    audio_channels: int = 2

    # Runtime.
    socket_timeout_s: float = 0.2
    flush_every: int = 120


DEFAULT_CONFIG = CaptureConfig()


def config_dict(config: CaptureConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return a JSON-friendly copy of a config."""

    data = asdict(config)
    if data["video_region"] is not None:
        data["video_region"] = list(data["video_region"])
    return data


def parse_region(value: str | None) -> tuple[int, int, int, int] | None:
    """Parse a DXcam region string: left,top,right,bottom."""

    if not value:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be left,top,right,bottom")
    left, top, right, bottom = parts
    if right <= left or bottom <= top:
        raise ValueError("region right/bottom must be greater than left/top")
    return left, top, right, bottom
