"""Record a Forza video/audio/UDP session with shared timestamps.

Run from VSCode or PowerShell:
    python capture_session.py --duration 300

Common smoke checks:
    python capture_session.py --duration 10 --no-audio
    python capture_session.py --duration 10 --no-video --no-audio

The script records raw streams first. Run ``align_session.py`` after recording
to create frame-level and clip-level alignment files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import queue
import socket
import threading
import time
import wave
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG, CaptureConfig, config_dict, parse_region
from packet_format import PacketSizeError, parse_packet


CORE_TELEMETRY_FIELDS = [
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
]


def perf_ns() -> int:
    """Return the collector's canonical monotonic timestamp.

    Every source writes this clock into its sidecar CSV. Wall-clock time is only
    used for human-readable folder names and logs.
    """

    return time.perf_counter_ns()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def make_session_dir(output_dir: str) -> Path:
    """Create the timestamped output folder for one recording run."""

    session_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def setup_logger(session_dir: Path) -> logging.Logger:
    """Log to both console and ``run.log`` inside the session folder."""

    logger = logging.getLogger("forza_capture")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(session_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def write_manifest(session_dir: Path, manifest: dict[str, Any]) -> None:
    """Write the session metadata file.

    The manifest is rewritten at shutdown so it contains both start metadata and
    final per-source counters.
    """

    with (session_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


def get_dxcam_timestamp_s(camera: Any) -> float | None:
    """Read a DXcam source timestamp when the installed version exposes one."""

    for name in ("timestamp", "last_timestamp", "frame_time", "_timestamp"):
        value = getattr(camera, name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def video_worker(
    session_dir: Path,
    config: CaptureConfig,
    stop_event: threading.Event,
    logger: logging.Logger,
    summary: dict[str, Any],
) -> None:
    """Capture frames with DXcam and write ``video.avi`` plus ``frames.csv``.

    ``video.avi`` is for convenient playback. ``frames.csv`` is the timing
    authority for research because it stores per-frame timestamps and drops.
    """

    try:
        import cv2
        import dxcam
    except ImportError as exc:
        raise RuntimeError("video capture needs dxcam and opencv-python installed") from exc

    frame_queue: queue.Queue[tuple[int, int, int, float | None, int, Any]] = queue.Queue(
        maxsize=config.video_queue_size
    )
    # The writer may briefly lag behind DXcam. A bounded queue keeps latency
    # from growing forever; dropped frames are counted instead of blocking UDP
    # or audio capture.
    dropped = 0
    captured = 0
    written = 0
    grabber_errors: list[str] = []

    def grabber() -> None:
        nonlocal captured, dropped
        camera = None
        try:
            camera = dxcam.create(output_idx=config.video_output_idx, output_color="BGR")
            camera.start(region=config.video_region, target_fps=config.video_fps, video_mode=True)
            logger.info("Video capture started at target_fps=%d", config.video_fps)
            while not stop_event.is_set():
                frame = camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                t_grab_ns = perf_ns()
                dxcam_ts_s = get_dxcam_timestamp_s(camera)
                # ``t_present_perf_ns`` currently uses grab time. If a future
                # DXcam version exposes a stable source clock, keep both values
                # and remap in offline alignment.
                item = (captured, t_grab_ns, t_grab_ns, dxcam_ts_s, dropped, frame)
                captured += 1
                try:
                    frame_queue.put_nowait(item)
                except queue.Full:
                    dropped += 1
        except Exception as exc:  # noqa: BLE001 - surfaced by the parent worker.
            grabber_errors.append(str(exc))
            logger.exception("Video grabber failed")
            stop_event.set()
        finally:
            if camera is not None:
                camera.stop()
            while True:
                try:
                    frame_queue.put_nowait((-1, 0, 0, None, dropped, None))
                    break
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
            logger.info("Video grabber stopped")

    grab_thread = threading.Thread(target=grabber, name="video-grabber", daemon=True)
    grab_thread.start()

    frames_path = session_dir / "frames.csv"
    video_path = session_dir / "video.avi"
    writer = None
    try:
        with frames_path.open("w", newline="", encoding="utf-8") as frames_file:
            fields = [
                "frame_id",
                "t_present_perf_ns",
                "t_grab_perf_ns",
                "dxcam_ts_s",
                "width",
                "height",
                "queue_dropped_before",
            ]
            csv_writer = csv.DictWriter(frames_file, fieldnames=fields)
            csv_writer.writeheader()

            while True:
                try:
                    frame_id, t_present_ns, t_grab_ns, dxcam_ts_s, dropped_before, frame = frame_queue.get(timeout=0.2)
                except queue.Empty:
                    if stop_event.is_set() and not grab_thread.is_alive():
                        break
                    continue
                if frame_id < 0:
                    break

                height, width = frame.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*config.video_fourcc)
                    writer = cv2.VideoWriter(str(video_path), fourcc, config.video_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open video writer: {video_path}")
                    logger.info("Video writer opened: %dx%d %dfps", width, height, config.video_fps)

                writer.write(frame)
                csv_writer.writerow(
                    {
                        "frame_id": frame_id,
                        "t_present_perf_ns": t_present_ns,
                        "t_grab_perf_ns": t_grab_ns,
                        "dxcam_ts_s": "" if dxcam_ts_s is None else f"{dxcam_ts_s:.9f}",
                        "width": width,
                        "height": height,
                        "queue_dropped_before": dropped_before,
                    }
                )
                written += 1
                if written % config.flush_every == 0:
                    frames_file.flush()
    finally:
        stop_event.set()
        grab_thread.join(timeout=3)
        if writer is not None:
            writer.release()
        summary["video"] = {"captured": captured, "written": written, "dropped": dropped}
        logger.info("Video done: captured=%d written=%d dropped=%d", captured, written, dropped)
        if grabber_errors:
            raise RuntimeError(f"video grabber failed: {grabber_errors[0]}")


def find_loopback_device(audio: Any, pyaudio: Any) -> dict[str, Any]:
    """Find the default Windows WASAPI loopback recording device."""

    wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_output = audio.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if default_output.get("isLoopbackDevice"):
        return default_output

    default_name = default_output["name"].split(" [")[0]
    for idx in range(audio.get_device_count()):
        device = audio.get_device_info_by_index(idx)
        if device.get("isLoopbackDevice") and default_name in device["name"]:
            return device

    for idx in range(audio.get_device_count()):
        device = audio.get_device_info_by_index(idx)
        if device.get("isLoopbackDevice"):
            return device
    raise RuntimeError("no WASAPI loopback input device found")


def audio_worker(
    session_dir: Path,
    config: CaptureConfig,
    stop_event: threading.Event,
    logger: logging.Logger,
    summary: dict[str, Any],
) -> None:
    """Record system output audio to ``audio.wav`` and ``audio_blocks.csv``.

    PyAudio callback timing is mapped onto ``perf_counter_ns`` so audio samples
    can be matched to video frames later.
    """

    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise RuntimeError("audio capture needs PyAudioWPatch installed") from exc

    audio = pyaudio.PyAudio()
    stream = None
    audio_queue: queue.Queue[tuple[int, bytes, int, int, dict[str, float], int]] = queue.Queue(
        maxsize=config.audio_queue_size
    )
    callback_dropped = 0
    block_id = 0
    pa_to_perf_offset_ns: int | None = None

    try:
        device = find_loopback_device(audio, pyaudio)
        channels = min(config.audio_channels, int(device.get("maxInputChannels") or config.audio_channels))
        rate = int(device["defaultSampleRate"])
        sample_format = pyaudio.paInt16
        sample_width = audio.get_sample_size(sample_format)

        def callback(in_data: bytes, frame_count: int, time_info: dict[str, float], status: int) -> tuple[None, int]:
            nonlocal block_id, callback_dropped
            t_callback_ns = perf_ns()
            item = (block_id, in_data, frame_count, t_callback_ns, dict(time_info), int(status))
            block_id += 1
            # Keep the callback tiny: no disk writes, no parsing, no logging.
            # This reduces audio glitch/overflow risk.
            try:
                audio_queue.put_nowait(item)
            except queue.Full:
                callback_dropped += 1
            return (None, pyaudio.paComplete if stop_event.is_set() else pyaudio.paContinue)

        stream = audio.open(
            format=sample_format,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=int(device["index"]),
            frames_per_buffer=config.audio_frames_per_buffer,
            stream_callback=callback,
        )
        stream.start_stream()
        pa_to_perf_offset_ns = perf_ns() - int(stream.get_time() * 1_000_000_000)
        logger.info("Audio loopback started: %s %dHz %dch", device["name"], rate, channels)

        first_sample = 0
        blocks_written = 0
        with wave.open(str(session_dir / "audio.wav"), "wb") as wav_file, (
            session_dir / "audio_blocks.csv"
        ).open("w", newline="", encoding="utf-8") as csv_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(rate)

            fields = [
                "block_id",
                "first_sample",
                "frame_count",
                "t_audio_start_perf_ns",
                "t_callback_perf_ns",
                "pa_current_time_s",
                "pa_input_buffer_adc_time_s",
                "status_flags",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            while not stop_event.is_set() or not audio_queue.empty():
                try:
                    item = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                current_block_id, data, frame_count, t_callback_ns, time_info, status = item
                adc_time_s = time_info.get("input_buffer_adc_time", 0.0) or 0.0
                current_time_s = time_info.get("current_time", 0.0) or 0.0
                if adc_time_s > 0 and pa_to_perf_offset_ns is not None:
                    t_audio_start_ns = pa_to_perf_offset_ns + int(adc_time_s * 1_000_000_000)
                else:
                    # Some drivers do not expose ADC time for loopback streams.
                    # The fallback is less exact but still monotonic and useful.
                    block_duration_ns = int(frame_count / rate * 1_000_000_000)
                    t_audio_start_ns = t_callback_ns - block_duration_ns

                wav_file.writeframes(data)
                writer.writerow(
                    {
                        "block_id": current_block_id,
                        "first_sample": first_sample,
                        "frame_count": frame_count,
                        "t_audio_start_perf_ns": t_audio_start_ns,
                        "t_callback_perf_ns": t_callback_ns,
                        "pa_current_time_s": f"{current_time_s:.9f}",
                        "pa_input_buffer_adc_time_s": f"{adc_time_s:.9f}",
                        "status_flags": status,
                    }
                )
                first_sample += frame_count
                blocks_written += 1
                if blocks_written % config.flush_every == 0:
                    csv_file.flush()

        summary["audio"] = {
            "device": device["name"],
            "sample_rate": rate,
            "channels": channels,
            "sample_width": sample_width,
            "blocks_written": blocks_written,
            "callback_dropped": callback_dropped,
        }
    finally:
        stop_event.set()
        if stream is not None:
            try:
                stream.stop_stream()
            finally:
                stream.close()
        audio.terminate()
        logger.info("Audio done: blocks=%s dropped=%s", summary.get("audio", {}).get("blocks_written"), callback_dropped)


def udp_worker(
    session_dir: Path,
    config: CaptureConfig,
    stop_event: threading.Event,
    logger: logging.Logger,
    summary: dict[str, Any],
) -> None:
    """Listen for Forza UDP packets and write raw + parsed telemetry.

    Raw payloads are always kept in ``udp_payloads.bin``. Parsed CSV/JSONL
    fields are convenience outputs, so a parser bug does not destroy the source
    data.
    """

    packets_path = session_dir / "packets.csv"
    payloads_path = session_dir / "udp_payloads.bin"
    telemetry_path = session_dir / "telemetry.jsonl"
    packet_count = 0
    parse_errors = 0
    payload_offset = 0

    fields = [
        "packet_id",
        "t_recv_perf_ns",
        "packet_size",
        "payload_offset",
        "parse_error",
        *CORE_TELEMETRY_FIELDS,
    ]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.bind((config.udp_host, config.udp_port))
        udp_socket.settimeout(config.socket_timeout_s)
        logger.info("UDP listener started on %s:%d", config.udp_host, config.udp_port)

        with payloads_path.open("wb") as payloads_file, packets_path.open(
            "w", newline="", encoding="utf-8"
        ) as packets_file, telemetry_path.open("w", encoding="utf-8") as telemetry_file:
            writer = csv.DictWriter(packets_file, fieldnames=fields)
            writer.writeheader()

            while not stop_event.is_set():
                try:
                    payload, _address = udp_socket.recvfrom(2048)
                except socket.timeout:
                    continue

                t_recv_ns = perf_ns()
                row: dict[str, Any] = {
                    "packet_id": packet_count,
                    "t_recv_perf_ns": t_recv_ns,
                    "packet_size": len(payload),
                    "payload_offset": payload_offset,
                    "parse_error": "",
                }
                try:
                    telemetry = parse_packet(payload, packet_format=config.packet_format)
                except (PacketSizeError, ValueError) as exc:
                    telemetry = {}
                    parse_errors += 1
                    row["parse_error"] = str(exc)
                for key in CORE_TELEMETRY_FIELDS:
                    row[key] = telemetry.get(key, "")

                # Store raw first-class data. ``payload_offset`` and
                # ``packet_size`` in packets.csv let downstream code recover
                # every UDP packet byte-for-byte.
                payloads_file.write(payload)
                writer.writerow(row)
                telemetry_file.write(
                    json.dumps(
                        {
                            "packet_id": packet_count,
                            "t_recv_perf_ns": t_recv_ns,
                            "packet_size": len(payload),
                            "payload_offset": payload_offset,
                            "parse_error": row["parse_error"],
                            **telemetry,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                payload_offset += len(payload)
                packet_count += 1
                if packet_count % config.flush_every == 0:
                    payloads_file.flush()
                    packets_file.flush()
                    telemetry_file.flush()

    summary["udp"] = {"packets": packet_count, "parse_errors": parse_errors}
    logger.info("UDP done: packets=%d parse_errors=%d", packet_count, parse_errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Forza video/audio/UDP session",
        epilog=(
            "Examples:\n"
            "  python capture_session.py --duration 300\n"
            "  python capture_session.py --duration 60 --region 0,0,1920,1080\n"
            "  python capture_session.py --duration 10 --no-video --no-audio"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=None, help="seconds to record; omit to stop with Ctrl+C")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--host", default=DEFAULT_CONFIG.udp_host)
    parser.add_argument("--port", type=int, default=DEFAULT_CONFIG.udp_port)
    parser.add_argument("--fps", type=int, default=DEFAULT_CONFIG.video_fps)
    parser.add_argument("--region", default=None, help="DXcam region: left,top,right,bottom")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-udp", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Start enabled workers, wait for duration/Ctrl+C, then close the session."""

    args = parse_args()
    config = replace(
        DEFAULT_CONFIG,
        output_dir=args.output_dir,
        udp_host=args.host,
        udp_port=args.port,
        video_fps=args.fps,
        video_region=parse_region(args.region),
    )
    session_dir = make_session_dir(config.output_dir)
    logger = setup_logger(session_dir)
    stop_event = threading.Event()
    summary: dict[str, Any] = {}
    manifest = {
        "session_dir": str(session_dir),
        "started_wall_time_iso": now_iso(),
        "perf_start_ns": perf_ns(),
        "config": config_dict(config),
        "enabled": {"video": not args.no_video, "audio": not args.no_audio, "udp": not args.no_udp},
    }
    write_manifest(session_dir, manifest)

    workers: list[threading.Thread] = []
    worker_errors: list[str] = []
    targets = []
    if not args.no_video:
        targets.append(("video", video_worker))
    if not args.no_audio:
        targets.append(("audio", audio_worker))
    if not args.no_udp:
        targets.append(("udp", udp_worker))

    try:
        def run_worker(name: str, target: Any) -> None:
            try:
                target(session_dir, config, stop_event, logger, summary)
            except Exception as exc:  # noqa: BLE001 - logged and surfaced after shutdown.
                worker_errors.append(f"{name}: {exc}")
                logger.exception("%s worker failed", name)
                stop_event.set()

        for name, target in targets:
            thread = threading.Thread(target=run_worker, args=(name, target), name=name)
            thread.start()
            workers.append(thread)

        if args.duration is None:
            while any(thread.is_alive() for thread in workers):
                if worker_errors:
                    break
                time.sleep(0.5)
        else:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline and not worker_errors:
                time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Stop requested")
    finally:
        stop_event.set()
        for thread in workers:
            thread.join()
        manifest["ended_wall_time_iso"] = now_iso()
        manifest["perf_end_ns"] = perf_ns()
        manifest["summary"] = summary
        if worker_errors:
            manifest["errors"] = worker_errors
        write_manifest(session_dir, manifest)
        logger.info("Session written to %s", session_dir)
        print(session_dir)
        if worker_errors:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
