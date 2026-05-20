# Forza End-to-End Data Collector

Minimal Windows Python collector for synced Forza video, system audio, and Data
Out UDP packets. No UI is included.

## Setup

Install Python 3.11 or 3.12, then:

```powershell
python -m pip install -r requirements.txt
```

In Forza HUD/gameplay options:

1. Enable Data Out.
2. Set Data Out IP to this PC's LAN IP.
3. Set Data Out port to `9999`.

## Record

```powershell
python capture_session.py --duration 300
```

Useful flags:

```powershell
python capture_session.py --duration 60 --fps 60
python capture_session.py --duration 60 --region 0,0,1920,1080
python capture_session.py --duration 30 --no-audio
```

Quick smoke checks:

```powershell
# UDP only: confirms Forza Data Out packets are arriving.
python capture_session.py --duration 15 --no-video --no-audio

# Video + UDP only: useful before debugging WASAPI loopback.
python capture_session.py --duration 15 --no-audio

# Full capture with an explicit 1080p region.
python capture_session.py --duration 60 --region 0,0,1920,1080
```

Each run writes:

```text
data/sessions/YYYYMMDD_HHMMSS/
  manifest.json
  video.avi
  frames.csv
  audio.wav
  audio_blocks.csv
  udp_payloads.bin
  packets.csv
  telemetry.jsonl
  run.log
```

## Align

```powershell
python align_session.py data/sessions/YYYYMMDD_HHMMSS
```

PowerShell example for the latest session:

```powershell
$s = Get-ChildItem data/sessions | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python align_session.py $s.FullName --clip-frames 120 --stride-frames 60
```

This writes:

```text
aligned_frames.csv
clips.csv
```

`video.avi` is constant-FPS for easy playback. Use `frames.csv` and
`aligned_frames.csv` as the real timing source for research data.

## Diagnose And Compress

Check whether a session actually dropped frames:

```powershell
python diagnose_session.py --latest
```

Compress a heavy MJPG AVI into MP4 after recording:

```powershell
python compress_session.py --latest --codec libx264 --crf 23
```

GPU examples if ffmpeg supports NVIDIA NVENC:

```powershell
python compress_session.py --latest --codec h264_nvenc --preset p4 --crf 23
python compress_session.py --latest --codec hevc_nvenc --preset p4 --crf 25
```

Install ffmpeg if the command is not found:

```powershell
winget install Gyan.FFmpeg
# or
conda install -n forza -c conda-forge ffmpeg
```

## Output timing columns

- `t_present_perf_ns`: frame timestamp on the collector monotonic clock.
- `t_audio_start_perf_ns`: start time of an audio block on the same clock.
- `t_recv_perf_ns`: UDP receive time on the same clock.
- `packet_dt_ms`: nearest packet time minus frame time after alignment.

## Minimal Python examples

Read aligned frame rows directly:

```python
import csv
from pathlib import Path

session = Path("data/sessions/YYYYMMDD_HHMMSS")
with (session / "aligned_frames.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

first_valid = next(row for row in rows if row["is_valid"] == "1")
print(first_valid["frame_id"], first_valid["Speed"], first_valid["Steer"])
```

Use the session loader:

```python
from session_data import ForzaSession

session = ForzaSession("data/sessions/YYYYMMDD_HHMMSS")
clip = session.load_clip(0, valid_only=True)

video = clip["video_bgr"]      # T,H,W,3 uint8, OpenCV BGR
audio = clip["audio_pcm"]      # N,channels int16
actions = clip["actions"]      # T,3 float32: steer, accel, brake
telemetry = clip["telemetry"]  # T,37 float32

print(video.shape, audio.shape, actions[:3])
```

Export one model-ready clip:

```powershell
python export_clip_npz.py data/sessions/YYYYMMDD_HHMMSS --clip-id 0 --valid-only
```

Recover one raw UDP packet:

```python
import csv
from pathlib import Path

session = Path("data/sessions/YYYYMMDD_HHMMSS")
with (session / "packets.csv").open(newline="", encoding="utf-8") as f:
    packet = next(csv.DictReader(f))

offset = int(packet["payload_offset"])
size = int(packet["packet_size"])
with (session / "udp_payloads.bin").open("rb") as f:
    f.seek(offset)
    payload = f.read(size)
```
