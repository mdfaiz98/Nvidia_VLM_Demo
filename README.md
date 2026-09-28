# Nvidia VLM Demo

A live, on-device Vision-Language-Model demo web UI for NVIDIA Jetson hardware. Stream a USB webcam, an Orbbec GMSL depth camera, an RTSP camera, or a local video file into a locally-running VLM (via [Ollama](https://ollama.com) or any OpenAI-compatible endpoint) and get real-time AI-generated descriptions of what the camera sees, with live GPU/CPU/RAM telemetry alongside the video feed.

Built for unattended demo stations as much as for interactive use — kiosk mode, auto-reconnect, and a video-file playback fallback all exist to keep a booth or showroom demo running reliably without a person babysitting it.

## Features

- **Real-time VLM analysis** over WebRTC — low-latency video feed with AI-generated captions overlaid live
- **Four video sources** — USB webcam, Orbbec GMSL depth camera, RTSP IP camera, or a looped local video file (useful when a live camera isn't practical)
- **Depth camera page** (`/depth`) — Color, Depth and IR (left/right) side by side with live VLM answers and IMU telemetry from an Orbbec Gemini 335Lg GMSL camera
- **Live system telemetry** — CPU, GPU, and RAM utilization as ring gauges with rolling sparkline history, fed straight from the Jetson's GPU monitor
- **Kiosk mode** — launch with `?autostart=1` to auto-connect and start analysis on page load, for unattended kiosk/demo-station deployments
- **Configurable prompt & response length** — edit the VLM prompt live, cap answer length (1-2 sentences up to unlimited), and adjust max tokens
- **Reasoning-effort control** — toggle model "thinking" (none/low/medium/high) for models that support it
- **Multi-session support** — each browser tab gets its own independent VLM session
- **Any OpenAI-compatible backend** — presets for Ollama, vLLM, SGLang, OpenAI, and the NVIDIA API Catalog, or point it at a custom endpoint

## Requirements

- An NVIDIA Jetson device (developed and tested on Jetson AGX Orin) or any Linux machine with an NVIDIA GPU
- Python 3.10+
- A running VLM backend — [Ollama](https://ollama.com) with a vision-capable model (e.g. `qwen2.5vl:7b`) is the simplest local option
- [`jetson-stats`](https://github.com/rbonghi/jetson_stats) installed (`pip install jetson-stats`) for live GPU/CPU/RAM telemetry on Jetson
- Optional, for the GMSL depth camera: the Orbbec GMSL driver for your carrier board, plus `pip install -e ".[depth-camera]"` (`pyorbbecsdk2`)

## Quick Start

```bash
git clone https://github.com/mdfaiz98/Nvidia_VLM_Demo.git
cd Nvidia_VLM_Demo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Start your VLM backend (example: Ollama with a local vision model):

```bash
ollama serve &
ollama pull qwen2.5vl:7b
```

Start the demo:

```bash
./scripts/start_server.sh --api-base http://localhost:11434/v1 --model qwen2.5vl:7b
```

Open **`https://localhost:8090`** in a browser (accept the self-signed certificate warning) and click **Start**.

On the demo Jetson itself, `scripts/run_demo.sh` does all of the above in one go (max power mode + clocks, Ollama, webui, then opens Firefox) and `scripts/stop_demo.sh` stops it. Both are safe to run repeatedly. `scripts/install_desktop_icons.sh` puts **VLM Demo** / **VLM Demo Stop** shortcuts for them on the desktop.

For an unattended kiosk-style launch that connects and starts analysis automatically:

```
https://localhost:8090/?autostart=1
```

## Cameras

Both live cameras are captured **on the server**, not in the browser. The browser only displays the video it is sent, so any browser works, and no camera permission prompt appears.

- **Webcam tab (USB webcam)** — the server lists USB webcams by name (from `/sys/class/video4linux`, GMSL nodes filtered out), so the webcam is found whatever `/dev/videoN` number it gets. The list is refreshed on every **Start**, so plugging or unplugging cameras is fine. Browser capture (`getUserMedia`) is deliberately not used: with the GMSL driver loaded, the browser's default camera is `/dev/video0`, a raw GMSL depth node, and opening it also disturbs the Orbbec SDK. Note that a browser on another machine also gets the Jetson's USB camera, not its own webcam.
- **GMSL Camera tab and `/depth` page (Orbbec)** — opened through the Orbbec SDK. The first open after the camera has been idle (for example right after power-on) often fails with `Property not found for aliasing`, then works on a retry a few seconds later. The UI shows **"Initializing GMSL camera"** with a countdown and keeps retrying for up to 2 minutes before it reports an error.

Latency tip: the model answers one request at a time, so every open demo tab that is streaming slows the others down. Keep one tab open.

## Configuration

All runtime settings — VLM prompt, answer length, max tokens, reasoning effort, API endpoint, camera source, processing interval — are adjustable live from the UI's side panel, no restart required. See `scripts/start_server.sh --help` for startup flags (port, default model, default API base, etc.).

## Project Layout

- `src/live_vlm_webui/server.py` — aiohttp server: WebRTC signaling, WebSocket protocol, RTSP/video-file/camera endpoints
- `src/live_vlm_webui/usb_camera.py` — server-side USB webcam capture (OpenCV/V4L2)
- `src/live_vlm_webui/orbbec_source.py` — Orbbec GMSL camera: Color track for WebRTC, Depth/IR/IMU for the `/depth` page, startup retry
- `src/live_vlm_webui/gpu_monitor.py` — platform GPU/CPU/RAM/thermal telemetry (NVML, Jetson via `jetson-stats`, Apple Silicon)
- `src/live_vlm_webui/vlm_service.py` — VLM API client (prompt handling, image encoding, response parsing)
- `src/live_vlm_webui/static/` — frontend (`index.html`, `app.js`, `style.css`; `depth.html`/`depth.js`/`depth.css` for `/depth`)
- `scripts/` — demo start/stop scripts and desktop shortcuts, server start/stop scripts, kiosk installer, certificate generation

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). This project builds on [NVIDIA-AI-IOT/live-vlm-webui](https://github.com/NVIDIA-AI-IOT/live-vlm-webui); its original README is kept at [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md) for reference.
