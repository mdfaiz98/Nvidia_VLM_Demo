# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WebRTC Live VLM WebUI Server
Main server that handles WebRTC connections and serves the web interface
"""

import asyncio
import glob
import ipaddress
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import aiohttp
import cv2
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaRelay
from PIL import Image

from .detection_processor import DetectionVideoTrack
from .detector import YoloDetector
from .gpu_monitor import create_monitor
from .orbbec_source import (
    OrbbecColorTrack,
    OrbbecDeviceNotFoundError,
    OrbbecWarmingUpError,
    get_orbbec_manager,
)
from .rtsp_track import RTSPVideoTrack
from .usb_camera import UsbCameraError, UsbCameraTrack, get_usb_camera, list_usb_cameras
from .video_file_track import VideoFileTrack
from .video_processor import VideoProcessorTrack
from .vlm_service import VLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global objects
relay = MediaRelay()
pcs = set()
vlm_service = None  # Kept for backwards compat; default session uses sessions["default"]
websockets = set()  # Track active WebSocket connections (all)
gpu_monitor = None  # GPU monitoring instance
gpu_monitor_task = None  # Background task for GPU monitoring
rtsp_tracks = {}  # Track active RTSP streams {session_id: (rtsp_track, processor_track)}

# Detection cascade demo (/traffic) - deliberately separate from the normal
# demo's session/VLM machinery above; see CLAUDE.md and
# VLM_Cascade_Architecture_Plan.md for why this is a standalone page.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Default detector: YOLOv8n on DLA 0. Measured on the highway video it finds the
# same vehicles as YOLO26n (the IQ9's model) but runs ~25% faster on the DLA
# (25 vs 33 ms/frame), since YOLO26's end-to-end head moves more work between
# the DLA and the GPU - both heads exceed the DLA's 8192 size limit and fall back.
TRAFFIC_MODEL_PATH = _REPO_ROOT / "models" / "yolov8n_dla0_fp16.engine"
TRAFFIC_LABELS_PATH = _REPO_ROOT / "models" / "coco_labels.txt"
TRAFFIC_VIDEO_PATH = _REPO_ROOT / "demo_assets" / "car-highway.mp4"
TRAFFIC_MODELS_DIR = _REPO_ROOT / "models"
TRAFFIC_VIDEOS_DIR = _REPO_ROOT / "demo_assets"
TRAFFIC_UPLOADS_DIR = TRAFFIC_VIDEOS_DIR / "uploads"
# Entry zone per demo video (x1, y1, x2, y2 as fractions of the frame) - only
# vehicles inside it are drawn and logged (see DetectionVideoTrack.entry_zone).
# The parking clip's zone is the ground in front of the barrier: street traffic
# behind it has box bottoms at y < 0.27, cars at/through the barrier at
# y >= 0.3, and the car parked at the right edge sits at x > 0.9 (measured from
# every detection in the clip). Videos not listed use the whole frame.
TRAFFIC_ENTRY_ZONES = {
    "parking-entrance-barrier.mp4": (0.15, 0.28, 0.80, 1.0),
}
TRAFFIC_VIDEO_NAMES = {
    "car-highway.mp4": "Highway traffic",
    "parking-entrance-barrier.mp4": "Parking entrance (barrier)",
}
yolo_detector = None  # The active detector - loaded at startup (TensorRT engine deserialize ~1s) - see on_startup
# Settings drawer: every TensorRT .engine (DLA or GPU build) and every .onnx
# (CPU, via ONNX Runtime) in models/ is selectable as the detection model, all
# sharing coco_labels.txt. Loaded on first use, then kept so switching back is
# instant.
traffic_detectors = {}  # model file name -> YoloDetector
traffic_detection_model_id = TRAFFIC_MODEL_PATH.name
traffic_websockets = set()  # Track active WebSocket connections for the /traffic page
# The single-worker detection executor (see detection_processor.py) means
# multiple simultaneous /traffic connections serialize against each other
# for the detector - if a reconnect creates a new connection before the old
# one is torn down, both compete for that one worker, frame delivery stalls,
# and WebRTC's own timeout closes the connection, triggering another
# reconnect (observed as a rapid connect/disconnect loop that eventually
# crashed the process). This is also just correct behavior for a
# single-screen kiosk demo: only one active /traffic connection at a time.
traffic_active_pc = None
traffic_active_video_track = None
traffic_active_processor_track = None  # the DetectionVideoTrack - see "toggle_detection"
# Opening/closing PyAV video containers back-to-back with no gap was
# observed to trigger a native heap-corruption crash ("munmap_chunk():
# invalid pointer") - this enforces a minimum spacing between /traffic
# connection setups server-side, independent of client behavior (a client
# retrying aggressively, e.g. a stale browser tab still running old JS
# without backoff, can't force the server back into that failure mode).
traffic_last_offer_time = 0.0
TRAFFIC_MIN_OFFER_INTERVAL_SECONDS = 2.0

# Phase 3: click-to-caption. traffic_vlm_service is a dedicated VLMService
# instance (separate lock state from the main demo's, but pointed at the
# same Ollama server underneath) used only for per-vehicle captions,
# never for continuous full-frame analysis. traffic_track_store retains each
# reported vehicle's full-resolution crop (NOT the shrunk display thumbnail
# broadcast to the panel) keyed by track id, since a click can arrive well
# after tracker.py's own Track object has expired/been GC'd; capped and
# FIFO-evicted so a long-running demo can't grow this unboundedly.
traffic_vlm_service: Optional[VLMService] = None
traffic_track_store: "OrderedDict[str, dict]" = OrderedDict()
TRAFFIC_TRACK_STORE_MAX = 200
# Describe prompts, one per kind of video: "default" (no entry zone - the
# highway video, the original prompt) and "zone" (entry-zone videos - the
# parking entrance).
TRAFFIC_CAPTION_DEFAULT_PROMPT = (
    "Describe this vehicle in 1-2 sentences: its color, type (car/truck/bus/etc.), "
    "and brand if visible. Only mention the license plate if it is clearly legible - "
    "don't guess at characters you can't read clearly."
)
# Plate on its own line, with an explicit "not readable" answer: with only
# "don't guess" Qwen3-VL invented a plate for a car whose plate was behind the
# parking barrier arm; with this wording it says "not readable" there, and
# reads the plate (and brand) once the car is clear.
TRAFFIC_CAPTION_PARKING_PROMPT = (
    "Describe this vehicle in one sentence: color, type (car/SUV/van/truck/etc.) and brand "
    "if visible. Then on a new line write 'Plate: ' followed by the license plate exactly as "
    "shown, only if every character is clearly readable. If the plate is blocked, blurry, "
    "cut off or partly hidden, write 'Plate: not readable'. Never guess."
)
TRAFFIC_CAPTION_DEFAULTS = {"default": TRAFFIC_CAPTION_DEFAULT_PROMPT, "zone": TRAFFIC_CAPTION_PARKING_PROMPT}
# The active prompts - each editable at runtime from the /traffic page while a
# video of that kind plays (see "update_prompt" in traffic_websocket_handler);
# TRAFFIC_CAPTION_DEFAULTS stays fixed as the "Reset to default" targets.
traffic_caption_prompts = dict(TRAFFIC_CAPTION_DEFAULTS)


def traffic_prompt_mode() -> str:
    """"zone" while an entry-zone video (parking) plays, else "default"."""
    track = traffic_active_processor_track
    return "zone" if track is not None and track.entry_zone is not None else "default"
TRAFFIC_CAPTION_MAX_TOKENS = 120
# Kept from the IQ9 demo, where a GenieX stall caused request pile-up; after a
# failed caption, further requests wait this long instead of queueing up.
TRAFFIC_CAPTION_FAILURE_COOLDOWN_SECONDS = 40.0
traffic_caption_last_failure_at: Optional[float] = None
traffic_caption_lock = asyncio.Lock()

# Multi-session state (0.4.0)
default_vlm_config = {}  # Set at startup; used to create new sessions
sessions = {}  # session_id -> {"vlm_service": VLMService}
session_websockets = defaultdict(set)  # session_id -> set of ws
ws_to_session = {}  # ws -> session_id


def get_or_create_session(session_id: str):
    """Get or create per-session state (VLM service). Thread-safe for aiohttp."""
    if session_id not in sessions:
        cfg = default_vlm_config
        sessions[session_id] = {
            "vlm_service": VLMService(
                model=cfg.get("model", "meta/llama-3.2-11b-vision-instruct"),
                api_base=cfg.get("api_base", "http://localhost:8000/v1"),
                api_key=cfg.get("api_key", "EMPTY"),
                prompt=cfg.get("prompt", "Describe what you see in this image in one sentence."),
                reasoning_effort=cfg.get("reasoning_effort"),
                max_tokens=cfg.get("max_tokens", 512),
            ),
            "show_request_payload": False,
            "show_response_payload": False,
        }
        logger.info(f"Created new session: {session_id}")
    return sessions[session_id]


def send_to_session(session_id: str, message: str):
    """Send a message only to WebSocket clients in this session."""
    for ws in session_websockets.get(session_id, set()):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to session {session_id}: {e}")


def get_session_callback(session_id: str):
    """Return a text_callback that sends VLM results only to this session."""

    def callback(text: str, metrics: dict):
        out = {"type": "vlm_response", "text": text, "metrics": metrics}
        session = sessions.get(session_id)
        if session and session.get("vlm_service"):
            svc = session["vlm_service"]
            if session.get("show_request_payload"):
                payload = svc.get_last_request_payload()
                if payload is not None:
                    out["request_payload"] = payload
            if session.get("show_response_payload"):
                payload = svc.get_last_response_payload()
                if payload is not None:
                    try:
                        out["response_payload"] = json.loads(json.dumps(payload, default=str))
                    except (TypeError, ValueError):
                        out["response_payload"] = payload
        send_to_session(session_id, json.dumps(out))

    return callback


def is_port_available(port, host="0.0.0.0"):
    """Check if a port is available for binding"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def find_process_using_port(port):
    """Find what process is using a port (Linux/Unix only)"""
    try:
        # Try lsof first (more reliable)
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            pid = result.stdout.strip().split()[0]
            # Get process name
            name_result = subprocess.run(
                ["ps", "-p", pid, "-o", "comm="], capture_output=True, text=True, timeout=2
            )
            if name_result.returncode == 0:
                return f"PID {pid} ({name_result.stdout.strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # lsof not available, try netstat
        try:
            result = subprocess.run(
                ["netstat", "-tulpn"], capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTEN" in line:
                    parts = line.split()
                    if len(parts) >= 7:
                        return parts[-1]  # PID/Program name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return "unknown process"


def find_available_port(start_port=8080, max_attempts=10):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    return None


def is_loopback_host(host):
    """Return True when a server host value is limited to localhost."""
    normalized = (host or "").strip().lower().strip("[]")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


async def detect_local_service_and_model():
    """
    Auto-detect available local VLM services and select a model
    Returns: (api_base, model_name) or (None, None) if no service found
    """
    services = [
        ("http://localhost:11434/v1", "Ollama"),
        ("http://localhost:8000/v1", "vLLM"),
        ("http://localhost:30000/v1", "SGLang"),
    ]

    for api_base, service_name in services:
        try:
            # Try to connect to the service
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{api_base}/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            # Prefer vision models
                            vision_keywords = ["vision", "llava", "llama-3.2", "gemini"]
                            for model in models:
                                model_id = model.get("id", "")
                                if any(keyword in model_id.lower() for keyword in vision_keywords):
                                    logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                                    logger.info(f"   Selected model: {model_id}")
                                    return (api_base, model_id)

                            # If no vision model found, use the first one
                            model_id = models[0].get("id", "")
                            logger.info(f"✅ Auto-detected {service_name} at {api_base}")
                            logger.info(
                                f"   Selected model: {model_id} (vision model preferred but not found)"
                            )
                            return (api_base, model_id)
        except Exception as e:
            logger.debug(f"Service {service_name} not available at {api_base}: {e}")
            continue

    return (None, None)


async def index(request):
    """Serve the main HTML page. Injects the configured --api-base CLI arg (if
    any) as the default shown in the UI."""
    content = open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r").read()

    configured_api_base = default_vlm_config.get("api_base")
    if configured_api_base:
        content = content.replace(
            'id="apiBaseUrl" value="http://localhost:11434/v1"',
            f'id="apiBaseUrl" value="{configured_api_base}"',
        )

    return web.Response(content_type="text/html", text=content, headers={"Cache-Control": "no-store"})


# AFE-R750 datasheet (Advantech, "AFE-R750_DS", last updated 12-Jun-2026), AGX Orin
# 32GB column (AFE-R750-X2A1U), shown in the page's Platform panel on this box.
AFE_R750_32GB_SPECS = {
    "name": "Advantech AFE-R750",
    "tagline": "NVIDIA Jetson AGX Orin 32GB AMR Control System",
    "datasheet": "AFE-R750 datasheet, last updated 12-Jun-2026",
    "sections": [
        ["Compute", [
            ["Module", "NVIDIA Jetson AGX Orin 32GB"],
            ["CPU", "8-core NVIDIA Arm Cortex-A78AE v8.2"],
            ["GPU", "1792-core NVIDIA Ampere with 56 Tensor Cores, up to 930 MHz"],
            ["AI performance", "Up to 200 sparse TOPS (INT8)"],
            ["Memory", "32 GB 256-bit LPDDR5"],
            ["Storage", "64 GB eMMC 5.1 + M.2 M-Key 2280 (PCIe x4)"],
        ]],
        ["I/O", [
            ["Display", "1x HDMI 2.0, up to 3840 x 2160 @ 24 Hz"],
            ["Ethernet", "4x 2.5GbE RJ-45 (Intel i226)"],
            ["USB", "4x USB 3.2 Type-A + 1x Type-C (system recovery)"],
            ["Serial / CAN", "2x RS-232/422/485, 2x isolated CAN FD (1/2 Mbps)"],
            ["GPIO", "16-bit isolated DIO"],
            ["Cameras", "4 or 8x GMSL FAKRA (optional)"],
            ["Expansion", "M.2 E-Key 2230 (PCIe x1, USB 2), M.2 B-Key 3042/3052 (PCIe x1, USB 2, SIM)"],
            ["Sensors", "Bosch BMI088 6-axis IMU on board; optional Xsens MTi3 9-axis AHRS"],
            ["Board", "TPM 2.0, GPS PPS time sync (optional)"],
        ]],
        ["Power & mechanical", [
            ["Power input", "20-28 V DC, ATX/AT (ATX default)"],
            ["TDP", "15-40 W"],
            ["Operating temp.", "-10 to 60 C (0.7 m/s airflow)"],
            ["Vibration", "3 Grms @ 5-500 Hz"],
            ["Size / weight", "177 x 174 x 80 mm, 2.66 kg, wall mount"],
            ["Certification", "CE / FCC Class B, CB, UL, CCC, BSMI"],
            ["OS", "Ubuntu 22.04 LTS with JetPack 6.1, Advantech Robotic Suite (ROS 2)"],
        ]],
    ],
}

# Jetson AGX Orin 32GB module specs (NVIDIA Jetson AGX Orin series datasheet),
# for an AGX Orin 32GB in any other carrier board. Module only - no carrier I/O.
AGX_ORIN_32GB_SPECS = {
    "name": "NVIDIA Jetson AGX Orin 32GB",
    "tagline": "Ampere GPU edge AI module",
    "datasheet": "NVIDIA Jetson AGX Orin series module datasheet",
    "sections": [
        ["Compute", [
            ["AI performance", "200 TOPS (INT8, sparse)"],
            ["GPU", "1792-core NVIDIA Ampere with 56 Tensor Cores, up to 930 MHz"],
            ["DL accelerators", "2x NVDLA v2.0"],
            ["Vision accelerator", "PVA v2.0"],
            ["CPU", "8-core Arm Cortex-A78AE, up to 2.2 GHz"],
            ["Memory", "32 GB 256-bit LPDDR5, 204.8 GB/s"],
            ["Storage", "64 GB eMMC 5.1"],
        ]],
        ["Power", [
            ["Module power", "15 W - 40 W (power modes)"],
        ]],
    ],
}

_runtime_version_cache: Optional[dict] = None


def _runtime_versions() -> dict:
    """Ollama and JetPack (L4T) versions, read once and cached."""
    global _runtime_version_cache
    if _runtime_version_cache is None:
        out = {}
        try:
            text = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"version is (\S+)", text)
            if m:
                out["ollama"] = m.group(1)
        except Exception:
            pass
        m = re.search(r"R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)", _read_text("/etc/nv_tegra_release"))
        if m:
            out["l4t"] = f"R{m.group(1)}.{m.group(2)}"
        _runtime_version_cache = out
    return _runtime_version_cache


def _read_text(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().replace("\0", "").strip()
    except OSError:
        return ""


async def platform_info(request):
    """Static + live platform facts for the page's Platform panel: datasheet
    specs (AFE-R750, or the bare AGX Orin 32GB module), board / OS / kernel, memory, storage, network
    addresses, uptime, CPU clocks, and the AI stack the demo is using."""
    board = _read_text("/proc/device-tree/model") or _read_text("/sys/class/dmi/id/product_name")
    os_pretty = ""
    for line in _read_text("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            os_pretty = line.split("=", 1)[1].strip('"')
    mem = {}
    for line in _read_text("/proc/meminfo").splitlines():
        k, v = line.split(":", 1)
        mem[k] = int(v.split()[0]) / 1024 / 1024
    disk = shutil.disk_usage("/")
    max_khz = [int(_read_text(f)) for f in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq")
               if _read_text(f).isdigit()]
    ips = []
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=1)
        ips = [ip for ip in result.stdout.split() if ":" not in ip and not ip.startswith(("127.", "172.17.", "10.10."))]
    except Exception:
        pass
    uptime_s = float((_read_text("/proc/uptime").split() or ["0"])[0])
    versions = _runtime_versions()
    ram_total_gb = mem.get("MemTotal", 0)
    is_agx_orin_32gb = "AGX ORIN" in board.upper() and 28 <= ram_total_gb <= 33
    # The AFE-R750 reports the devkit's device-tree model, so recognise it by hostname
    is_afe_r750 = "AFER750" in (board + socket.gethostname()).replace("-", "").upper()
    if is_agx_orin_32gb:
        specs = AFE_R750_32GB_SPECS if is_afe_r750 else AGX_ORIN_32GB_SPECS
    else:
        specs = None
    api_base = default_vlm_config.get("api_base") or ""
    uses_ollama = ":11434" in api_base
    return web.json_response({
        "specs": specs,
        "live": {
            "hostname": socket.gethostname(),
            "board": board,
            "os": os_pretty,
            "kernel": platform.release(),
            "l4t": versions.get("l4t"),
            "uptime_s": uptime_s,
            "cpu_cores": os.cpu_count(),
            "cpu_max_ghz": round(max(max_khz) / 1e6, 2) if max_khz else None,
            "ram_total_gb": round(ram_total_gb, 1),
            "ram_used_gb": round(ram_total_gb - mem.get("MemAvailable", 0), 1),
            "disk_total_gb": round(disk.total / 1e9),
            "disk_used_gb": round(disk.used / 1e9),
            "ips": ips,
        },
        "ai": {
            "model": default_vlm_config.get("model"),
            "runtime": "Ollama (llama.cpp, CUDA)" if uses_ollama else None,
            "ollama": versions.get("ollama") if uses_ollama else None,
            "accelerator": "Ampere GPU (CUDA)" if versions.get("l4t") else None,
        },
    })


async def app_style(request):
    """Serve the stylesheet, no-store so edits show up on refresh."""
    path = os.path.join(os.path.dirname(__file__), "static", "style.css")
    content = open(path, "r").read()
    return web.Response(content_type="text/css", text=content, headers={"Cache-Control": "no-store"})


async def app_script(request):
    """Serve the frontend JS, no-store so edits show up on refresh."""
    path = os.path.join(os.path.dirname(__file__), "static", "app.js")
    content = open(path, "r").read()
    return web.Response(content_type="text/javascript", text=content, headers={"Cache-Control": "no-store"})


async def depth_page(request):
    """Serve the depth-camera test page (Color/Depth/IR grid + VLM output)."""
    path = os.path.join(os.path.dirname(__file__), "static", "depth.html")
    content = open(path, "r").read()
    return web.Response(content_type="text/html", text=content, headers={"Cache-Control": "no-store"})


async def depth_style(request):
    """Serve the depth-camera page's stylesheet. See app_style() re: no-store."""
    path = os.path.join(os.path.dirname(__file__), "static", "depth.css")
    content = open(path, "r").read()
    return web.Response(content_type="text/css", text=content, headers={"Cache-Control": "no-store"})


async def depth_script(request):
    """Serve the depth-camera page's JS. See app_style() re: no-store."""
    path = os.path.join(os.path.dirname(__file__), "static", "depth.js")
    content = open(path, "r").read()
    return web.Response(content_type="text/javascript", text=content, headers={"Cache-Control": "no-store"})


async def usb_camera_list(request):
    """
    USB webcams the server can capture from (GMSL nodes excluded).

    GET /api/usbcam/list -> {"cameras": [{"device": "/dev/video8", "name": "..."}]}
    """
    return web.json_response({"cameras": list_usb_cameras()})


async def orbbec_ready(request):
    """
    Try to open the Orbbec camera and report whether it is ready.

    Both pages poll this before streaming, so the GMSL camera's slow first
    open (see orbbec_source.WARMUP_WINDOW_S) shows as "Initializing GMSL
    camera" with a countdown instead of an error.

    GET /api/orbbec/ready
    200 {"ready": true, "device": {...}}
    503 {"ready": false, "warming_up": true, "retry_after": <s>, "remaining": <s>}
    404/500 {"ready": false, "error": "..."}
    """
    try:
        device = get_orbbec_manager().start()
    except OrbbecWarmingUpError as e:
        return web.json_response(
            {
                "ready": False,
                "warming_up": True,
                "retry_after": e.retry_after,
                "remaining": e.remaining,
            },
            status=503,
        )
    except OrbbecDeviceNotFoundError as e:
        return web.json_response({"ready": False, "error": str(e)}, status=404)
    except Exception as e:
        logger.error(f"Failed to start Orbbec camera: {e}")
        return web.json_response(
            {"ready": False, "error": f"Failed to start Orbbec camera: {e}"}, status=500
        )
    return web.json_response({"ready": True, "device": device})


async def orbbec_ws(request):
    """
    Serve Depth/IR JPEG frames and IMU telemetry from the Orbbec camera.

    Kept separate from the main /ws handler (VLM text/prompt/config messages)
    since this is a different concern entirely. Color goes over the existing
    WebRTC pipeline instead (see offer()'s "orbbec" branch); Depth/IR ride a
    plain WebSocket rather than extra WebRTC tracks - see CLAUDE.md's
    aioice/loopback notes on why more WebRTC tracks isn't the easy win it
    looks like here.

    Pull-based, so at most one frame is ever in flight: the client sends
    {"type": "ready", "ir": "left"|"right"} once it has drawn the previous
    frame, and the server answers with the next *new* camera frame. A fixed-
    rate push would let frames pile up in socket buffers whenever the
    browser falls behind, making Depth/IR drift further behind Color. Each
    answer is a JSON header, then one binary message per image, tagged by
    its first byte (0 = depth, 1 = IR) - raw JPEG, no base64 inflation.

    GET /ws/orbbec
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    try:
        manager = get_orbbec_manager()
        manager.start()
    except OrbbecDeviceNotFoundError as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return ws
    except Exception as e:
        logger.error(f"Failed to start Orbbec camera for WebSocket: {e}", exc_info=True)
        await ws.send_json({"type": "error", "message": f"Failed to start Orbbec camera: {e}"})
        await ws.close()
        return ws

    loop = asyncio.get_event_loop()
    last_version = -1
    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                request_data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if request_data.get("type") != "ready":
                continue
            ir_side = "right" if request_data.get("ir") == "right" else "left"

            # Paced by the camera itself (30fps), not a timer. Times out so a
            # stalled camera still answers (with no images) and the client
            # keeps asking rather than hanging.
            last_version = await loop.run_in_executor(
                None, manager.wait_for_frame_after, last_version, 1.0
            )

            # cv2.imencode is real CPU work - run it off the event loop so it
            # can't add scheduling jitter to the Color track's own VLM
            # inference round trip, which shares this same asyncio loop.
            depth_jpeg, ir_jpeg = await asyncio.gather(
                loop.run_in_executor(None, manager.encode_depth_jpeg),
                loop.run_in_executor(None, manager.encode_ir_jpeg, ir_side),
            )

            await ws.send_json(
                {
                    "type": "orbbec_frame",
                    "has_depth": depth_jpeg is not None,
                    "has_ir": ir_jpeg is not None,
                    "telemetry": manager.get_telemetry(),
                }
            )
            if depth_jpeg:
                await ws.send_bytes(b"\x00" + depth_jpeg)
            if ir_jpeg:
                await ws.send_bytes(b"\x01" + ir_jpeg)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:
        logger.error(f"Orbbec WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("Orbbec WebSocket client disconnected")

    return ws


async def guide_index(request):
    """Serve the Orin demo guide (diagrams + technical details + troubleshooting,
    static, offline), linked from the main demo's Settings and /traffic."""
    content = open(os.path.join(os.path.dirname(__file__), "static", "guide.html"), "r").read()
    return web.Response(content_type="text/html", text=content, headers={"Cache-Control": "no-store"})


async def compare_index(request):
    """Serve the Orin GPU vs IQ9 NPU explainer + cheat sheet (static, offline),
    linked from the main demo's Settings and the /traffic settings drawer."""
    content = open(os.path.join(os.path.dirname(__file__), "static", "compare.html"), "r").read()
    return web.Response(content_type="text/html", text=content, headers={"Cache-Control": "no-store"})


async def traffic_index(request):
    """Serve the /traffic cascade demo page - separate from index.html."""
    content = open(os.path.join(os.path.dirname(__file__), "static", "traffic.html"), "r").read()
    return web.Response(content_type="text/html", text=content, headers={"Cache-Control": "no-store"})


async def traffic_style(request):
    """Serve the /traffic demo's stylesheet. See app_style() re: no-store."""
    path = os.path.join(os.path.dirname(__file__), "static", "traffic.css")
    content = open(path, "r").read()
    return web.Response(content_type="text/css", text=content, headers={"Cache-Control": "no-store"})


async def traffic_script(request):
    """Serve the /traffic demo's JS. See app_style() re: no-store."""
    path = os.path.join(os.path.dirname(__file__), "static", "traffic.js")
    content = open(path, "r").read()
    return web.Response(content_type="text/javascript", text=content, headers={"Cache-Control": "no-store"})


def broadcast_traffic_detections(tracks, crops, crops_large, frame_width, frame_height):
    """Send newly-confirmed vehicle sightings (with cropped thumbnails) to
    all connected /traffic WebSocket clients - called once per vehicle, the
    first time the tracker confirms it (see tracker.py), not once per frame.

    Two display sizes are sent: a small "crop" for the panel list, and a
    larger "crop_large" for the click-to-open detail modal - both are still
    below the full-resolution crop the VLM itself analyzes (retained
    separately in traffic_track_store), which is why the VLM can sometimes
    make out detail (e.g. plate text) a viewer can't see in the modal."""
    objects = [
        {
            "id": t.id,
            "label": t.label,
            "score": round(t.best_score, 3),
            "bbox": list(t.best_bbox),
            "crop": crop_b64,
            "crop_large": crop_large_b64,
        }
        for t, crop_b64, crop_large_b64 in zip(tracks, crops, crops_large)
    ]

    for t in tracks:
        traffic_track_store[t.id] = {
            "label": t.label,
            "score": t.best_score,
            "best_crop": t.best_crop,  # full-res ndarray, not the shrunk thumbnail
            # Entry-zone (parking) videos only: the tracker keeps refining
            # t.best_crop after the card appears, and Describe uses that latest
            # crop - see caption_traffic_vehicle. Other videos (the highway) send
            # the crop as it was when the card appeared, as originally.
            "track": t,
            "mode": traffic_prompt_mode(),
            "caption": None,
            "caption_pending": False,
        }
        traffic_track_store.move_to_end(t.id)
    while len(traffic_track_store) > TRAFFIC_TRACK_STORE_MAX:
        traffic_track_store.popitem(last=False)

    message = json.dumps({"type": "new_detections", "objects": objects})
    for ws in list(traffic_websockets):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error broadcasting traffic detection: {e}")


def broadcast_traffic_caption_status(track_id, status, caption=None):
    """Send a caption_pending/caption_result/caption_error update for one
    vehicle to all connected /traffic WebSocket clients."""
    message = json.dumps({"type": "caption_status", "id": track_id, "status": status, "caption": caption})
    for ws in list(traffic_websockets):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error broadcasting traffic caption status: {e}")


async def caption_traffic_vehicle(track_id: str):
    """Generate a VLM caption for one previously-detected vehicle's stored
    crop, on demand (click-to-caption). Serialized via traffic_caption_lock
    (one caption call at a time), with a short cooldown after a failure so
    requests can't pile up behind a stalled one."""
    global traffic_caption_last_failure_at

    entry = traffic_track_store.get(track_id)
    if entry is None:
        logger.debug(f"[traffic] Caption requested for unknown/expired track {track_id}")
        return
    if entry["caption"] is not None or entry["caption_pending"]:
        return  # already captioned or already in flight
    if traffic_vlm_service is None:
        broadcast_traffic_caption_status(track_id, "error", "Captioning unavailable")
        return

    if traffic_caption_last_failure_at is not None:
        elapsed = time.time() - traffic_caption_last_failure_at
        if elapsed < TRAFFIC_CAPTION_FAILURE_COOLDOWN_SECONDS:
            logger.info(
                f"[traffic] In cooldown after a recent VLM failure "
                f"({elapsed:.0f}s / {TRAFFIC_CAPTION_FAILURE_COOLDOWN_SECONDS:.0f}s), "
                f"skipping caption request for {track_id}"
            )
            broadcast_traffic_caption_status(track_id, "error", "Busy, try again shortly")
            return

    if traffic_caption_lock.locked():
        broadcast_traffic_caption_status(track_id, "error", "Busy, try again shortly")
        return

    entry["caption_pending"] = True
    broadcast_traffic_caption_status(track_id, "pending")

    async with traffic_caption_lock:
        try:
            mode = entry["mode"]
            crop = entry["best_crop"]
            if mode == "zone" and entry["track"].best_crop is not None:
                crop = entry["track"].best_crop
            if crop is None or crop.size == 0:
                entry["caption_pending"] = False
                broadcast_traffic_caption_status(track_id, "error", "No image of this vehicle")
                return
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            caption = await traffic_vlm_service.analyze_image(image, prompt=traffic_caption_prompts[mode])
            if caption.startswith("Error:"):
                raise RuntimeError(caption)
            entry["caption"] = caption
            entry["caption_pending"] = False
            broadcast_traffic_caption_status(track_id, "done", caption)
        except Exception as e:
            logger.error(f"[traffic] Caption failed for {track_id}: {e}")
            traffic_caption_last_failure_at = time.time()
            entry["caption_pending"] = False
            broadcast_traffic_caption_status(track_id, "error", "Description failed")


def traffic_list_detection_models():
    """Selectable detection models, one per model and compute unit, e.g.
    yolov8n_dla0_fp16.engine -> "YOLOv8n · NPU (DLA 0)", yolov8n_gpu_fp16.engine
    -> "YOLOv8n · GPU", yolov8n.onnx -> "YOLOv8n · CPU". Listed per model in
    NPU, GPU, CPU order, the default model first."""
    unit_order = {"npu": 0, "gpu": 1, "cpu": 2}
    models = []
    for path in [*TRAFFIC_MODELS_DIR.glob("*.engine"), *TRAFFIC_MODELS_DIR.glob("*.onnx")]:
        m = re.match(r"(yolo)(v?\d+)([nsmlx])", path.stem, re.IGNORECASE)
        name = f"YOLO{m.group(2)}{m.group(3).lower()}" if m else path.stem
        dla = re.search(r"dla(\d)", path.stem, re.IGNORECASE)
        unit = "cpu" if path.suffix == ".onnx" else "npu" if dla else "gpu"
        label = {"npu": f"NPU (DLA {dla.group(1) if dla else ''})", "gpu": "GPU", "cpu": "CPU"}[unit]
        is_default_model = path.stem.split("_")[0] == TRAFFIC_MODEL_PATH.stem.split("_")[0]
        models.append({"id": path.name, "name": f"{name} \u00b7 {label}",
                       "_key": (not is_default_model, name, unit_order[unit])})
    models.sort(key=lambda m: m.pop("_key"))
    return models


def traffic_get_detector(model_id: str) -> YoloDetector:
    """Load (once) and return the detector for a models/ file. Blocking - call
    via run_in_executor."""
    if model_id not in traffic_detectors:
        traffic_detectors[model_id] = YoloDetector(
            model_path=str(TRAFFIC_MODELS_DIR / model_id),
            labels_path=str(TRAFFIC_LABELS_PATH),
        )
    return traffic_detectors[model_id]


def _video_files(folder: Path):
    if not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS]


def traffic_list_videos():
    """Videos selectable as the /traffic source: the demo videos in
    demo_assets/, then videos uploaded from the settings drawer
    (demo_assets/uploads/ - kept apart from the main demo's upload folder)."""
    videos = [
        {"id": str(p), "name": TRAFFIC_VIDEO_NAMES.get(p.name, p.stem.replace("-", " ").capitalize())}
        for p in _video_files(TRAFFIC_VIDEOS_DIR)
    ]
    videos += [{"id": str(p), "name": f"{p.name} (uploaded)"} for p in _video_files(TRAFFIC_UPLOADS_DIR)]
    return videos


async def traffic_upload(request):
    """POST /api/traffic/upload (multipart field "file") - add a video to the
    /traffic list, saved under its own (sanitized) name in demo_assets/uploads/."""
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response({"error": "Expected a multipart field named 'file'"}, status=400)
    original = field.filename or "video.mp4"
    stem, ext = os.path.splitext(os.path.basename(original))
    if ext.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return web.json_response(
            {"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"}, status=400
        )
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:60] or "video"
    TRAFFIC_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path, n = TRAFFIC_UPLOADS_DIR / f"{stem}{ext.lower()}", 2
    while path.exists():  # never overwrite an earlier upload
        path, n = TRAFFIC_UPLOADS_DIR / f"{stem}-{n}{ext.lower()}", n + 1
    size = 0
    with open(path, "wb") as f:
        while chunk := await field.read_chunk(size=1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_UPLOAD_BYTES:
                f.close()
                path.unlink()
                return web.json_response({"error": "File too large (max 2GB)"}, status=413)
            f.write(chunk)
    logger.info(f"[traffic] Video uploaded: {path} ({size} bytes)")
    return web.json_response({"id": str(path), "name": f"{path.name} (uploaded)"})


async def traffic_settings(request):
    """GET /api/traffic/settings - options + current values for the settings drawer."""
    vlm_models = []
    if traffic_vlm_service is not None:
        try:
            listed = await traffic_vlm_service.client.models.list()
            vlm_models = sorted({m.id for m in listed.data})
        except Exception as e:
            logger.warning(f"[traffic] Could not list VLM models: {e}")
        if traffic_vlm_service.model not in vlm_models:
            vlm_models.insert(0, traffic_vlm_service.model)
    return web.json_response(
        {
            "detection_models": traffic_list_detection_models(),
            "detection_model": traffic_detection_model_id,
            "vlm_models": [{"id": m, "name": m.split("/")[-1]} for m in vlm_models],
            "vlm_model": traffic_vlm_service.model if traffic_vlm_service else None,
            "videos": traffic_list_videos(),
            "default_video": str(TRAFFIC_VIDEO_PATH),
        }
    )


async def traffic_set_detection_model(model_id: str):
    """Switch the active detection model (settings drawer), live if streaming."""
    global yolo_detector, traffic_detection_model_id
    if model_id not in {m["id"] for m in traffic_list_detection_models()}:
        return
    try:
        detector = await asyncio.get_event_loop().run_in_executor(None, traffic_get_detector, model_id)
    except Exception as e:
        logger.error(f"[traffic] Failed to load detection model {model_id}: {e}")
        message = {"type": "settings_error", "text": f"Could not load {model_id}"}
    else:
        yolo_detector, traffic_detection_model_id = detector, model_id
        if traffic_active_processor_track is not None:
            traffic_active_processor_track.detector = detector
        logger.info(f"[traffic] Detection model set to {model_id}")
        message = {"type": "settings_updated", "detection_model": model_id}
    for ws in list(traffic_websockets):
        asyncio.create_task(ws.send_json(message))


async def traffic_websocket_handler(request):
    """WebSocket for the /traffic demo - detection results, plus pause/resume
    control of the video source (see VideoFileTrack.pause/resume) and
    editing the click-to-caption prompt. No per-session prompt/model
    machinery like the main demo's /ws; that's the normal demo's concern."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    traffic_websockets.add(ws)
    logger.info(f"Traffic WebSocket client connected, total: {len(traffic_websockets)}")

    try:
        await ws.send_json(
            {
                "type": "status",
                "text": "Connected to traffic demo",
                "prompt": traffic_caption_prompts[traffic_prompt_mode()],
                "default_prompt": TRAFFIC_CAPTION_DEFAULTS[traffic_prompt_mode()],
            }
        )
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                msg_type = data.get("type")
                if msg_type in ("pause", "resume") and traffic_active_video_track:
                    getattr(traffic_active_video_track, msg_type)()
                elif msg_type == "toggle_detection" and traffic_active_processor_track:
                    if data.get("enabled"):
                        traffic_active_processor_track.enable_detection()
                    else:
                        traffic_active_processor_track.disable_detection()
                elif msg_type == "request_caption":
                    track_id = data.get("id")
                    if track_id:
                        asyncio.create_task(caption_traffic_vehicle(track_id))
                elif msg_type == "set_detection_model":
                    asyncio.create_task(traffic_set_detection_model(data.get("id", "")))
                elif msg_type == "set_vlm_model" and traffic_vlm_service is not None:
                    model_id = (data.get("id") or "").strip()
                    if model_id:
                        traffic_vlm_service.model = model_id
                        logger.info(f"[traffic] VLM model set to {model_id}")
                        for other_ws in list(traffic_websockets):
                            asyncio.create_task(
                                other_ws.send_json({"type": "settings_updated", "vlm_model": model_id})
                            )
                elif msg_type == "update_prompt":
                    new_prompt = (data.get("prompt") or "").strip()
                    if new_prompt:
                        mode = traffic_prompt_mode()
                        traffic_caption_prompts[mode] = new_prompt
                        logger.info(f"[traffic] Caption prompt ({mode}) updated: {new_prompt!r}")
                        for other_ws in list(traffic_websockets):
                            try:
                                asyncio.create_task(
                                    other_ws.send_json(
                                        {"type": "prompt_updated", "prompt": new_prompt}
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Error broadcasting prompt update: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"Traffic WebSocket error: {ws.exception()}")
    finally:
        traffic_websockets.discard(ws)
        logger.info(f"Traffic WebSocket client disconnected, total: {len(traffic_websockets)}")

    return ws


async def traffic_offer(request):
    """Handle WebRTC offer for the /traffic cascade demo, wrapping the chosen
    source video with object detection. The video (settings drawer) is any
    file from traffic_list_videos(), looped (default: car-highway.mp4).
    Kept deliberately separate from the normal demo's offer()/VLM flow (see
    CLAUDE.md)."""
    if yolo_detector is None:
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps(
                {"error": "Detection model not available - check server startup logs"}
            ),
        )

    global traffic_active_pc, traffic_active_video_track, traffic_active_processor_track, traffic_last_offer_time

    # Rate-limit connection setup itself (see TRAFFIC_MIN_OFFER_INTERVAL_SECONDS
    # comment above) - protects the server even if a client hammers this
    # endpoint (e.g. a stale tab stuck retrying every few seconds).
    now = time.monotonic()
    elapsed = now - traffic_last_offer_time
    if elapsed < TRAFFIC_MIN_OFFER_INTERVAL_SECONDS:
        await asyncio.sleep(TRAFFIC_MIN_OFFER_INTERVAL_SECONDS - elapsed)
    traffic_last_offer_time = time.monotonic()

    # Close any existing /traffic connection first (see traffic_active_pc
    # comment above) - a page reload should cleanly supersede the old
    # connection, never run alongside it.
    #
    # NOTE: closing the old pc/video track synchronously here, immediately
    # before opening a new VideoFileTrack on the same file, was observed to
    # trigger a native heap-corruption crash ("munmap_chunk(): invalid
    # pointer") under rapid reconnect churn - almost certainly a PyAV/libav
    # thread-safety issue with tearing down and standing up decode contexts
    # for the same file back-to-back with no gap. Stopping the old track and
    # yielding back to the event loop before continuing gives its native
    # cleanup a chance to finish first.
    if traffic_active_pc is not None:
        logger.info("[traffic] New connection superseding existing one - closing old first")
        old_pc, old_track = traffic_active_pc, traffic_active_video_track
        traffic_active_pc, traffic_active_video_track, traffic_active_processor_track = None, None, None
        if old_track:
            old_track.stop()
        pcs.discard(old_pc)
        await old_pc.close()
        await asyncio.sleep(0.2)

    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    video_path = params.get("video") or str(TRAFFIC_VIDEO_PATH)
    if video_path not in {v["id"] for v in traffic_list_videos()}:
        return web.json_response({"error": "Unknown video file"}, status=400)

    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)
    traffic_active_pc = pc

    video_cleanup_track = None

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        global traffic_active_pc, traffic_active_video_track, traffic_active_processor_track
        logger.info(f"[traffic] Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            if video_cleanup_track:
                video_cleanup_track.stop()
            await pc.close()
            pcs.discard(pc)
            if traffic_active_pc is pc:
                traffic_active_pc = None
                traffic_active_video_track = None
                traffic_active_processor_track = None

    try:
        video_track = VideoFileTrack(video_path, loop=True)
        video_cleanup_track = video_track
        traffic_active_video_track = video_track
    except Exception as e:
        logger.error(f"[traffic] Failed to open demo video: {e}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": f"Failed to open demo video: {str(e)}"}),
        )

    # buffered=False: keep only the latest frame, don't queue every frame the
    # relay pulls. DetectionVideoTrack does real per-frame work (DLA/GPU
    # inference + drawing + re-encoding) that can run slightly behind 30fps,
    # and the default buffered=True relay queue is unbounded - a backlog of
    # stale frames would build up over time, and pause() would then have to
    # drain that whole backlog (showing "old" motion) before actually
    # freezing. Dropping stale frames instead is also just correct for a
    # real-time low-latency pipeline like this one.
    relayed_video = relay.subscribe(video_track, buffered=False)
    processor_track = DetectionVideoTrack(
        relayed_video, yolo_detector, detection_callback=broadcast_traffic_detections
    )
    processor_track.set_entry_zone(TRAFFIC_ENTRY_ZONES.get(Path(video_path).name))
    if processor_track.entry_zone is not None:
        # On a loop, the last car of one pass and the first car of the next sit
        # in the same spot at the barrier - the tracker would take them for one
        # car (already logged) and skip the new one. Videos without a zone (the
        # highway) keep their original behaviour.
        video_track.on_loop = processor_track.reset_tracker
    traffic_active_processor_track = processor_track
    pc.addTrack(processor_track)

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("[traffic] Created answer for traffic demo peer connection")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def models(request):
    """Return available models from the VLM API"""
    try:
        # Check if custom API base and key are provided in query params
        api_base = request.rel_url.query.get("api_base")
        api_key = request.rel_url.query.get("api_key")

        if api_base:
            # Query models from the provided API endpoint
            from openai import AsyncOpenAI

            temp_client = AsyncOpenAI(base_url=api_base, api_key=api_key if api_key else "EMPTY")
            models_response = await temp_client.models.list()
            # Mark the server's --model as current - otherwise app.js auto-picks
            # the first entry, and Ollama lists the newest model first (e.g. the
            # Hermes agent's gemma4-orin, which then OOMs on video frames).
            default_model = get_or_create_session("default")["vlm_service"].model
            models_list = [
                {"id": model.id, "name": model.id, "current": model.id == default_model}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
        else:
            # Use default session's VLM service (backwards compat when no api_base in query)
            default_svc = get_or_create_session("default")["vlm_service"]
            models_response = await default_svc.client.models.list()
            models_list = [
                {"id": model.id, "name": model.id, "current": model.id == default_svc.model}
                for model in models_response.data
            ]
            return web.Response(
                content_type="application/json", text=json.dumps({"models": models_list})
            )
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        # Return current model as fallback
        if sessions.get("default"):
            default_svc = sessions["default"]["vlm_service"]
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {
                        "models": [
                            {"id": default_svc.model, "name": default_svc.model, "current": True}
                        ]
                    }
                ),
            )
        return web.Response(
            content_type="application/json", text=json.dumps({"models": [], "error": str(e)})
        )


async def detect_services(request):
    """Detect available local VLM services"""
    services = [
        {"name": "Ollama", "url": "http://localhost:11434/v1", "port": 11434, "path": "/api/tags"},
        {"name": "vLLM", "url": "http://localhost:8000/v1", "port": 8000, "path": "/v1/models"},
        {"name": "SGLang", "url": "http://localhost:30000/v1", "port": 30000, "path": "/v1/models"},
    ]

    detected = []

    async def check_service(service):
        """Check if a service is running by probing its endpoint"""
        try:
            timeout = aiohttp.ClientTimeout(total=1.0)  # 1 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"http://localhost:{service['port']}{service['path']}"
                async with session.get(url) as response:
                    if response.status in [200, 404]:  # 404 is ok, means server is running
                        logger.info(f"Detected {service['name']} at {service['url']}")
                        return service
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        return None

    # Check all services concurrently
    results = await asyncio.gather(*[check_service(s) for s in services])
    detected = [s for s in results if s is not None]

    # Default to NVIDIA API Catalog if no local services found
    if not detected:
        detected.append(
            {
                "name": "NVIDIA API Catalog",
                "url": "https://integrate.api.nvidia.com/v1",
                "port": None,
                "path": None,
                "requires_key": True,
            }
        )

    return web.Response(
        content_type="application/json",
        text=json.dumps({"detected": detected, "default": detected[0] if detected else None}),
    )


async def websocket_handler(request):
    """Handle WebSocket connections for text updates. Supports ?session_id= for multi-session."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Session ID from query or generate new (client should send same id in /offer)
    session_id = request.query.get("session_id", "").strip() or str(uuid.uuid4())
    ws_to_session[ws] = session_id
    session_websockets[session_id].add(ws)
    websockets.add(ws)
    logger.info(
        f"WebSocket client connected. session_id={session_id}, total clients: {len(websockets)}"
    )

    session = get_or_create_session(session_id)
    svc = session["vlm_service"]

    try:
        # Send initial message with current server configuration (include session_id if we generated it)
        await ws.send_json(
            {
                "type": "status",
                "text": "Connected to server",
                "status": "Ready",
                "session_id": session_id,
            }
        )

        # Send current server configuration for this session
        from .video_processor import VideoProcessorTrack as _VPT

        await ws.send_json(
            {
                "type": "server_config",
                "model": svc.model,
                "api_base": svc.api_base,
                "prompt": svc.prompt,
                "reasoning_effort": svc.reasoning_effort,
                "max_tokens": svc.max_tokens,
                "process_every": _VPT.process_every_n_frames,
                "session_id": session_id,
            }
        )

        # Keep connection alive and handle incoming messages
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Re-resolve session in case it was recreated
                    svc = get_or_create_session(session_id)["vlm_service"]

                    if data.get("type") == "update_prompt":
                        new_prompt = data.get("prompt", "").strip()
                        max_tokens = data.get("max_tokens")
                        if new_prompt and svc:
                            svc.update_prompt(new_prompt, max_tokens)
                            logger.info(
                                f"[{session_id}] Prompt updated: {new_prompt}, max_tokens: {max_tokens}"
                            )

                            await ws.send_json(
                                {
                                    "type": "prompt_updated",
                                    "prompt": new_prompt,
                                    "max_tokens": max_tokens,
                                }
                            )

                    elif data.get("type") == "update_reasoning":
                        effort = data.get("reasoning_effort")
                        reply = {"type": "reasoning_updated"}
                        if effort not in (None, "none", "low", "medium", "high"):
                            reply["error"] = "Unsupported thinking setting"
                        else:
                            svc.reasoning_effort = effort
                            logger.info(f"[{session_id}] Reasoning effort updated: {effort}")
                        reply["reasoning_effort"] = svc.reasoning_effort
                        await ws.send_json(reply)

                    elif data.get("type") == "update_model":
                        new_model = data.get("model", "").strip()
                        api_base = data.get("api_base", "").strip()
                        api_key = data.get("api_key", "").strip()

                        if new_model and svc:
                            svc.model = new_model
                            if api_base:
                                svc.update_api_settings(api_base, api_key if api_key else None)
                                logger.info(
                                    f"[{session_id}] Model updated: {new_model}, API: {api_base}"
                                )
                            else:
                                logger.info(f"[{session_id}] Model updated: {new_model}")

                            await ws.send_json(
                                {
                                    "type": "model_updated",
                                    "model": new_model,
                                    "api_base": svc.api_base,
                                }
                            )

                    elif data.get("type") == "update_processing":
                        process_every = data.get("process_every", 30)
                        try:
                            process_every = int(process_every)
                            if 1 <= process_every <= 3600:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.process_every_n_frames
                                VideoProcessorTrack.process_every_n_frames = process_every
                                logger.info(
                                    f"[{session_id}] Processing interval updated: {old_value} → {process_every} frames"
                                )

                                await ws.send_json(
                                    {"type": "processing_updated", "process_every": process_every}
                                )
                            else:
                                logger.warning(
                                    f"Processing interval out of range (1-3600): {process_every}"
                                )
                        except ValueError:
                            logger.error(f"Invalid processing interval: {process_every}")

                    elif data.get("type") == "set_debug":
                        session_data = get_or_create_session(session_id)
                        if "show_request_payload" in data:
                            session_data["show_request_payload"] = bool(
                                data["show_request_payload"]
                            )
                        if "show_response_payload" in data:
                            session_data["show_response_payload"] = bool(
                                data["show_response_payload"]
                            )
                        logger.debug(
                            f"[{session_id}] Debug: request_payload="
                            f"{session_data.get('show_request_payload')}, response_payload="
                            f"{session_data.get('show_response_payload')}"
                        )

                    elif data.get("type") == "update_max_latency":
                        max_latency = data.get("max_latency", 0.0)
                        try:
                            max_latency = float(max_latency)
                            if 0 <= max_latency <= 10.0:
                                from .video_processor import VideoProcessorTrack

                                old_value = VideoProcessorTrack.max_frame_latency
                                VideoProcessorTrack.max_frame_latency = max_latency
                                status = "disabled" if max_latency == 0 else f"{max_latency:.1f}s"
                                old_status = "disabled" if old_value == 0 else f"{old_value:.1f}s"
                                logger.info(
                                    f"[{session_id}] Max frame latency updated: {old_status} → {status}"
                                )

                                await ws.send_json(
                                    {"type": "max_latency_updated", "max_latency": max_latency}
                                )
                            else:
                                logger.warning(f"Max latency out of range (0-10.0): {max_latency}")
                        except ValueError:
                            logger.error(f"Invalid max latency value: {max_latency}")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from client")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
    finally:
        session_websockets[session_id].discard(ws)
        ws_to_session.pop(ws, None)
        websockets.discard(ws)
        logger.info(
            f"WebSocket client disconnected. session_id={session_id}, total clients: {len(websockets)}"
        )

    return ws


def broadcast_text_update(text: str, metrics: dict):
    """Broadcast text update and metrics to all connected WebSocket clients"""
    if not websockets:
        return

    message = json.dumps({"type": "vlm_response", "text": text, "metrics": metrics})

    # Send to all connected clients
    dead_websockets = set()
    for ws in websockets:
        try:
            # Use asyncio to send without blocking
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending to websocket: {e}")
            dead_websockets.add(ws)

    # Clean up dead connections
    websockets.difference_update(dead_websockets)


def broadcast_gpu_stats(stats: dict):
    """Broadcast GPU stats to all connected WebSocket clients (main demo and /traffic)"""
    if not websockets and not traffic_websockets:
        return

    message = json.dumps({"type": "gpu_stats", "stats": stats})

    for ws in list(traffic_websockets):
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending GPU stats to traffic websocket: {e}")

    # Send to all connected clients
    dead_websockets = set()
    for ws in websockets:
        try:
            asyncio.create_task(ws.send_str(message))
        except Exception as e:
            logger.error(f"Error sending GPU stats to websocket: {e}")
            dead_websockets.add(ws)

    # Clean up dead connections
    websockets.difference_update(dead_websockets)


async def gpu_monitor_loop():
    """Background task to periodically collect and broadcast GPU stats"""
    global gpu_monitor

    if not gpu_monitor:
        logger.warning("GPU monitor not initialized, skipping monitoring")
        return

    logger.info("GPU monitoring loop started")

    try:
        while True:
            # Get current stats
            stats = gpu_monitor.get_stats()

            # Update history with current stats
            gpu_monitor.update_history(stats)

            # Add history to stats
            stats["history"] = gpu_monitor.get_history()

            # Broadcast to all connected clients
            broadcast_gpu_stats(stats)

            # Update every 0.25 seconds for detailed GPU monitoring
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        logger.info("GPU monitoring loop cancelled")
    except Exception as e:
        logger.error(f"Error in GPU monitoring loop: {e}")


async def offer(request):
    """Handle WebRTC offer from client (supports both webcam and RTSP). Uses session_id for per-session VLM."""
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    rtsp_url = params.get("rtsp_url")  # Optional RTSP URL for IP camera mode
    video_file_path = params.get("video_file_path")  # Optional local video file mode
    use_orbbec = params.get("orbbec")  # Optional Orbbec depth camera Color stream
    usb_camera = params.get("usb_camera")  # Optional server-side USB webcam, e.g. "/dev/video8"
    session_id = params.get("session_id", "default")

    session = get_or_create_session(session_id)
    session_vlm = session["vlm_service"]
    session_callback = get_session_callback(session_id)

    # Create RTCPeerConnection with STUN servers for Docker/NAT compatibility
    config = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        ]
    )
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    # Store RTSP track for cleanup
    rtsp_cleanup_track = None

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            # Clean up RTSP track if exists
            if rtsp_cleanup_track:
                rtsp_cleanup_track.stop()
                logger.info("RTSP track stopped on connection close")
            await pc.close()
            pcs.discard(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            logger.error("ICE connection failed - check firewall/NAT settings")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state: {pc.iceGatheringState}")

    # If RTSP URL provided, create RTSP track instead of waiting for browser track
    if rtsp_url:
        logger.info(f"[{session_id}] Creating RTSP track for: {rtsp_url}")
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
            rtsp_cleanup_track = rtsp_track  # Store for cleanup

            # Wait for initial connection to get stream info
            await asyncio.sleep(0.5)

            # Wrap RTSP track with relay first (same pattern as webcam)
            relayed_rtsp = relay.subscribe(rtsp_track)

            processor_track = VideoProcessorTrack(
                relayed_rtsp, session_vlm, text_callback=session_callback
            )

            # Add processor directly to peer connection
            pc.addTrack(processor_track)
            logger.info("Added RTSP processor track to peer connection")

        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )
    elif video_file_path:
        logger.info(f"[{session_id}] Creating video file track for: {video_file_path}")

        if not os.path.isfile(video_file_path):
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": f"Video file not found: {video_file_path}"}),
            )

        try:
            video_track = VideoFileTrack(video_file_path, loop=True)
            rtsp_cleanup_track = video_track  # Same cleanup path as RTSP - just needs .stop()

            relayed_video = relay.subscribe(video_track)

            processor_track = VideoProcessorTrack(
                relayed_video, session_vlm, text_callback=session_callback
            )

            pc.addTrack(processor_track)
            logger.info("Added video file processor track to peer connection")

        except Exception as e:
            logger.error(f"Failed to create video file track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to open video file: {str(e)}"}),
            )
    elif use_orbbec:
        logger.info(f"[{session_id}] Creating Orbbec Color track")
        try:
            manager = get_orbbec_manager()
            manager.start()

            orbbec_track = OrbbecColorTrack(manager)
            rtsp_cleanup_track = orbbec_track  # Stops this session's track wrapper, not the shared device

            relayed_orbbec = relay.subscribe(orbbec_track)

            processor_track = VideoProcessorTrack(
                relayed_orbbec, session_vlm, text_callback=session_callback
            )

            pc.addTrack(processor_track)
            logger.info("Added Orbbec Color processor track to peer connection")

        except OrbbecDeviceNotFoundError as e:
            logger.error(f"Orbbec device not found: {e}")
            return web.Response(
                status=404,
                content_type="application/json",
                text=json.dumps({"error": str(e)}),
            )
        except Exception as e:
            logger.error(f"Failed to create Orbbec track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to start Orbbec camera: {str(e)}"}),
            )
    elif usb_camera:
        logger.info(f"[{session_id}] Creating USB camera track for {usb_camera}")
        try:
            usb_track = UsbCameraTrack(get_usb_camera(usb_camera))
            rtsp_cleanup_track = usb_track  # stop() releases the shared device

            processor_track = VideoProcessorTrack(
                relay.subscribe(usb_track), session_vlm, text_callback=session_callback
            )
            pc.addTrack(processor_track)
            logger.info("Added USB camera processor track to peer connection")

        except Exception as e:
            logger.error(f"Failed to create USB camera track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to start USB camera: {str(e)}"}),
            )
    else:
        # Webcam mode: wait for browser to send track
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")

            if track.kind == "video":
                # Create processor track with this session's VLM and session-scoped callback
                processor_track = VideoProcessorTrack(
                    relay.subscribe(track), session_vlm, text_callback=session_callback
                )

                # Add processed track back to connection
                pc.addTrack(processor_track)
                logger.info("Added processed video track back to peer connection")

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.kind} ended")

    # Handle offer
    await pc.setRemoteDescription(offer_sdp)

    # Create answer - this must happen after tracks are added
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"Created answer with {len(pc.getTransceivers())} transceivers")

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


# Video files are typically much larger than the JSON/SDP bodies this server
# otherwise handles - cap uploads at 2GB to avoid accidentally filling disk
VIDEO_UPLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache", "live_vlm_webui", "uploaded_videos")
MAX_VIDEO_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


async def upload_video_file(request):
    """
    Accept a video file upload (multipart/form-data) and save it to disk so
    a VideoFileTrack can open it. Returns the saved path for use as
    video_file_path in a subsequent /offer request.

    POST /api/video/upload
    """
    try:
        os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)

        reader = await request.multipart()
        field = await reader.next()

        if field is None or field.name != "file":
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Expected a multipart field named 'file'"}),
            )

        original_name = field.filename or "upload"
        ext = os.path.splitext(original_name)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps(
                    {
                        "error": f"Unsupported file type '{ext}'. Allowed: "
                        f"{', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
                    }
                ),
            )

        # Unique filename so concurrent/repeated uploads never collide
        saved_name = f"{uuid.uuid4().hex}{ext}"
        saved_path = os.path.join(VIDEO_UPLOAD_DIR, saved_name)

        bytes_written = 0
        with open(saved_path, "wb") as f:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_VIDEO_UPLOAD_BYTES:
                    f.close()
                    os.remove(saved_path)
                    return web.Response(
                        status=413,
                        content_type="application/json",
                        text=json.dumps({"error": "File too large (max 2GB)"}),
                    )
                f.write(chunk)

        logger.info(f"Video file uploaded: {saved_path} ({bytes_written} bytes)")

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {
                    "video_file_path": saved_path,
                    "original_filename": original_name,
                    "size_bytes": bytes_written,
                }
            ),
        )

    except Exception as e:
        logger.error(f"Error uploading video file: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_start(request):
    """
    Start RTSP stream processing.

    Accepts RTSP URL and creates a video processing pipeline.

    POST /api/rtsp/start
    Body: {"rtsp_url": "rtsp://...", "session_id": "optional-id"}
    """
    try:
        data = await request.json()
        rtsp_url = data.get("rtsp_url")
        session_id = data.get("session_id", "default")

        if not rtsp_url:
            logger.warning("RTSP start request missing rtsp_url")
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"error": "Missing rtsp_url parameter"}),
            )

        # Check if session already exists
        if session_id in rtsp_tracks:
            logger.warning(f"RTSP session {session_id} already exists, stopping it first")
            await _stop_rtsp_session(session_id)

        logger.info(f"Starting RTSP stream for session {session_id}")

        # Create RTSP video track
        try:
            rtsp_track = RTSPVideoTrack(rtsp_url)
        except Exception as e:
            logger.error(f"Failed to create RTSP track: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Failed to connect to RTSP stream: {str(e)}"}),
            )

        # Create processor track with this session's VLM and session-scoped callback
        session = get_or_create_session(session_id)
        session_vlm = session["vlm_service"]
        session_callback = get_session_callback(session_id)
        processor_track = VideoProcessorTrack(
            rtsp_track, session_vlm, text_callback=session_callback
        )

        # Start background task to consume frames
        async def consume_frames():
            """Background task to continuously pull frames from processor track"""
            try:
                while not rtsp_track._stopped:
                    try:
                        _ = await processor_track.recv()
                        # Frame is processed, just discard it (VLM analysis happens in recv())
                    except StopAsyncIteration:
                        logger.info(f"RTSP stream {session_id} ended")
                        break
                    except Exception as e:
                        logger.error(f"Error consuming RTSP frame for {session_id}: {e}")
                        break
            finally:
                logger.info(f"Frame consumption stopped for {session_id}")

        frame_task = asyncio.create_task(consume_frames())

        # Store reference with frame task
        rtsp_tracks[session_id] = (rtsp_track, processor_track, frame_task)

        # Get stream stats
        stats = rtsp_track.get_stats()

        logger.info(
            f"RTSP stream started: {session_id} - {stats.get('codec')} "
            f"{stats.get('width')}x{stats.get('height')}"
        )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "started", "session_id": session_id, "stream_info": stats}),
        )

    except Exception as e:
        logger.error(f"Error starting RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_stop(request):
    """
    Stop RTSP stream processing.

    POST /api/rtsp/stop
    Body: {"session_id": "optional-id"}
    """
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")

        await _stop_rtsp_session(session_id)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"status": "stopped", "session_id": session_id}),
        )

    except Exception as e:
        logger.error(f"Error stopping RTSP: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def rtsp_status(request):
    """
    Get status of all RTSP streams.

    GET /api/rtsp/status
    """
    try:
        status_list = []

        for session_id, (rtsp_track, processor_track, frame_task) in rtsp_tracks.items():
            stats = rtsp_track.get_stats()
            status_list.append(
                {
                    "session_id": session_id,
                    "connected": stats.get("connected"),
                    "frames_received": stats.get("frames_received"),
                    "stream_info": {
                        "codec": stats.get("codec"),
                        "width": stats.get("width"),
                        "height": stats.get("height"),
                        "fps": stats.get("fps"),
                    },
                }
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({"active_streams": len(rtsp_tracks), "streams": status_list}),
        )

    except Exception as e:
        logger.error(f"Error getting RTSP status: {e}", exc_info=True)
        return web.Response(
            status=500, content_type="application/json", text=json.dumps({"error": str(e)})
        )


async def _stop_rtsp_session(session_id: str):
    """Helper function to stop an RTSP session"""
    if session_id in rtsp_tracks:
        rtsp_track, processor_track, frame_task = rtsp_tracks[session_id]

        # Cancel frame consumption task
        if frame_task and not frame_task.done():
            frame_task.cancel()
            try:
                await frame_task
            except asyncio.CancelledError:
                pass

        # Stop tracks
        try:
            processor_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping processor track: {e}")

        try:
            rtsp_track.stop()
        except Exception as e:
            logger.warning(f"Error stopping RTSP track: {e}")

        # Remove from tracking
        del rtsp_tracks[session_id]
        logger.info(f"RTSP stream stopped: {session_id}")
    else:
        logger.warning(f"RTSP session {session_id} not found")


async def on_startup(app):
    """Initialize resources on server startup"""
    global gpu_monitor, gpu_monitor_task, yolo_detector, traffic_vlm_service

    # Initialize GPU monitor
    try:
        gpu_monitor = create_monitor()
        # DLA busy % for /traffic's telemetry: the detector's own inference time
        # on the DLA (jtop has no DLA utilization counter), see YoloDetector.
        gpu_monitor.dla_busy_source = lambda: YoloDetector.dla_busy_seconds
        logger.info("GPU monitor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GPU monitor: {e}")
        gpu_monitor = None

    # Start GPU monitoring background task
    if gpu_monitor:
        gpu_monitor_task = asyncio.create_task(gpu_monitor_loop())
        logger.info("GPU monitoring task started")

    # Load the /traffic cascade demo's detection model once at startup. A
    # missing model/video only disables /traffic; it must never take down the
    # normal demo.
    if TRAFFIC_MODEL_PATH.exists() and TRAFFIC_LABELS_PATH.exists():
        try:
            yolo_detector = traffic_get_detector(TRAFFIC_MODEL_PATH.name)
            logger.info("Traffic detection model loaded - /traffic demo ready")
        except Exception as e:
            logger.error(f"Failed to load traffic detection model: {e}")
            yolo_detector = None
    else:
        logger.warning(
            f"Traffic detection model not found at {TRAFFIC_MODEL_PATH} - "
            "/traffic demo will be unavailable until it's built (see scripts/build_traffic_engines.sh)"
        )
        yolo_detector = None

    # Dedicated VLM client for /traffic click-to-caption - same Ollama server
    # as the main demo (default_vlm_config), but its own instance so its
    # lock/busy state never interferes with the main demo's.
    try:
        traffic_vlm_service = VLMService(
            model=default_vlm_config.get("model", "meta/llama-3.2-11b-vision-instruct"),
            api_base=default_vlm_config.get("api_base", "http://localhost:8000/v1"),
            api_key=default_vlm_config.get("api_key", "EMPTY"),
            prompt=TRAFFIC_CAPTION_DEFAULT_PROMPT,
            max_tokens=TRAFFIC_CAPTION_MAX_TOKENS,
        )
        logger.info("Traffic caption VLM service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize traffic caption VLM service: {e}")
        traffic_vlm_service = None


async def on_shutdown(app):
    """Cleanup on server shutdown"""
    global gpu_monitor, gpu_monitor_task

    logger.info("Shutting down server...")

    # Stop GPU monitoring task
    if gpu_monitor_task:
        gpu_monitor_task.cancel()
        try:
            await gpu_monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("GPU monitoring task stopped")

    # Cleanup GPU monitor
    if gpu_monitor:
        gpu_monitor.cleanup()
        logger.info("GPU monitor cleaned up")

    # Release the Orbbec camera, if it was ever opened
    try:
        get_orbbec_manager().stop()
    except Exception as e:
        logger.warning(f"Error stopping Orbbec camera: {e}")

    # Close all websockets and clear session state
    for ws in list(websockets):
        await ws.close()
    websockets.clear()
    session_websockets.clear()
    ws_to_session.clear()

    # Close /traffic demo websockets
    for ws in list(traffic_websockets):
        await ws.close()
    traffic_websockets.clear()

    # Close all RTSP streams
    for session_id in list(rtsp_tracks.keys()):
        await _stop_rtsp_session(session_id)
    logger.info("RTSP streams closed")

    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    logger.info("Cleanup complete")


async def create_app(test_mode=False):
    """
    Create and configure the aiohttp web application.

    Args:
        test_mode: If True, skip GPU monitoring and use test configuration

    Returns:
        Configured web.Application instance
    """
    # Create web application
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/style.css", app_style)
    app.router.add_get("/app.js", app_script)
    app.router.add_get("/models", models)
    app.router.add_get("/detect-services", detect_services)
    app.router.add_get("/api/platform", platform_info)
    # /traffic cascade demo - separate page/pipeline from the normal demo above
    app.router.add_get("/traffic", traffic_index)
    app.router.add_get("/compare", compare_index)
    app.router.add_get("/guide", guide_index)
    app.router.add_get("/traffic.css", traffic_style)
    app.router.add_get("/traffic.js", traffic_script)
    app.router.add_get("/api/traffic/ws", traffic_websocket_handler)
    app.router.add_post("/api/traffic/offer", traffic_offer)
    app.router.add_get("/api/traffic/settings", traffic_settings)
    app.router.add_post("/api/traffic/upload", traffic_upload)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/offer", offer)

    # RTSP endpoints
    app.router.add_post("/api/rtsp/start", rtsp_start)
    app.router.add_post("/api/rtsp/stop", rtsp_stop)
    app.router.add_get("/api/rtsp/status", rtsp_status)

    # Local video file upload/playback endpoint
    app.router.add_post("/api/video/upload", upload_video_file)

    # Orbbec depth camera endpoints
    app.router.add_get("/depth", depth_page)
    app.router.add_get("/depth.css", depth_style)
    app.router.add_get("/depth.js", depth_script)
    app.router.add_get("/ws/orbbec", orbbec_ws)
    app.router.add_get("/api/orbbec/ready", orbbec_ready)

    # Server-side USB webcam
    app.router.add_get("/api/usbcam/list", usb_camera_list)

    # Serve static files (images, etc.)
    # Always serve from static/images within the package (works for both pip and dev installs)
    images_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    images_dir = os.path.abspath(images_dir)

    if os.path.exists(images_dir):
        app.router.add_static("/images", images_dir, name="images")
        logger.info(f"Serving static files from: {images_dir}")
    else:
        logger.warning(f"⚠️  Static images directory not found: {images_dir}")

    # Serve self-hosted vendor JS libraries (lucide, marked, dompurify, motion)
    vendor_dir = os.path.join(os.path.dirname(__file__), "static", "vendor")
    vendor_dir = os.path.abspath(vendor_dir)

    if os.path.exists(vendor_dir):
        app.router.add_static("/vendor", vendor_dir, name="vendor")
        logger.info(f"Serving vendored JS libraries from: {vendor_dir}")
    else:
        logger.warning(f"⚠️  Vendor JS directory not found: {vendor_dir}")

    # Serve favicon files
    favicon_dir = os.path.join(os.path.dirname(__file__), "static", "favicon")
    favicon_dir = os.path.abspath(favicon_dir)

    if os.path.exists(favicon_dir):
        app.router.add_static("/favicon", favicon_dir, name="favicon")
        logger.info(f"Serving favicon files from: {favicon_dir}")
    else:
        logger.warning(f"⚠️  Favicon directory not found: {favicon_dir}")

    if not test_mode:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)

    return app


def get_app_config_dir():
    """Get the application config directory following OS conventions"""
    import os
    from pathlib import Path

    # Follow XDG Base Directory spec on Linux, use OS-appropriate paths elsewhere
    if os.name == "posix":
        if "darwin" in os.sys.platform.lower():
            # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "live-vlm-webui"
        else:
            # Linux/Unix (including Jetson)
            config_dir = (
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "live-vlm-webui"
            )
    else:
        # Windows
        config_dir = Path(os.environ.get("APPDATA", Path.home())) / "live-vlm-webui"

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def generate_self_signed_cert(cert_path="cert.pem", key_path="key.pem"):
    """Generate a self-signed SSL certificate if it doesn't exist"""
    import os
    import subprocess

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return True

    logger.info("🔐 Generating self-signed SSL certificate...")
    logger.info(f"   Saving to: {os.path.dirname(os.path.abspath(cert_path)) or '.'}")
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:4096",
                "-nodes",
                "-out",
                cert_path,
                "-keyout",
                key_path,
                "-days",
                "365",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"✅ Generated {cert_path} and {key_path}")
        return True
    except FileNotFoundError:
        logger.warning("⚠️  openssl not found - cannot auto-generate certificates")
        logger.warning(
            "⚠️  Install openssl: sudo apt install openssl (Linux) or brew install openssl (Mac)"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️  Failed to generate certificates: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    import ssl

    from . import __version__

    parser = argparse.ArgumentParser(
        description="WebRTC Live VLM WebUI - Real-time vision model interaction",
        epilog="Examples:\n"
        "  vLLM:    python server.py --model llama-3.2-11b-vision-instruct --api-base http://localhost:8000/v1\n"
        "  SGLang:  python server.py --model llama-3.2-11b-vision-instruct --api-base http://localhost:30000/v1\n"
        "  Ollama:  python server.py --model llava:7b --api-base http://localhost:11434/v1\n"
        "  HTTPS:   python server.py --model llava:7b --api-base http://localhost:11434/v1 --ssl-cert cert.pem --ssl-key key.pem\n"
        "  Kiosk:   python server.py --localhost-http --model llava:7b --api-base http://localhost:11434/v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_prompt = os.environ.get(
        "LIVE_VLM_DEFAULT_PROMPT",
        "Describe what you see in this image in one sentence.",
    )
    try:
        default_max_tokens = int(os.environ.get("LIVE_VLM_MAX_TOKENS", "512"))
    except ValueError:
        logger.warning("Invalid LIVE_VLM_MAX_TOKENS value; using 512")
        default_max_tokens = 512
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind to (default: 8090)")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Automatically find available port if default is taken",
    )
    parser.add_argument(
        "--model", help="VLM model name (optional, will auto-detect if not specified)"
    )
    parser.add_argument(
        "--api-base", help="VLM API base URL (optional, will auto-detect or use NVIDIA NGC)"
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key - use 'EMPTY' for local servers, required for NVIDIA NGC/OpenAI (default: EMPTY)",
    )
    parser.add_argument(
        "--prompt",
        default=default_prompt,
        help="Prompt to send to VLM (default can be set with LIVE_VLM_DEFAULT_PROMPT)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=default_max_tokens,
        help="Maximum tokens to generate (default: 512, or LIVE_VLM_MAX_TOKENS)",
    )
    # Get default SSL cert paths (platform-specific)
    default_config_dir = get_app_config_dir()
    default_cert_path = str(default_config_dir / "cert.pem")
    default_key_path = str(default_config_dir / "key.pem")

    parser.add_argument("--process-every", type=int, default=30, help="Process every Nth frame")
    parser.add_argument(
        "--ssl-cert",
        default=None,  # Will be set to config dir if not specified
        help=f"Path to SSL certificate file (default: {default_cert_path}, auto-generated if missing)",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,  # Will be set to config dir if not specified
        help=f"Path to SSL private key file (default: {default_key_path}, auto-generated if missing)",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Disable SSL (webcam works on http://localhost; remote access requires HTTPS)",
    )
    parser.add_argument(
        "--localhost-http",
        action="store_true",
        help=(
            "Serve HTTP on 127.0.0.1 for same-device kiosk use. "
            "This avoids self-signed certificate prompts while keeping the browser on localhost."
        ),
    )

    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default=None,
        help="Optional reasoning setting for compatible backends; 'none' disables Ollama thinking",
    )

    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.localhost_http:
        args.host = "127.0.0.1"
        args.no_ssl = True

    # Cloud deployment: env overrides for default API base, model, and frame interval
    if os.environ.get("LIVE_VLM_API_BASE"):
        if not args.api_base:
            args.api_base = os.environ.get("LIVE_VLM_API_BASE").strip()
            logger.info(f"Using API base from env: {args.api_base}")
    if os.environ.get("LIVE_VLM_DEFAULT_MODEL"):
        if not args.model:
            args.model = os.environ.get("LIVE_VLM_DEFAULT_MODEL").strip()
            logger.info(f"Using default model from env: {args.model}")
    if os.environ.get("LIVE_VLM_PROCESS_EVERY"):
        try:
            args.process_every = int(os.environ.get("LIVE_VLM_PROCESS_EVERY"))
            logger.info(f"Using process_every from env: {args.process_every}")
        except ValueError:
            pass

    # Set default SSL cert paths to config directory if not specified
    if args.ssl_cert is None:
        config_dir = get_app_config_dir()
        args.ssl_cert = str(config_dir / "cert.pem")
    if args.ssl_key is None:
        config_dir = get_app_config_dir()
        args.ssl_key = str(config_dir / "key.pem")

    # Auto-detect service and model if not specified
    api_base = args.api_base
    model = args.model
    api_key = args.api_key

    if not model or not api_base:
        logger.info("No model/API specified, auto-detecting local services...")
        detected_api_base, detected_model = asyncio.run(detect_local_service_and_model())

        if detected_api_base and detected_model:
            if not api_base:
                api_base = detected_api_base
            if not model:
                model = detected_model
        else:
            # Fall back to NVIDIA NGC
            logger.warning("⚠️  No local VLM service found (Ollama, vLLM, SGLang)")
            logger.info("📡 Falling back to NVIDIA API Catalog")
            logger.info("   You'll need an API key from: https://build.nvidia.com")
            if not api_base:
                api_base = "https://integrate.api.nvidia.com/v1"
            if not model:
                model = (
                    os.environ.get("LIVE_VLM_DEFAULT_MODEL") or "meta/llama-3.2-11b-vision-instruct"
                ).strip()
                if os.environ.get("LIVE_VLM_DEFAULT_MODEL"):
                    logger.info(f"Using default model from env: {model}")
            if api_key == "EMPTY":
                logger.warning("⚠️  API key required for NVIDIA API Catalog")
                logger.warning("   Set with: --api-key YOUR_API_KEY")
                logger.warning("   Or use WebUI to configure API settings after starting")

    # Initialize VLM service and default session for multi-session support
    global vlm_service, default_vlm_config
    vlm_service = VLMService(
        model=model,
        api_base=api_base,
        api_key=api_key,
        prompt=args.prompt,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
    )
    default_vlm_config = {
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "prompt": args.prompt,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
    }
    sessions["default"] = {
        "vlm_service": vlm_service,
        "show_request_payload": False,
        "show_response_payload": False,
    }

    # Log initialization with better formatting
    service_name = "Local" if "localhost" in api_base or "127.0.0.1" in api_base else "Cloud"
    logger.info("Initialized VLM service:")
    logger.info(f"  Model: {model}")
    logger.info(f"  API: {api_base} ({service_name})")
    logger.info(f"  Prompt: {args.prompt}")

    # Update frame processing rate in VideoProcessorTrack if needed
    # (This is a bit hacky but works for this demo)
    VideoProcessorTrack.process_every_n_frames = args.process_every

    # Create web application using create_app
    app = asyncio.run(create_app(test_mode=False))

    # Setup SSL (auto-generate certificates if needed)
    ssl_context = None
    protocol = "http"
    if not args.no_ssl:
        # Try to auto-generate if certificates don't exist
        if not os.path.exists(args.ssl_cert) or not os.path.exists(args.ssl_key):
            success = generate_self_signed_cert(args.ssl_cert, args.ssl_key)
            if not success:
                # FAIL FAST - SSL is required for webcam access
                logger.error("")
                logger.error("❌ Cannot start server without SSL certificates")
                logger.error("❌ Remote webcam access requires HTTPS!")
                logger.error("")
                logger.error("🔧 To fix, install openssl:")
                logger.error("   Linux/Jetson: sudo apt install openssl")
                logger.error("   macOS: brew install openssl")
                logger.error("")
                logger.error("   Then restart the server")
                logger.error("")
                logger.error("⚠️  For a local kiosk, use --localhost-http and http://localhost")
                logger.error("")
                sys.exit(1)

        # Load certificates (they must exist at this point)
        if os.path.exists(args.ssl_cert) and os.path.exists(args.ssl_key):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(args.ssl_cert, args.ssl_key)
            protocol = "https"
            logger.info("SSL enabled - using HTTPS")
        else:
            # This should never happen, but just in case
            logger.error("❌ SSL certificates missing after generation - unexpected error")
            sys.exit(1)
    else:
        logger.warning("⚠️  SSL disabled with --no-ssl flag")
        if args.localhost_http or is_loopback_host(args.host):
            logger.info("Using localhost HTTP mode for same-device browser access")
            logger.info("Open http://localhost:%s on this machine for webcam access", args.port)
        else:
            logger.warning("⚠️  Webcam access may not work from non-local HTTP origins")
            logger.warning("⚠️  Use HTTPS for browsers connecting from another device")

    # Get network addresses
    import socket
    import subprocess

    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Access the server at:")
    logger.info(f"  Local:   {protocol}://localhost:{args.port}")

    # Get network interfaces - try multiple methods for cross-platform support
    network_ips = []

    # Method 1: hostname -I (Linux)
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                # Filter out loopback and docker bridges (172.17.x.x)
                if not ip.startswith("127.") and not ip.startswith("172.17."):
                    network_ips.append(ip)
    except Exception:
        pass

    # Method 2: Socket method (cross-platform fallback)
    if not network_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip != "127.0.0.1":
                network_ips.append(ip)
        except Exception:
            pass

    # Display all found network IPs
    for ip in network_ips:
        logger.info(f"  Network: {protocol}://{ip}:{args.port}")

    logger.info("=" * 70)
    logger.info("")
    logger.info("Press Ctrl+C to stop")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("\nReceived signal to terminate. Shutting down gracefully...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


def stop():
    """Stop the running live-vlm-webui server"""
    import sys
    import time

    try:
        import psutil
    except ImportError:
        logger.error("psutil is required for the stop command")
        logger.error("Install it with: pip install live-vlm-webui[dev]")
        sys.exit(1)

    print("Stopping Live VLM WebUI server...")

    # Find and kill processes running live_vlm_webui.server
    found = False
    killed = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "live_vlm_webui.server" in cmdline_str or "live-vlm-webui" in cmdline_str:
                    # Don't kill the stop command itself
                    if "stop" not in cmdline_str:
                        found = True
                        print(f"  Stopping process {proc.info['pid']}: {proc.info['name']}")
                        proc.terminate()
                        killed.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not found:
        print("✓ No running server found")
        return

    # Wait for graceful shutdown
    time.sleep(2)

    # Force kill if still running
    for proc in killed:
        try:
            if proc.is_running():
                print(f"  Force killing process {proc.pid}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Final verification
    time.sleep(1)
    still_running = False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline:
                cmdline_str = " ".join(cmdline)
                if "live_vlm_webui.server" in cmdline_str or "live-vlm-webui" in cmdline_str:
                    if "stop" not in cmdline_str:
                        still_running = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if still_running:
        print("❌ Failed to stop server")
        sys.exit(1)
    else:
        print("✓ Server stopped successfully")


if __name__ == "__main__":
    main()
