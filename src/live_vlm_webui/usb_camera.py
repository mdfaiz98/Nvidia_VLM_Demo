"""
USB (UVC) Webcam Source, captured server-side

The browser can't be trusted to pick the right camera on this device: V4L2
lists the Orbbec GMSL camera's raw nodes (/dev/video0-7) before any USB
webcam, and a browser's "default camera" is /dev/video0 - a GMSL depth node.
So instead of getUserMedia(), the server opens the webcam itself (OpenCV,
V4L2) and sends it to the page over WebRTC, the same way the GMSL Camera tab
and /depth page work. The browser never touches a camera.

- list_usb_cameras(): USB capture nodes found via sysfs, GMSL nodes excluded.
- UsbCamera: one reader thread per device, shared by every session using it
  (a V4L2 device can only be streamed by one opener), closed when the last
  session releases it.
- UsbCameraTrack: aiortc VideoStreamTrack over a shared UsbCamera.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import asyncio
import glob
import logging
import os
import re
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)

# Same filter as app.js used for browser devices: the Orbbec GMSL camera's
# raw V4L2 nodes ("vi-output, G300 depth", "tegra-capture-vi-metadata-0").
GMSL_V4L2_NODE = re.compile(r"vi-output|tegra-capture|\bG300\b", re.IGNORECASE)

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30


class UsbCameraError(RuntimeError):
    """Raised when a USB camera can't be found or opened."""


def list_usb_cameras() -> List[dict]:
    """
    USB webcams as [{"device": "/dev/videoN", "name": "..."}], sorted by node.

    Uses sysfs rather than probing devices, so it never opens (and never
    disturbs) a camera. A UVC webcam registers two nodes - capture (index 0)
    and metadata (index 1) - only index 0 is listed.
    """
    cameras = []
    for sys_dir in glob.glob("/sys/class/video4linux/video*"):
        try:
            with open(os.path.join(sys_dir, "name")) as f:
                name = f.read().strip()
            with open(os.path.join(sys_dir, "index")) as f:
                index = f.read().strip()
        except OSError:
            continue
        if index != "0" or GMSL_V4L2_NODE.search(name):
            continue
        # UVC names look like "Trust Full HD Webcam: Trust Ful" - keep the first half
        cameras.append({"device": f"/dev/{os.path.basename(sys_dir)}", "name": name.split(":")[0].strip()})
    return sorted(cameras, key=lambda c: int(c["device"][len("/dev/video") :]))


class UsbCamera:
    """Reader thread for one V4L2 device, caching the latest BGR frame."""

    def __init__(self, device: str):
        self.device = device
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._users = 0
        self._latest: Optional[np.ndarray] = None
        self._frame_version = 0
        self._new_frame = threading.Condition()

    def acquire(self):
        """Register a user, opening the device on first use."""
        if self._users == 0:
            self._open()
        self._users += 1

    def release(self):
        """Unregister a user, closing the device when nobody is left."""
        self._users = max(0, self._users - 1)
        if self._users == 0:
            self._close()

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise UsbCameraError(f"Could not open {self.device}")
        # MJPG is what makes 720p30 possible over USB 2.0 - raw YUYV tops out far lower
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise UsbCameraError(f"{self.device} opened but delivered no frames")

        self._cap = cap
        self._latest = frame
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name=f"usbcam-{self.device}", daemon=True)
        self._thread.start()
        logger.info(
            "USB camera started: %s %dx%d", self.device, frame.shape[1], frame.shape[0]
        )

    def _close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        self._latest = None
        logger.info("USB camera stopped: %s", self.device)

    def _read_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("USB camera %s: read failed", self.device)
                continue
            with self._new_frame:
                self._latest = frame
                self._frame_version += 1
                self._new_frame.notify_all()

    def wait_for_frame_after(self, version: int, timeout: float) -> int:
        """Blocking wait (call via executor) until a frame newer than `version` exists."""
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._frame_version != version, timeout)
            return self._frame_version

    def get_latest(self) -> Optional[np.ndarray]:
        with self._new_frame:
            return self._latest


_cameras: Dict[str, UsbCamera] = {}


def get_usb_camera(device: str) -> UsbCamera:
    """Shared UsbCamera for a device - only devices list_usb_cameras() reports are allowed."""
    if device not in {c["device"] for c in list_usb_cameras()}:
        raise UsbCameraError(f"{device} is not a connected USB camera")
    if device not in _cameras:
        _cameras[device] = UsbCamera(device)
    return _cameras[device]


class UsbCameraTrack(VideoStreamTrack):
    """
    aiortc VideoStreamTrack over a shared UsbCamera, so the webcam rides the
    same WebRTC + VLM pipeline as the RTSP/file/GMSL sources.
    """

    kind = "video"

    def __init__(self, camera: UsbCamera):
        super().__init__()
        self._camera = camera
        self._camera.acquire()
        self._released = False
        self._last_version = -1

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        loop = asyncio.get_event_loop()
        version = self._last_version
        while version == self._last_version and self.readyState == "live":
            version = await loop.run_in_executor(
                None, self._camera.wait_for_frame_after, self._last_version, 1.0
            )
        self._last_version = version

        image = self._camera.get_latest()
        if image is None:
            image = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)

        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self):
        super().stop()
        if not self._released:
            self._released = True
            self._camera.release()
