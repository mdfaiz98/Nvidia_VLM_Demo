"""
Orbbec Depth Camera Source (e.g. Gemini 2L)

Wraps pyorbbecsdk2 to expose Color, Depth, and IR streams plus IMU telemetry
from a single physical device to the rest of live-vlm-webui:

- OrbbecColorTrack: an aiortc VideoStreamTrack for the Color stream, so it can
  ride the exact same WebRTC + VLM capture pipeline as webcam/RTSP/file
  sources (see rtsp_track.py / video_file_track.py for the sibling pattern).
- OrbbecCameraManager: opens the device once and owns the SDK pipelines.
  Depth/IR aren't run through WebRTC (see CLAUDE.md's aioice/loopback
  caveats) - server.py instead pulls JPEG-encoded frames and IMU readings
  from this manager directly for a WebSocket push.

The camera can only be opened by one process/pipeline at a time, so
get_orbbec_manager() returns a single shared instance for the whole server.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import asyncio
import logging
import os
import threading
import time
from fractions import Fraction
from typing import Optional

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)

# Valid depth range in mm - matches pyorbbecsdk's own bundled examples.
MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 10000


class OrbbecDeviceNotFoundError(RuntimeError):
    """Raised when no Orbbec device is connected."""


def _frame_to_bgr_image(frame) -> Optional[np.ndarray]:
    """Convert a pyorbbecsdk ColorFrame to a BGR numpy image (any common format)."""
    from pyorbbecsdk import OBFormat

    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.NV12:
        y = data[0:height, :]
        uv = data[height : height + height // 2].reshape(height // 2, width)
        return cv2.cvtColor(cv2.merge([y, uv]), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.NV21:
        y = data[0:height, :]
        uv = data[height : height + height // 2].reshape(height // 2, width)
        return cv2.cvtColor(cv2.merge([y, uv]), cv2.COLOR_YUV2BGR_NV21)

    logger.warning("Unsupported Orbbec color format: %s", fmt)
    return None


def _depth_frame_to_colormap(frame) -> Optional[np.ndarray]:
    """Convert a raw uint16 DepthFrame to a JET-colormapped BGR image."""
    try:
        depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
            (frame.get_height(), frame.get_width())
        )
    except ValueError:
        return None

    depth_data = depth_data.astype(np.float32) * frame.get_depth_scale()
    depth_data = np.where((depth_data > MIN_DEPTH_MM) & (depth_data < MAX_DEPTH_MM), depth_data, 0)
    depth_8bit = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def _ir_frame_to_bgr_image(frame) -> Optional[np.ndarray]:
    """Convert an IRFrame (Y8/Y16/MJPG) to a normalized grayscale-as-BGR image."""
    from pyorbbecsdk import OBFormat

    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.MJPG:
        decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return None
        ir_data = np.resize(decoded, (height, width, 1)).astype(np.uint8)
    elif fmt == OBFormat.Y8:
        ir_data = np.resize(data, (height, width, 1)).astype(np.uint8)
    else:
        # Y16 and similar wide formats
        ir_data = np.resize(np.frombuffer(data, dtype=np.uint16), (height, width, 1))
        ir_data = cv2.normalize(ir_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.cvtColor(ir_data, cv2.COLOR_GRAY2BGR)


class OrbbecCameraManager:
    """
    Owns the Orbbec SDK pipelines for one physical device and caches the
    latest frame/telemetry from each stream behind a lock, so multiple
    consumers (the WebRTC color track, the depth/IR WebSocket pusher) can
    read the current state without each managing their own SDK session.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline = None
        self._imu_pipeline = None
        self._device_info: dict = {}
        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_ir_left: Optional[np.ndarray] = None
        self._latest_ir_right: Optional[np.ndarray] = None
        self._latest_accel: Optional[dict] = None
        self._latest_gyro: Optional[dict] = None
        self._frame_version = 0
        # Per-consumer "newer than version N" waits (see wait_for_frame_after),
        # not a shared Event - with the main page's Color track and the /depth
        # page's WebSocket both waiting, one consumer clearing a shared Event
        # could swallow the other's wakeup and delay it by a frame.
        self._new_frame = threading.Condition(self._lock)
        self._started = False

    def start(self) -> dict:
        """
        Open the device and start Color+Depth+IR (single pipeline, async
        callback) plus Accel+Gyro (separate pipeline - IMU is a distinct
        sensor group in the SDK). Returns the device info dict.
        Raises OrbbecDeviceNotFoundError if nothing is connected.
        """
        if self._started:
            return self._device_info

        from pyorbbecsdk import Config, Context, OBError, OBSensorType, Pipeline

        ctx = Context()
        if ctx.query_devices().get_count() == 0:
            raise OrbbecDeviceNotFoundError("No Orbbec device connected")

        # Pipeline() resolves its property-alias table from a relative
        # config/OrbbecSDKConfig.xml under the process's *working directory*,
        # not from the package/library location. Our server's CWD is
        # wherever `live-vlm-webui` was launched from, which has no such
        # config/ dir - Pipeline() then fails immediately with "Property not
        # found for aliasing" before touching any stream config. Fix: chdir
        # into the pyorbbecsdk package dir (which ships that config/) just
        # for construction, then restore - safe here since this call is
        # synchronous/blocking, so no other coroutine runs while CWD is
        # flipped.
        import pyorbbecsdk

        pkg_dir = os.path.dirname(pyorbbecsdk.__file__)
        orig_cwd = os.getcwd()
        try:
            os.chdir(pkg_dir)
            pipeline = Pipeline()
        finally:
            os.chdir(orig_cwd)
        config = Config()
        for sensor_type in (
            OBSensorType.COLOR_SENSOR,
            OBSensorType.DEPTH_SENSOR,
            OBSensorType.LEFT_IR_SENSOR,
            OBSensorType.RIGHT_IR_SENSOR,
        ):
            try:
                config.enable_stream(sensor_type)
            except OBError as e:
                logger.warning("Orbbec: could not enable %s: %s", sensor_type, e)

        try:
            pipeline.start(config, self._on_video_frames)
        except OBError as e:
            raise OrbbecDeviceNotFoundError(f"Failed to start Orbbec pipeline: {e}") from e

        info = pipeline.get_device().get_device_info()
        self._device_info = {
            "name": info.get_name(),
            "serial_number": info.get_serial_number(),
            "firmware_version": info.get_firmware_version(),
            "connection_type": info.get_connection_type(),
        }
        self._pipeline = pipeline

        try:
            device = pipeline.get_device()
            device.get_sensor(OBSensorType.ACCEL_SENSOR)
            device.get_sensor(OBSensorType.GYRO_SENSOR)

            imu_pipeline = Pipeline()
            imu_config = Config()
            imu_config.enable_accel_stream()
            imu_config.enable_gyro_stream()
            imu_pipeline.start(imu_config, self._on_imu_frames)
            self._imu_pipeline = imu_pipeline
        except Exception as e:
            logger.warning("Orbbec: IMU not available: %s", e)

        self._started = True
        logger.info("Orbbec camera started: %s", self._device_info)
        return self._device_info

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._imu_pipeline:
            self._imu_pipeline.stop()
            self._imu_pipeline = None
        self._started = False

    # -- SDK callbacks (run on an SDK-internal thread, not the asyncio loop) --

    def _on_video_frames(self, frames):
        if frames is None:
            return
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        ir_left_frame = frames.get_left_ir_frame()
        ir_right_frame = frames.get_right_ir_frame()

        color = _frame_to_bgr_image(color_frame) if color_frame else None
        depth = _depth_frame_to_colormap(depth_frame) if depth_frame else None
        ir_left = _ir_frame_to_bgr_image(ir_left_frame) if ir_left_frame else None
        ir_right = _ir_frame_to_bgr_image(ir_right_frame) if ir_right_frame else None

        with self._lock:
            if color is not None:
                self._latest_color = color
            if depth is not None:
                self._latest_depth = depth
            if ir_left is not None:
                self._latest_ir_left = ir_left
            if ir_right is not None:
                self._latest_ir_right = ir_right
            self._frame_version += 1
            self._new_frame.notify_all()

    def _on_imu_frames(self, imu_frames):
        if imu_frames is None:
            return
        accel_frame = imu_frames.get_accel_frame()
        gyro_frame = imu_frames.get_gyro_frame()

        with self._lock:
            if accel_frame:
                self._latest_accel = {
                    "x": accel_frame.get_x(),
                    "y": accel_frame.get_y(),
                    "z": accel_frame.get_z(),
                    "temperature": accel_frame.get_temperature(),
                }
            if gyro_frame:
                self._latest_gyro = {
                    "x": gyro_frame.get_x(),
                    "y": gyro_frame.get_y(),
                    "z": gyro_frame.get_z(),
                    "temperature": gyro_frame.get_temperature(),
                }

    # -- Consumer-facing reads --

    def get_latest_color(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_color is None else self._latest_color.copy()

    def get_frame_version(self) -> int:
        with self._lock:
            return self._frame_version

    def wait_for_frame_after(self, version: int, timeout: float) -> int:
        """
        Blocking wait (call via executor) until the frame version moves past
        `version`. Returns the current version - unchanged if it timed out.
        """
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._frame_version != version, timeout)
            return self._frame_version

    def encode_depth_jpeg(self, quality: int = 80) -> Optional[bytes]:
        with self._lock:
            depth = self._latest_depth
        if depth is None:
            return None
        ok, buf = cv2.imencode(".jpg", depth, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def encode_ir_left_jpeg(self, quality: int = 80) -> Optional[bytes]:
        with self._lock:
            ir = self._latest_ir_left
        if ir is None:
            return None
        ok, buf = cv2.imencode(".jpg", ir, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def encode_ir_right_jpeg(self, quality: int = 80) -> Optional[bytes]:
        with self._lock:
            ir = self._latest_ir_right
        if ir is None:
            return None
        ok, buf = cv2.imencode(".jpg", ir, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def encode_ir_jpeg(self, side: str, quality: int = 80) -> Optional[bytes]:
        """Encode just the IR sensor currently on screen ("left"/"right")."""
        if side == "right":
            return self.encode_ir_right_jpeg(quality)
        return self.encode_ir_left_jpeg(quality)

    def get_telemetry(self) -> dict:
        with self._lock:
            return {
                "device": dict(self._device_info),
                "accel": dict(self._latest_accel) if self._latest_accel else None,
                "gyro": dict(self._latest_gyro) if self._latest_gyro else None,
            }


_manager: Optional[OrbbecCameraManager] = None


def get_orbbec_manager() -> OrbbecCameraManager:
    """Shared singleton - the physical device can only be opened once."""
    global _manager
    if _manager is None:
        _manager = OrbbecCameraManager()
    return _manager


class OrbbecColorTrack(VideoStreamTrack):
    """
    aiortc VideoStreamTrack for the Orbbec Color stream, so it rides the same
    WebRTC + VLM capture pipeline as webcam/RTSP/file sources.
    """

    kind = "video"

    def __init__(self, manager: OrbbecCameraManager):
        super().__init__()
        self._manager = manager
        self._last_version = -1

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        loop = asyncio.get_event_loop()
        # Wait for a frame newer than the last one we returned, rather than
        # polling on a fixed interval - keeps latency close to the camera's
        # own frame rate without busy-looping. Times out and re-waits while
        # no frames arrive (e.g. device still starting up).
        version = self._last_version
        while version == self._last_version:
            version = await loop.run_in_executor(
                None, self._manager.wait_for_frame_after, self._last_version, 1.0
            )

        self._last_version = version
        image = self._manager.get_latest_color()
        if image is None:
            # Shouldn't happen once frame_version has advanced, but guard anyway.
            image = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
