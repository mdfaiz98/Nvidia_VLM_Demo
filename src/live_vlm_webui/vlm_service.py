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
VLM Service
Handles async image analysis using any OpenAI-compatible VLM API
(Works with vLLM, SGLang, Ollama, OpenAI, etc.)
"""

import asyncio
import base64
import io
import time
from openai import AsyncOpenAI
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Cap the long side of frames sent to the VLM. Measured on this Jetson with
# qwen2.5vl:3b via Ollama (which rescales images itself before the vision
# encoder): 768px -> 1069 input tokens / 3.17s prefill, 1024px -> 1112 /
# 3.25s, native 1280x720 -> 1233 / 3.78s. 1024px gets most of the extra
# detail for ~0.08s; full resolution costs ~0.6s more per answer.
MAX_IMAGE_DIMENSION = 1024

# JPEG quality for the re-encode. PIL's default (75) stacks visible
# artifacts on top of the camera's own MJPEG compression; 90 costs nothing
# measurable in latency.
JPEG_QUALITY = 90

# Sampling temperature. 0.7 let qwen2.5vl:3b sometimes pick an unlikely
# "yes" on presence questions (claimed a phone in 2 of 6 runs on a frame
# with none); 0.2 answered correctly 6/6, with no latency cost.
TEMPERATURE = 0.2


class VLMService:
    """Service for analyzing images using VLM via OpenAI-compatible API"""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        prompt: str = "Describe what you see in this image in one sentence.",
        max_tokens: int = 512,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize VLM service

        Args:
            model: Model name (e.g., "llama-3.2-11b-vision-instruct" for vLLM)
            api_base: Base URL for the API (e.g., "http://localhost:8000/v1" for vLLM)
            api_key: API key (use "EMPTY" for local servers)
            prompt: Default prompt to use for image analysis
            max_tokens: Maximum tokens to generate
            reasoning_effort: Optional backend reasoning setting; "none" disables thinking in Ollama
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key if api_key else "EMPTY"
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self.current_response = "Initializing..."
        self.is_processing = False
        self._processing_lock = asyncio.Lock()
        self._last_request_payload = None  # For debug: request body (image truncated)
        self._last_response_payload = None  # For debug: API response body

        # Metrics tracking
        self.last_inference_time = 0.0  # seconds
        self.total_inferences = 0
        self.total_inference_time = 0.0
        # Streaming metrics: time to first token, answer length, decode speed
        self.last_ttft: Optional[float] = None  # seconds
        self.last_tokens: Optional[int] = None
        self.last_tokens_per_s: Optional[float] = None
        self.total_ttft = 0.0
        self.ttft_count = 0
        self.total_tokens_per_s = 0.0
        self.tokens_per_s_count = 0
        # The answer so far while one is being streamed, else None
        self.partial_response: Optional[str] = None

    async def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Analyze an image using the VLM model

        Args:
            image: PIL Image to analyze
            prompt: Prompt for the VLM (uses default if None)

        Returns:
            Generated response string
        """
        if prompt is None:
            prompt = self.prompt

        try:
            start_time = time.perf_counter()

            # Downscale before sending - only ever shrinks, never upscales
            if max(image.size) > MAX_IMAGE_DIMENSION:
                scale = MAX_IMAGE_DIMENSION / max(image.size)
                new_size = (round(image.width * scale), round(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)

            # Convert PIL Image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="JPEG", quality=JPEG_QUALITY)
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

            # Create message with image
            image_url = f"data:image/jpeg;base64,{img_base64}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

            # Store request payload for debug (truncate base64 for display)
            truncate_len = 120
            if len(img_base64) > truncate_len:
                image_url_debug = f"data:image/jpeg;base64,{img_base64[:truncate_len]}...<{len(img_base64)} chars total>"
            else:
                image_url_debug = image_url
            messages_debug = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url_debug}},
                    ],
                }
            ]
            self._last_request_payload = {
                "model": self.model,
                "messages": messages_debug,
                "max_tokens": self.max_tokens,
                "temperature": TEMPERATURE,
            }

            # Omit this optional parameter for backends that do not support it.
            request_options = {}
            if self.reasoning_effort is not None:
                request_options["reasoning_effort"] = self.reasoning_effort
            self._last_request_payload.update(request_options)

            # Call API, streamed: the answer is shown word by word as it's
            # generated (self.partial_response), and the first token's
            # arrival gives TTFT - same metrics as the Qualcomm IQ9 demo.
            pieces: list[str] = []
            ttft = None
            usage = None
            chunks = 0
            self.partial_response = ""
            stream = None
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=TEMPERATURE,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=request_options,
                )
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    for choice in chunk.choices or []:
                        piece = getattr(choice.delta, "content", None) or ""
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - start_time
                            pieces.append(piece)
                            chunks += 1
                            self.partial_response = "".join(pieces)
            finally:
                self.partial_response = None
                if stream is not None:
                    await stream.close()

            # Calculate latency
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            tokens = (getattr(usage, "completion_tokens", None) if usage else None) or chunks
            # Decode speed: tokens after the first one, over the time after the first one
            tokens_per_s = (
                (tokens - 1) / (inference_time - ttft)
                if ttft is not None and tokens > 1 and inference_time > ttft
                else None
            )
            result = "".join(pieces).strip()

            # Store response payload for debug (the streamed answer, reassembled)
            self._last_response_payload = {
                "model": self.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result}}],
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
                "timings": {
                    "ttft_ms": round(ttft * 1000) if ttft is not None else None,
                    "total_ms": round(inference_time * 1000),
                    "completion_tokens": tokens,
                    "tokens_per_s": round(tokens_per_s, 1) if tokens_per_s else None,
                },
            }

            # Update metrics
            self.last_inference_time = inference_time
            self.total_inferences += 1
            self.total_inference_time += inference_time
            self.last_ttft = ttft
            self.last_tokens = tokens
            self.last_tokens_per_s = tokens_per_s
            if ttft is not None:
                self.total_ttft += ttft
                self.ttft_count += 1
            if tokens_per_s:
                self.total_tokens_per_s += tokens_per_s
                self.tokens_per_s_count += 1

            logger.info(
                f"VLM response: {result} (latency: {inference_time*1000:.0f}ms, "
                f"ttft: {ttft*1000 if ttft is not None else -1:.0f}ms, {tokens} tok, "
                f"{tokens_per_s or 0:.1f} tok/s)"
            )
            return result

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error: {str(e)}"

    def get_last_request_payload(self) -> Optional[dict]:
        """
        Return the last request payload sent to the API (for debug).
        Image data is truncated to avoid huge JSON. Returns None if no request has been made yet.
        """
        return self._last_request_payload

    def get_last_response_payload(self) -> Optional[dict]:
        """
        Return the last API response payload (for debug).
        Returns None if no response has been received yet.
        """
        return self._last_response_payload

    async def process_frame(self, image: Image.Image, prompt: Optional[str] = None) -> None:
        """
        Process a frame asynchronously. Updates self.current_response when done.
        If already processing, this call is skipped.

        Args:
            image: PIL Image to process
            prompt: Optional custom prompt (uses default if None)
        """
        # Non-blocking check if we're already processing
        if self._processing_lock.locked():
            logger.debug("VLM busy, skipping frame")
            return

        async with self._processing_lock:
            self.is_processing = True
            try:
                response = await self.analyze_image(image, prompt)
                self.current_response = response
            finally:
                self.is_processing = False

    def get_current_response(self) -> tuple[str, bool]:
        """
        Get the current response and processing status

        Returns:
            Tuple of (response, is_processing)
        """
        return self.current_response, self.is_processing

    def get_metrics(self) -> dict:
        """
        Get current performance metrics

        Returns:
            Dict with latency and throughput metrics
        """
        avg_latency = (
            self.total_inference_time / self.total_inferences if self.total_inferences > 0 else 0.0
        )

        return {
            "last_latency_ms": self.last_inference_time * 1000,
            "avg_latency_ms": avg_latency * 1000,
            "total_inferences": self.total_inferences,
            "is_processing": self.is_processing,
            "last_ttft_ms": self.last_ttft * 1000 if self.last_ttft is not None else None,
            "avg_ttft_ms": self.total_ttft / self.ttft_count * 1000 if self.ttft_count else None,
            "last_tokens": self.last_tokens,
            "last_tokens_per_s": self.last_tokens_per_s,
            "avg_tokens_per_s": (
                self.total_tokens_per_s / self.tokens_per_s_count if self.tokens_per_s_count else None
            ),
            # The answer being generated right now (streamed), for the page to
            # show word by word; None between answers
            "partial_text": self.partial_response,
        }

    def update_prompt(self, new_prompt: str, max_tokens: Optional[int] = None) -> None:
        """
        Update the default prompt and optionally max_tokens

        Args:
            new_prompt: New prompt to use
            max_tokens: Maximum tokens to generate (optional)
        """
        self.prompt = new_prompt
        if max_tokens is not None:
            self.max_tokens = max_tokens
            logger.info(f"Updated prompt to: {new_prompt}, max_tokens: {max_tokens}")
        else:
            logger.info(f"Updated prompt to: {new_prompt}")

    def update_api_settings(
        self, api_base: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        """
        Update API base URL and/or API key, recreating the client

        Args:
            api_base: New API base URL (optional)
            api_key: New API key (optional, use empty string for local services)
        """
        if api_base:
            self.api_base = api_base
        if api_key is not None:  # Allow empty string
            self.api_key = api_key if api_key else "EMPTY"

        # Recreate the client with new settings
        self.client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)

        masked_key = (
            "***" + self.api_key[-4:]
            if self.api_key and len(self.api_key) > 4 and self.api_key != "EMPTY"
            else "EMPTY"
        )
        logger.info(f"Updated API settings - base: {self.api_base}, key: {masked_key}")
