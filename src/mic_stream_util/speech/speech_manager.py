"""Silero VAD (Voice Activity Detection) implementation."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Callable

import numpy as np
import torch
from silero_vad import load_silero_vad

from mic_stream_util.core.audio_config import AudioConfig
from mic_stream_util.core.microphone_manager import MicrophoneStream

torch.set_num_threads(1)


class CallbackEventType(Enum):
    """Types of callback events that can be sent to the callback thread."""

    VAD_CHANGED = "vad_changed"
    SPEECH_START = "speech_start"
    SPEECH_CHUNK = "speech_chunk"
    AUDIO_CHUNK = "audio_chunk"
    SPEECH_ENDED = "speech_ended"


@dataclass
class CallbackEvent:
    """Represents a callback event to be processed."""

    event_type: CallbackEventType
    data: dict


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""

    threshold: float = 0.5
    padding_before_ms: int = 300
    padding_after_ms: int = 300
    max_silence_ms: int = 1000
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")

    def __post_init__(self) -> None:
        """Validate VAD configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("VAD threshold must be between 0.0 and 1.0")
        if self.padding_before_ms < 0:
            raise ValueError("Padding before must be non-negative")
        if self.padding_after_ms < 0:
            raise ValueError("Padding after must be non-negative")
        if self.max_silence_ms <= 0:
            raise ValueError("Max silence duration must be positive")


@dataclass
class SpeechChunk:
    """Represents a chunk of speech audio with timing information.

    Attributes:
        audio_chunk: The audio data as a numpy array
        start_time: Start time of the chunk in seconds
        end_time: End time of the chunk in seconds
        duration: Duration of the chunk in seconds
    """

    audio_chunk: np.ndarray
    start_time: float
    end_time: float
    duration: float


class CallbackProcessor:
    """Handles callback execution in a separate thread to avoid blocking audio processing."""

    def __init__(self):
        """Initialize the callback processor."""
        self.callback_queue: Queue[CallbackEvent] = Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        # Callback functions
        self._on_speech_start_cb: Callable[[float], None] | None = None
        self._on_vad_changed_cb: Callable[[float], None] | None = None
        self._on_speech_chunk_cb: Callable[[SpeechChunk, float], None] | None = None
        self._on_speech_ended_cb: Callable[[SpeechChunk], None] | None = None
        self._on_audio_chunk_cb: Callable[[np.ndarray, float], None] | None = None

    def set_callbacks(
        self,
        on_speech_start: Callable[[float], None] | None = None,
        on_vad_changed: Callable[[float], None] | None = None,
        on_speech_chunk: Callable[[SpeechChunk, float], None] | None = None,
        on_audio_chunk: Callable[[np.ndarray, float], None] | None = None,
        on_speech_ended: Callable[[SpeechChunk], None] | None = None,
    ) -> None:
        """Set all callbacks at once for convenience."""
        self._on_speech_start_cb = on_speech_start or self._on_speech_start_cb
        self._on_vad_changed_cb = on_vad_changed or self._on_vad_changed_cb
        self._on_speech_chunk_cb = on_speech_chunk or self._on_speech_chunk_cb
        self._on_audio_chunk_cb = on_audio_chunk or self._on_audio_chunk_cb
        self._on_speech_ended_cb = on_speech_ended or self._on_speech_ended_cb

    def start(self) -> None:
        """Start the callback processing thread."""
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._callback_worker, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the callback processing thread."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=1.0)

    def send_event(self, event: CallbackEvent) -> None:
        """Send a callback event to the processing queue."""
        if self.thread and self.thread.is_alive():
            try:
                self.callback_queue.put(event, timeout=0.1)
            except Exception as e:
                print(f"Warning: Could not send callback event: {e}")

    def _callback_worker(self) -> None:
        """Worker thread that handles callback execution."""
        while not self.stop_event.is_set():
            try:
                # Non-blocking get with timeout
                event = self.callback_queue.get(timeout=0.1)
                self._process_event(event)
            except Exception:
                # Timeout or other exception, continue loop
                continue

    def _process_event(self, event: CallbackEvent) -> None:
        """Process a single callback event."""
        try:
            if event.event_type == CallbackEventType.VAD_CHANGED:
                if self._on_vad_changed_cb:
                    self._on_vad_changed_cb(event.data["vad_score"])

            elif event.event_type == CallbackEventType.SPEECH_START:
                if self._on_speech_start_cb:
                    self._on_speech_start_cb(event.data["time"])

            elif event.event_type == CallbackEventType.SPEECH_CHUNK:
                if self._on_speech_chunk_cb:
                    speech_chunk = event.data["speech_chunk"]
                    vad_score = event.data["vad_score"]
                    self._on_speech_chunk_cb(speech_chunk, vad_score)

            elif event.event_type == CallbackEventType.AUDIO_CHUNK:
                if self._on_audio_chunk_cb:
                    audio_chunk = event.data["audio_chunk"]
                    time = event.data["time"]
                    self._on_audio_chunk_cb(audio_chunk, time)

            elif event.event_type == CallbackEventType.SPEECH_ENDED:
                if self._on_speech_ended_cb:
                    speech_chunk = event.data["speech_chunk"]
                    self._on_speech_ended_cb(speech_chunk)

        except Exception as e:
            print(f"Error processing callback event {event.event_type}: {e}")


class SpeechManager:
    """Manages voice activity detection and speech processing using Silero VAD.

    This class handles real-time audio processing, voice activity detection,
    and provides callbacks for speech events. It maintains buffers for
    pre-speech and post-speech padding to capture complete speech segments.

    Uses threading to separate audio processing from callback execution
    to prevent input overflows when callbacks perform heavy work.

    Attributes:
        audio_config: Configuration for audio input/output
        vad_config: Configuration for voice activity detection
        device: Device to run the VAD model on (CPU/GPU)
        model: The Silero VAD model
        microphone_stream: Manages the audio stream
        callback_processor: Handles callback execution in separate thread
    """

    def __init__(self, audio_config: AudioConfig, vad_config: VADConfig):
        """Initialize the SpeechManager with audio and VAD configuration.

        Args:
            audio_config: Configuration for audio input/output settings
            vad_config: Configuration for voice activity detection parameters
        """
        self.audio_config = audio_config
        self.vad_config = vad_config

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.model = load_silero_vad()
        if hasattr(self.model, "eval"):
            self.model.eval()  # type: ignore
        if hasattr(self.model, "to"):
            self.model.to(self.device)  # type: ignore

        self.microphone_stream = MicrophoneStream(audio_config)
        self.microphone_stream.set_callback(callback=self._process_audio_chunk)

        # Initialize callback processor
        self.callback_processor = CallbackProcessor()

        # Speech state tracking
        self.was_speaking = False
        self.speech_start_time = 0.0
        self.speech_duration = 0.0
        self.post_speech_counter = 0
        self.vad_score = 0.0
        self.current_time = 0.0

        # Audio padding buffers
        self.pre_speech_buffer: list[np.ndarray] = []
        self.pre_speech_buffer_duration = 0.0
        self.speech_buffer: list[np.ndarray] = []
        self.speech_buffer_duration = 0.0
        self.frame_duration_s = self.audio_config.num_samples / self.audio_config.sample_rate

        # Calculate buffer sizes
        self.pre_speech_chunks_needed = max(1, int(self.vad_config.padding_before_ms / (self.frame_duration_s * 1000)))
        self.post_speech_chunks_needed = max(1, int(self.vad_config.padding_after_ms / (self.frame_duration_s * 1000)))

    def start_stream(self) -> None:
        """Start the audio stream and begin processing audio chunks."""
        self.callback_processor.start()
        self.microphone_stream.start_stream()

    def stop_stream(self) -> None:
        """Stop the audio stream and cease processing audio chunks."""
        self.microphone_stream.stop_stream()
        self.callback_processor.stop()

    @contextmanager
    def stream_context(self):
        """Context manager for audio stream lifecycle.

        Yields:
            None: The context manager yields nothing, but manages stream start/stop
        """
        self.start_stream()
        yield
        self.stop_stream()

    def set_on_speech_start_callback(self, cb: Callable[[float], None]) -> None:
        """Set callback for when speech starts.

        Args:
            cb: Callback function that receives the start time of the speech
        """
        self.callback_processor._on_speech_start_cb = cb

    def set_on_vad_changed_callback(self, cb: Callable[[float], None]) -> None:
        """Set callback for VAD score changes.

        Args:
            cb: Callback function that receives the current VAD score (0.0 to 1.0)
        """
        self.callback_processor._on_vad_changed_cb = cb

    def set_on_speech_chunk_callback(self, cb: Callable[[SpeechChunk, float], None]) -> None:
        """Set callback for individual speech chunks during speech detection.

        Args:
            cb: Callback function that receives a SpeechChunk and the current VAD score
        """
        self.callback_processor._on_speech_chunk_cb = cb

    def set_on_audio_chunk_callback(self, cb: Callable[[np.ndarray, float], None]) -> None:
        """Set callback for all audio chunks (before VAD processing).

        Args:
            cb: Callback function that receives the raw audio chunk and timestamp
        """
        self.callback_processor._on_audio_chunk_cb = cb

    def set_on_speech_ended_callback(self, cb: Callable[[SpeechChunk], None]) -> None:
        """Set callback for when a complete speech segment ends.

        Args:
            cb: Callback function that receives the complete SpeechChunk with padding
        """
        self.callback_processor._on_speech_ended_cb = cb

    def set_callbacks(
        self,
        on_speech_start: Callable[[float], None] | None = None,
        on_vad_changed: Callable[[float], None] | None = None,
        on_speech_chunk: Callable[[SpeechChunk, float], None] | None = None,
        on_audio_chunk: Callable[[np.ndarray, float], None] | None = None,
        on_speech_ended: Callable[[SpeechChunk], None] | None = None,
    ) -> None:
        """Set all callbacks at once for convenience.

        Args:
            on_speech_start: Callback for when speech starts
            on_vad_changed: Callback for VAD score changes
            on_speech_chunk: Callback for individual speech chunks
            on_audio_chunk: Callback for all audio chunks
            on_speech_ended: Callback for complete speech segments
        """
        self.callback_processor.set_callbacks(
            on_speech_start=on_speech_start,
            on_vad_changed=on_vad_changed,
            on_speech_chunk=on_speech_chunk,
            on_audio_chunk=on_audio_chunk,
            on_speech_ended=on_speech_ended,
        )

    def _add_to_pre_speech_buffer(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to pre-speech buffer with duration tracking."""
        self.pre_speech_buffer.append(audio_chunk.copy())
        self.pre_speech_buffer_duration += self.frame_duration_s

        # Keep only the most recent chunks needed for padding
        while len(self.pre_speech_buffer) > self.pre_speech_chunks_needed:
            self.pre_speech_buffer.pop(0)
            self.pre_speech_buffer_duration -= self.frame_duration_s

    def _add_speech_audio_chunk(self, audio_chunk: np.ndarray, time: float) -> None:
        """Add audio chunk to speech buffer with duration tracking."""
        self.speech_buffer.append(audio_chunk.copy())
        self.speech_buffer_duration += self.frame_duration_s

        # Send speech chunk event to callback processor
        speech_chunk = SpeechChunk(audio_chunk, time, time + self.frame_duration_s, self.frame_duration_s)
        event = CallbackEvent(CallbackEventType.SPEECH_CHUNK, {"speech_chunk": speech_chunk, "vad_score": self.vad_score})
        self.callback_processor.send_event(event)

    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Process audio chunk with VAD. This runs in the audio thread and should be fast."""
        try:
            # Update current time
            self.current_time += self.frame_duration_s

            # Get VAD probability
            tensor = torch.from_numpy(audio_chunk).squeeze()
            vad_score = self.model(tensor, self.audio_config.sample_rate).item()

            self.vad_score = vad_score

            # Send VAD changed event to callback processor
            event = CallbackEvent(CallbackEventType.VAD_CHANGED, {"vad_score": vad_score})
            self.callback_processor.send_event(event)

            # Send audio chunk event to callback processor
            event = CallbackEvent(CallbackEventType.AUDIO_CHUNK, {"audio_chunk": audio_chunk, "time": self.current_time})
            self.callback_processor.send_event(event)

            voice_detected = vad_score > self.vad_config.threshold

            # Speech started
            if voice_detected and not self.was_speaking:
                # Send speech start event to callback processor
                event = CallbackEvent(CallbackEventType.SPEECH_START, {"time": self.current_time})
                self.callback_processor.send_event(event)

                self.was_speaking = True
                self.speech_start_time = self.current_time
                self.speech_duration = 0.0
                self.post_speech_counter = 0

                # Add pre-speech padding to speech buffer
                for chunk in self.pre_speech_buffer:
                    self._add_speech_audio_chunk(chunk, self.current_time)
                self.pre_speech_buffer.clear()
                self.pre_speech_buffer_duration = 0.0

                # Add current chunk
                self._add_speech_audio_chunk(audio_chunk, self.current_time)

            # Speech in progress
            elif voice_detected and self.was_speaking:
                self.post_speech_counter = 0
                self._add_speech_audio_chunk(audio_chunk, self.current_time)

            # Speech ended (silence detected)
            elif not voice_detected and self.was_speaking:
                self.post_speech_counter += 1

                # Add current chunk (part of post-speech padding)
                self._add_speech_audio_chunk(audio_chunk, self.current_time)

                # Check if we've reached max silence duration
                if self.post_speech_counter * self.frame_duration_s * 1000 >= self.vad_config.max_silence_ms:
                    self.was_speaking = False

                    # Send speech ended event to callback processor
                    concatenated_audio = np.concatenate(self.speech_buffer)
                    if self.vad_config.max_speech_duration_s <= 0 or self.speech_buffer_duration < self.vad_config.max_speech_duration_s:
                        speech_chunk = SpeechChunk(concatenated_audio, self.speech_start_time, self.current_time, self.speech_buffer_duration)
                        event = CallbackEvent(CallbackEventType.SPEECH_ENDED, {"speech_chunk": speech_chunk})
                        self.callback_processor.send_event(event)

                    self.speech_duration = 0.0
                    self.post_speech_counter = 0
                    self.speech_buffer.clear()
                    self.speech_buffer_duration = 0.0

            # Always add to pre-speech buffer for potential padding
            self._add_to_pre_speech_buffer(audio_chunk)

        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            raise e


if __name__ == "__main__":
    speech = SpeechManager(AudioConfig(device_name="sm900", num_samples=512), VADConfig())

    def vad_changed(vad_score: float) -> None:
        # print(f"VAD score: {vad_score}")
        pass

    def on_audio_chunk(speech_chunk: SpeechChunk, vad_score: float) -> None:
        print(f"Audio chunk: {speech_chunk.duration}, VAD score: {vad_score}")

    def on_speech_ended(speech_chunk: SpeechChunk) -> None:
        print(f"Speech ended: {speech_chunk.duration}")

    speech.set_on_vad_changed_callback(vad_changed)
    speech.set_on_speech_chunk_callback(on_audio_chunk)
    speech.set_on_speech_ended_callback(on_speech_ended)

    speech.start_stream()

    time.sleep(10)
