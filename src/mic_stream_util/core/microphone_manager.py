from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from multiprocessing import Event, Process, Queue
from typing import Callable

import numpy as np
import sounddevice as sd

from mic_stream_util.backends import DeviceBackend
from mic_stream_util.core.audio_buffer import SharedAudioBuffer
from mic_stream_util.core.audio_config import AudioConfig


class MicrophoneStream:
    """
    Manages the microphone stream.
    The stream is started in a separate process placing the raw audio data in a shared memory buffer.
    """

    def __init__(self, config: AudioConfig | None = None):
        """
        Initialize the microphone stream.

        Parameters
        ----------
        config : AudioConfig | None, optional
            Audio configuration. If None, uses default configuration.
        """
        self.config = config or AudioConfig()
        self.buffer: SharedAudioBuffer | None = None
        self.process: Process | None = None
        self.stop_event: Event | None = None  # type: ignore
        self.error_queue: Queue | None = None
        self._streaming = False

        # Backend for device management
        self.backend: DeviceBackend | None = None

        # Callback-related attributes
        self._callback: Callable[[np.ndarray], None] | None = None
        self._callback_thread: threading.Thread | None = None
        self._callback_stop_event: threading.Event | None = None

    def set_callback(self, callback: Callable[[np.ndarray], None] | None) -> None:
        """
        Set a callback function to be called when audio data is available.
        The callback will be called in a separate thread.

        Parameters
        ----------
        callback : Callable[[np.ndarray], None] | None
            Function to call with audio data. Should accept a numpy array with shape (num_samples, channels).
            If None, callback mode is disabled.
        """
        if self._streaming:
            raise RuntimeError("Cannot set callback while stream is active")

        self._callback = callback

    def clear_callback(self) -> None:
        """Clear the callback function and disable callback mode."""
        if self._streaming:
            raise RuntimeError("Cannot clear callback while stream is active")

        self._callback = None

    def has_callback(self) -> bool:
        """Check if a callback function is set."""
        return self._callback is not None

    def _callback_processing_thread(self) -> None:
        """Thread that continuously reads audio data and calls the callback function."""
        if not self._callback or not self.buffer or not self._callback_stop_event:
            return

        try:
            while not self._callback_stop_event.is_set():
                try:
                    # Read audio data from buffer
                    audio_array = self._read(self.config.num_samples)

                    # Call the callback function
                    self._callback(audio_array)

                except Exception as e:
                    print(f"Error in callback processing thread: {e}")
                    # Continue processing even if callback fails

        except Exception as e:
            print(f"Fatal error in callback processing thread: {e}")

    @staticmethod
    def _audio_capture_process(
        config: AudioConfig,
        buffer: SharedAudioBuffer,
        stop_event: Event,  # type: ignore
        error_queue: Queue,
        backend: DeviceBackend | None = None,
    ) -> None:
        """
        Audio capture process that continuously reads from microphone and writes to shared buffer.

        Parameters
        ----------
        config : AudioConfig
            Audio configuration for the stream.
        buffer_name : str
            Name of the shared buffer to write audio data to.
        stop_event : Event
            Event to signal the process to stop.
        error_queue : Queue
            Queue to report errors back to the main process.
        """

        try:
            print(f"Starting audio capture process with config: {config} and buffer: {buffer.shm_name}")

            def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
                """Callback function for audio stream."""

                if status:
                    error_queue.put(f"Audio callback error: {status}")
                    return

                # Convert to bytes and write to buffer
                data = indata.tobytes()
                buffer.write(data)

            # Prepare device for streaming if backend is available and device is specified
            stream_kwargs = config.to_sounddevice_kwargs()
            if backend and config.device is not None:
                try:
                    # Prepare the device for streaming (backend-specific setup)
                    device_params = backend.prepare_device_for_streaming(config.device)
                    # Update stream kwargs with backend-prepared device parameters
                    stream_kwargs.update(device_params)
                    print(f"Prepared device for streaming: {device_params}")
                except Exception as e:
                    error_queue.put(f"Failed to prepare device for streaming: {e}")
                    return

            # Start the audio stream
            with sd.InputStream(callback=audio_callback, **stream_kwargs):
                try:
                    # Keep the stream running until stop event is set
                    stop_event.wait()
                except KeyboardInterrupt:
                    pass

        except Exception as e:
            error_queue.put(f"Audio capture process error: {e}")
        finally:
            buffer.close()

    def start_stream(self, ignore_already_started: bool = True) -> None:
        """Start the microphone stream in a separate process."""
        if self._streaming and not ignore_already_started:
            return

        if self._streaming:
            logging.warning("Microphone stream already started, ignoring start request")
            return

        # Initialize backend if not already done
        if self.backend is None:
            try:
                self.backend = DeviceBackend.get_backend()
                print(f"Using backend: {self.backend.__class__.__name__}")
            except Exception as e:
                logging.warning(f"Could not initialize backend: {e}")
                self.backend = None

        # Create shared buffer
        self.buffer = SharedAudioBuffer(self.config)
        # print(f"Created shared buffer with name: {self.buffer.shm_name}")

        # Create process control objects
        self.stop_event = Event()
        self.error_queue = Queue()

        # Start audio capture process
        self.process = Process(target=MicrophoneStream._audio_capture_process, args=(self.config, self.buffer, self.stop_event, self.error_queue, self.backend), daemon=True)
        self.process.start()

        # Wait a bit for the stream to start
        time.sleep(0.1)

        # Start callback processing thread if callback is set
        if self._callback:
            self._callback_stop_event = threading.Event()
            self._callback_thread = threading.Thread(target=self._callback_processing_thread, daemon=True)
            self._callback_thread.start()

        self._streaming = True

        # Check for immediate errors
        if not self.error_queue.empty():
            error = self.error_queue.get()
            self.stop_stream()
            raise RuntimeError(f"Failed to start audio stream: {error}")

    @contextmanager
    def stream(self):
        """Context manager for automatic stream start/stop."""
        try:
            self.start_stream()
            yield self
        finally:
            self.stop_stream()

    def stop_stream(self, ignore_not_started: bool = True) -> None:
        """Stop the microphone stream."""
        if not self._streaming and ignore_not_started:
            return

        if not self._streaming:
            logging.warning("Microphone stream not started, cannot stop")
            return

        # Signal process to stop
        if self.stop_event:
            self.stop_event.set()

        # Stop callback processing thread
        if self._callback_stop_event:
            self._callback_stop_event.set()

        if self._callback_thread and self._callback_thread.is_alive():
            self._callback_thread.join(timeout=2.0)

        # Wait for process to finish
        if self.process and self.process.is_alive():
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
                if self.process.is_alive():
                    self.process.kill()

        # Clean up resources
        if self.buffer:
            self.buffer.close()
            self.buffer.unlink()
            self.buffer = None

        self.process = None
        self.stop_event = None
        self.error_queue = None
        self._callback_thread = None
        self._callback_stop_event = None
        self._streaming = False

    def is_streaming(self) -> bool:
        """Check if the stream is currently active."""
        return self._streaming and self.process is not None and self.process.is_alive()

    def _read_raw(self, num_samples: int) -> bytes:
        """
        Reads raw audio data from the stream buffer.
        Blocks until at least num_samples are available.
        """
        if not self.is_streaming() or self.buffer is None:
            raise RuntimeError("Stream is not active")

        return self.buffer.read(num_samples)

    def _read(self, num_samples: int) -> np.ndarray:
        """
        Reads audio data from the stream buffer.
        Blocks until at least num_samples are available.
        """

        if num_samples is None:
            num_samples = self.config.num_samples

        raw_data = self._read_raw(num_samples)

        audio_array = np.frombuffer(raw_data, dtype=self.config.dtype)

        # Reshape to (num_samples, channels)
        audio_array = audio_array.reshape(-1, self.config.channels)

        return audio_array

    def read_raw(self, num_samples: int) -> bytes:
        """
        Reads raw audio data from the stream buffer.
        Blocks until at least num_samples are available.

        Parameters
        ----------
        num_samples : int
            The number of samples to read.

        Returns
        -------
        bytes
            Raw audio data as bytes.
        """

        if self._callback:
            raise RuntimeError("Cannot use read methods when callback is active. Use set_callback(None) to disable callback mode.")

        return self._read_raw(num_samples)

    def read(self, num_samples: int | None = None) -> np.ndarray:
        """
        Reads audio data from the stream buffer.
        Blocks until at least num_samples are available.

        Parameters
        ----------
        num_samples : int | None, optional
            The number of samples to read. If None, uses the default number of samples from the config.

        Returns
        -------
        np.ndarray
            Audio data as numpy array with shape (num_samples, channels).
        """
        if num_samples is None:
            num_samples = self.config.num_samples

        raw_data = self.read_raw(num_samples)

        audio_array = np.frombuffer(raw_data, dtype=self.config.dtype)

        # Reshape to (num_samples, channels)
        audio_array = audio_array.reshape(-1, self.config.channels)

        return audio_array
