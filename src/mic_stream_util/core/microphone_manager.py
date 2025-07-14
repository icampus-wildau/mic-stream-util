from __future__ import annotations

# import threading
import time
from contextlib import contextmanager
from multiprocessing import Event, Lock, Process, Queue, shared_memory

import numpy as np
import sounddevice as sd

from mic_stream_util.core.audio_config import AudioConfig


class SharedAudioBuffer:
    """
    Shared audio buffer for the microphone stream.

    Implements a thread-safe ring buffer using shared memory for inter-process communication.
    """

    def __init__(
        self,
        config: AudioConfig,
        shm_name: str | None = None,
    ):
        """
        Initializes the shared audio buffer.
        If shm_name is not provided, a new shared memory object is created.

        Parameters
        ----------
        config : AudioConfig
            The audio configuration.
        shm_name : str | None, optional
            The name of the shared memory object. If not provided, a new shared memory object is created.
        """
        self.config = config
        self.shm_name = shm_name or f"mic_buffer_{id(self)}"

        # Calculate sample size in bytes based on dtype
        self.sample_dtype_size = AudioConfig.get_dtype_size(config.dtype)

        # Calculate total buffer size in bytes
        # buffer_size is guaranteed to be non-None after __post_init__
        assert config.buffer_size is not None
        self.buffer_size_samples = config.buffer_size * config.channels
        self.buffer_size_bytes = self.buffer_size_samples * self.sample_dtype_size

        if shm_name is None:
            shm_name = self.shm_name
            # Create shared memory for the ring buffer
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.buffer_size_bytes)

            # Create shared memory for metadata (read/write positions) using unsigned long
            self.meta_shm = shared_memory.SharedMemory(
                name=f"{self.shm_name}_meta",
                create=True,
                size=16,  # 2 uint64 values
            )
        else:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.meta_shm = shared_memory.SharedMemory(name=f"{self.shm_name}_meta")

        # Initialize metadata using unsigned long (uint64)
        self.meta_array = np.frombuffer(self.meta_shm.buf, dtype=np.uint64)
        self.meta_array[0] = 0  # read position
        self.meta_array[1] = 0  # write position

        # Lock for thread safety
        self.lock = Lock()

        # Event for signaling new data availability
        self.data_available = Event()

    # -------- pickling helpers --------
    def __getstate__(self):
        d = self.__dict__.copy()
        # SharedMemory itself can't be pickled
        d["shm"] = None
        d["meta_shm"] = None
        d["meta_array"] = None

        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # re-attach to the already-existing block
        if self.shm_name is not None:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.meta_shm = shared_memory.SharedMemory(name=f"{self.shm_name}_meta")
            self.meta_array = np.frombuffer(self.meta_shm.buf, dtype=np.uint64)

    # -------- end of pickling helpers --------

    def write(self, data: bytes) -> None:
        """
        Write audio data to the buffer.
        Takes care about thread safety.

        Parameters
        ----------
        data : bytes
            The audio data to write.
        """
        with self.lock:
            data_len = len(data)
            if data_len > self.buffer_size_bytes:
                # If data is larger than buffer, only keep the latest portion
                data = data[-self.buffer_size_bytes :]
                data_len = self.buffer_size_bytes

            # Get current positions
            read_pos = self.meta_array[0]
            write_pos = self.meta_array[1]

            # Write data to buffer
            if write_pos + data_len <= self.buffer_size_bytes:
                # Simple case: data fits without wrapping
                self.shm.buf[write_pos : write_pos + data_len] = data
                write_pos = (write_pos + data_len) % self.buffer_size_bytes
            else:
                # Wrapping case: split data across buffer end
                first_part = self.buffer_size_bytes - write_pos
                self.shm.buf[write_pos:] = data[:first_part]
                self.shm.buf[: data_len - first_part] = data[first_part:]
                write_pos = data_len - first_part

            # Update write position
            self.meta_array[1] = write_pos

            # If buffer is full, advance read position (drop oldest data)
            if (write_pos + 1) % self.buffer_size_bytes == read_pos:
                read_pos = (read_pos + data_len) % self.buffer_size_bytes
                self.meta_array[0] = read_pos

            # Signal that new data is available
            self.data_available.set()

    def read(self, num_samples: int) -> bytes:
        """
        Read audio data from the buffer.
        Blocks until at least num_samples are available.

        Parameters
        ----------
        num_samples : int
            The number of samples to read.

        Returns
        -------
        bytes
            The audio data read from the buffer.
        """
        num_bytes = num_samples * self.config.channels * self.sample_dtype_size

        while True:
            with self.lock:
                read_pos = self.meta_array[0]
                write_pos = self.meta_array[1]

                # Calculate available data
                if write_pos >= read_pos:
                    available = write_pos - read_pos
                else:
                    available = self.buffer_size_bytes - read_pos + write_pos

                if available >= num_bytes:
                    # Read data
                    if read_pos + num_bytes <= self.buffer_size_bytes:
                        # Simple case: data is contiguous
                        data = bytes(self.shm.buf[read_pos : read_pos + num_bytes])
                        read_pos = (read_pos + num_bytes) % self.buffer_size_bytes
                    else:
                        # Wrapping case: combine data from end and beginning
                        first_part = self.buffer_size_bytes - read_pos
                        data = bytes(self.shm.buf[read_pos:]) + bytes(self.shm.buf[: num_bytes - first_part])
                        read_pos = num_bytes - first_part

                    # Update read position
                    self.meta_array[0] = read_pos
                    return data

                # Clear the event since we don't have enough data
                # print("No data available, clearing event")
                self.data_available.clear()

            # Wait for new data to be available (lock is released during wait)
            self.data_available.wait()

    def close(self) -> None:
        """Close the shared memory buffers."""
        try:
            print(f"Closing shared memory buffers with name: {self.shm_name}")
            self.meta_array = None
            self.shm.close()
            self.meta_shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        """Unlink the shared memory buffers from the system."""
        try:
            print(f"Unlinking shared memory buffers with name: {self.shm_name}")
            self.meta_array = None
            self.shm.unlink()
            self.meta_shm.unlink()
        except Exception:
            pass


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

    @staticmethod
    def _audio_capture_process(
        config: AudioConfig,
        buffer: SharedAudioBuffer,
        stop_event: Event,  # type: ignore
        error_queue: Queue,
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

            # Start the audio stream
            with sd.InputStream(callback=audio_callback, **config.to_sounddevice_kwargs()):
                # Keep the stream running until stop event is set
                while not stop_event.is_set():
                    time.sleep(0.1)

        except Exception as e:
            error_queue.put(f"Audio capture process error: {e}")
        finally:
            buffer.close()

    def start_stream(self) -> None:
        """Start the microphone stream in a separate process."""
        if self._streaming:
            return

        # Create shared buffer
        self.buffer = SharedAudioBuffer(self.config)
        print(f"Created shared buffer with name: {self.buffer.shm_name}")

        # Create process control objects
        self.stop_event = Event()
        self.error_queue = Queue()

        # Start audio capture process
        self.process = Process(target=MicrophoneStream._audio_capture_process, args=(self.config, self.buffer, self.stop_event, self.error_queue), daemon=True)
        self.process.start()

        self._streaming = True

        # Wait a bit for the stream to start
        time.sleep(0.1)

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

    def stop_stream(self) -> None:
        """Stop the microphone stream."""
        if not self._streaming:
            return

        # Signal process to stop
        if self.stop_event:
            self.stop_event.set()

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
        self._streaming = False

    def is_streaming(self) -> bool:
        """Check if the stream is currently active."""
        return self._streaming and self.process is not None and self.process.is_alive()

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
        if not self.is_streaming() or self.buffer is None:
            raise RuntimeError("Stream is not active")

        return self.buffer.read(num_samples)

    def read(self, num_samples: int) -> np.ndarray:
        """
        Reads audio data from the stream buffer.
        Blocks until at least num_samples are available.

        Parameters
        ----------
        num_samples : int
            The number of samples to read.

        Returns
        -------
        np.ndarray
            Audio data as numpy array with shape (num_samples, channels).
        """
        raw_data = self.read_raw(num_samples)

        audio_array = np.frombuffer(raw_data, dtype=self.config.dtype)

        # Reshape to (num_samples, channels)
        audio_array = audio_array.reshape(-1, self.config.channels)

        return audio_array
