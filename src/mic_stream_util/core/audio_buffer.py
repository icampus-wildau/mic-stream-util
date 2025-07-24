from __future__ import annotations

from multiprocessing import Event, Lock, shared_memory

import numpy as np

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
            # print(f"Creating shared memory for buffer with name: {self.shm_name} and size: {self.buffer_size_bytes}")
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
            read_pos = self.meta_array[0]  # type: ignore
            write_pos = self.meta_array[1]  # type: ignore

            # Write data to buffer
            if write_pos + data_len <= self.buffer_size_bytes:
                # Simple case: data fits without wrapping
                self.shm.buf[write_pos : write_pos + data_len] = data
                write_pos = (write_pos + data_len) % self.buffer_size_bytes
            else:
                # Wrapping case: split data across buffer end
                first_part = self.buffer_size_bytes - write_pos
                try:
                    self.shm.buf[write_pos : write_pos + first_part] = data[:first_part]
                except Exception as e:
                    print(f"Error writing data to buffer: {e}")
                    print(f"write_pos: {write_pos}, data_len: {data_len}, first_part: {first_part}")
                    print(f"data part: {len(data[:first_part])}")
                    print(f"shm.buf: {len(self.shm.buf)}")
                    print(f"shm.buf part: {len(self.shm.buf[write_pos : write_pos + first_part])}")

                    raise e
                self.shm.buf[: data_len - first_part] = data[first_part:]
                write_pos = data_len - first_part

            # Update write position
            self.meta_array[1] = write_pos  # type: ignore

            # If buffer is full, advance read position (drop oldest data)
            if (write_pos + 1) % self.buffer_size_bytes == read_pos:
                read_pos = (read_pos + data_len) % self.buffer_size_bytes
                self.meta_array[0] = read_pos  # type: ignore

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
                read_pos = self.meta_array[0]  # type: ignore
                write_pos = self.meta_array[1]  # type: ignore

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
                    self.meta_array[0] = read_pos  # type: ignore
                    return data

                # Clear the event since we don't have enough data
                # print("No data available, clearing event")
                self.data_available.clear()

            # Wait for new data to be available (lock is released during wait)
            self.data_available.wait()

    def close(self) -> None:
        """Close the shared memory buffers."""
        try:
            # print(f"Closing shared memory buffers with name: {self.shm_name}")
            self.meta_array = None
            self.shm.close()
            self.meta_shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        """Unlink the shared memory buffers from the system."""
        try:
            # print(f"Unlinking shared memory buffers with name: {self.shm_name}")
            self.meta_array = None
            self.shm.unlink()
            self.meta_shm.unlink()
        except Exception:
            pass
