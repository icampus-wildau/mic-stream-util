"""Audio configuration classes for microphone management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..backends import DeviceBackend


@dataclass
class AudioConfig:
    """Configuration for audio input/output settings."""

    # Static dtype map for size calculations
    ALLOWED_DTYPES = ["float32", "int32", "int16", "int8", "uint8"]

    # Sample rate for the audio stream
    sample_rate: int = 16000

    # Number of channels for the audio stream
    channels: int = 1

    # Data type for the audio stream. Supported types are 'float32', 'int32', 'int16', 'int8' and 'uint8'. See https://python-sounddevice.readthedocs.io/en/0.3.12/api.html#sounddevice.default.dtype
    dtype: str = "float32"

    # Blocksize for the audio stream fetching. If not specified, it is set to 1/10 of sample_rate
    blocksize: int = None  # type: ignore

    # Buffer size for the audio stream fetching. If not specified, it is set to 10 * sample_rate
    buffer_size: Optional[int] = None

    # The device index for the audio stream
    device: Optional[int] = None

    # The name of the device for the audio stream
    device_name: Optional[str] = None

    # The latency for the audio stream. Supported values are "low" and "high"
    latency: str = "low"

    # The number of samples to be processed at a time in callbacks (or as default value for read calls).
    num_samples: int = 512

    @classmethod
    def get_dtype_size(cls, dtype: str) -> int:
        """Get the size in bytes for a given dtype."""
        if dtype not in cls.ALLOWED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return np.dtype(dtype).itemsize

    def __post_init__(self) -> None:
        """Validate configuration parameters and set device."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.channels <= 0:
            raise ValueError("Channels must be positive")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.latency not in ["low", "high"]:
            raise ValueError("Latency must be 'low' or 'high'")

        if self.dtype not in ["float32", "int32", "int16", "int8", "uint8"]:
            raise ValueError("dtype must be 'float32', 'int32', 'int16', 'int8' or 'uint8'")

        # Set buffer_size to 10 * sample_rate if not specified
        if self.buffer_size is None:
            self.buffer_size = self.sample_rate * 10

        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        # Set blocksize to 1/10 of sample_rate if not specified
        if self.blocksize is None:
            self.blocksize = self.sample_rate // 10

        if self.blocksize <= 0:
            raise ValueError("blocksize must be positive")

        # Set device from device_name if specified
        if self.device_name is not None:
            backend = DeviceBackend.get_backend()
            device = backend.get_device_info(self.device_name)
            self.device = device["index"]

    def to_sounddevice_kwargs(self) -> dict:
        """Convert to sounddevice stream parameters."""
        return {
            "samplerate": self.sample_rate,
            "channels": self.channels,
            "dtype": self.dtype,
            "blocksize": self.blocksize,
            "device": self.device,
            "latency": self.latency,
        }

    @classmethod
    def from_sounddevice_kwargs(cls, **kwargs) -> AudioConfig:
        """Create from sounddevice parameters."""
        return cls(
            sample_rate=kwargs.get("samplerate", 16000),
            channels=kwargs.get("channels", 1),
            dtype=kwargs.get("dtype", "float32"),
            blocksize=kwargs.get("blocksize", kwargs.get("samplerate", 16000) // 10),
            device=kwargs.get("device"),
            device_name=kwargs.get("device_name"),
            latency=kwargs.get("latency", "low"),
            num_samples=kwargs.get("num_samples", 1024),
        )
