"""Audio configuration classes for microphone management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .device_manager import DeviceManager


@dataclass
class AudioConfig:
    """Configuration for audio input/output settings."""

    # Sample rate for the audio stream
    sample_rate: int = 16000

    # Number of channels for the audio stream
    channels: int = 1

    # Data type for the audio stream. Supported types are "float32" and "int16"
    dtype: str = "float32"

    # Blocksize for the audio stream fetching. If not specified, it is set to 1/10 of sample_rate
    blocksize: int = None  # type: ignore

    # The device index for the audio stream
    device: Optional[int] = None

    # The name of the device for the audio stream
    device_name: Optional[str] = None

    # The latency for the audio stream. Supported values are "low" and "high"
    latency: str = "low"

    # The number of samples to be processed at a time in callbacks.
    num_samples: int = 512

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

        # Set blocksize to 1/10 of sample_rate if not specified
        if self.blocksize is None:
            self.blocksize = self.sample_rate // 10

        if self.blocksize <= 0:
            raise ValueError("blocksize must be positive")

        # Set device from device_name if specified
        if self.device_name is not None:
            device = DeviceManager.find_device(self.device_name)
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
