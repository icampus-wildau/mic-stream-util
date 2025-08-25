from __future__ import annotations

import re
from abc import ABC, abstractmethod

from mic_stream_util.util.fuzzy_match import FuzzySearch

from .device_info import DeviceInfo, SampleSpecification


class DeviceBackend(ABC):
    device_cache: dict[int | str, DeviceInfo] = {}
    """
    A dictionary of devices, indexed by their name or index.
    This cache should be at least contain the devices mapped by their index, the name is optional.
    """

    def __init__(self):
        self.device_cache = {}
        self.refresh_devices()

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of the backend."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_backend(cls, backend_name: str | None = None) -> DeviceBackend:
        """
        Get the best available backend or a specific backend.

        Args:
            backend_name: Optional backend name ("pipewire" or "sounddevice").
                         If None, tries pipewire first, then sounddevice.

        Returns:
            DeviceBackend instance

        Raises:
            RuntimeError: If no backend is available
        """
        if backend_name is None:
            # Try Pipewire first (preferred)
            try:
                from .pipewire import PipewireBackend

                backend = PipewireBackend()
                if backend.backend_is_available():
                    return backend
            except Exception:
                pass

            # Fallback to Sounddevice
            try:
                from .sounddevice import SounddeviceBackend

                backend = SounddeviceBackend()
                if backend.backend_is_available():
                    return backend
            except Exception:
                pass

            raise RuntimeError("No audio backend available. Neither Pipewire nor Sounddevice backends are working.")

        elif backend_name.lower() == "pipewire":
            from .pipewire import PipewireBackend

            backend = PipewireBackend()
            if not backend.backend_is_available():
                raise RuntimeError("Pipewire backend is not available")
            return backend

        elif backend_name.lower() == "sounddevice":
            from .sounddevice import SounddeviceBackend

            backend = SounddeviceBackend()
            if not backend.backend_is_available():
                raise RuntimeError("Sounddevice backend is not available")
            return backend

        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    @abstractmethod
    def prepare_device_for_streaming(self, device_identifier: int | str) -> dict:
        """
        Prepare a device for streaming. This method should handle any backend-specific
        setup needed before opening an audio stream.

        Args:
            device_identifier: Device index or name

        Returns:
            Dictionary with streaming parameters (e.g., device index for sounddevice)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def backend_is_available(self) -> bool:
        """Check if the backend is available."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def refresh_devices(self) -> list[DeviceInfo]:
        """
        Refresh the device cache.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_all_devices(self, refresh: bool = False) -> list[DeviceInfo]:
        """Get all available devices."""
        if refresh:
            self.refresh_devices()

        # Remove duplicates by index
        unique_devices = {}
        for device in self.device_cache.values():
            unique_devices[device["index"]] = device

        return list(unique_devices.values())

    def get_device_info(self, device: int | str, refresh: bool = False) -> DeviceInfo:
        """Get information about a specific device."""
        if refresh:
            self.refresh_devices()

        if device in self.device_cache:
            return self.device_cache[device]

        # Try to convert string to int if it looks like a number
        if isinstance(device, str):
            try:
                device_int = int(device)
                if device_int in self.device_cache:
                    return self.device_cache[device_int]
            except ValueError:
                pass

        if isinstance(device, str):
            devices = self.get_all_devices()
            device_names = [device["name"] for device in devices]
            index, name = FuzzySearch.find_best_match(device, device_names)
            if index is not None:
                return devices[index]

        raise ValueError(f"Device {device} not found")

    def get_default_device(self) -> DeviceInfo | None:
        """Get the default input device."""
        devices = self.get_all_devices()
        if not devices:
            return None

        # Look for a device with DEFAULT flag
        for device in devices:
            if "DEFAULT" in device.get("flags", []):
                return device

        # Return the first available device
        return devices[0]

    @staticmethod
    def get_short_name_from_device_info(device_info: DeviceInfo) -> str:
        """Get a short name from device info by cleaning the device name."""

        return device_info["name"]

        # TODO: This is too easy
        # name = device_info["name"]
        # short_name = name
        # # Extract the name by ignoring special chars
        # pattern = r"(.*?)([^\w\s]|$)"
        # match = re.match(pattern, name)
        # if match:
        #     short_name = match.group(1)
        # short_name = short_name.strip().replace(" ", "_")
        # return short_name

    def print_devices(self) -> None:
        """Print all available devices."""
        devices = self.get_all_devices()

        if not devices:
            print("No devices found")
            return

        print(f"\nFound {len(devices)} devices:")
        print("-" * 80)

        for device in devices:
            print(f"[{device['index']}] {device['name']}")
            print(f"     Description: {device['description']}")
            print(f"     Sample Specification: {device['sample_specification']}")
