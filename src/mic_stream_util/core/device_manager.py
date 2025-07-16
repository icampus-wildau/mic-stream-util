"""Device management for audio input devices."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import sounddevice as sd

from mic_stream_util.exceptions import DeviceNotFoundError
from mic_stream_util.util.fuzzy_match import find_best_match


class DeviceInfo(TypedDict):
    index: int
    name: str
    max_input_channels: int
    default_samplerate: int
    hostapi: str
    supported_samplerates: List[int]


class DeviceManager:
    """Manages audio input device discovery and selection."""

    _devices_cache: Optional[List[Dict[str, Any]]] = None

    @staticmethod
    def get_devices(refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get a list of available audio input devices.

        Args:
            refresh: Force refresh of device cache

        Returns:
            List of device dictionaries with index and device info
        """
        if DeviceManager._devices_cache is None or refresh:
            try:
                devices: sd.DeviceList = sd.query_devices()  # type: ignore
                DeviceManager._devices_cache = [{"index": i, **device} for i, device in enumerate(devices) if device.get("max_input_channels", 0) > 0]
            except Exception as e:
                raise RuntimeError(f"Failed to query devices: {e}")

        return DeviceManager._devices_cache.copy()

    @staticmethod
    def print_devices(refresh: bool = False) -> None:
        """
        Print a formatted list of available audio input devices.

        Args:
            refresh: Force refresh of device cache
        """
        devices = DeviceManager.get_devices(refresh)

        if not devices:
            print("No audio input devices found.")
            return

        print(f"\nAvailable Audio Input Devices ({len(devices)} found):")
        print("-" * 80)

        for device in devices:
            index = device["index"]
            name = device["name"]
            max_inputs = device["max_input_channels"]
            default_samplerate = device.get("default_samplerate", "Unknown")

            print(f"[{index:2d}] {name}")
            print(f"     Inputs: {max_inputs}, Default Sample Rate: {default_samplerate}")

            # Show additional info if available
            if "hostapi" in device:
                print(f"     Host API: {device['hostapi']}")

            print()

    @staticmethod
    def get_device_by_index(device_index: int) -> Dict[str, Any] | None:
        """
        Get device information by index.

        Args:
            device_index: Index of the device

        Returns:
            Device dictionary or None if not found
        """
        try:
            devices: sd.DeviceList = sd.query_devices()  # type: ignore
            if 0 <= device_index < len(devices):
                return {"index": device_index, **(devices[device_index])}
            return None
        except Exception as e:
            raise RuntimeError(f"Error querying device {device_index}: {e}")

    @staticmethod
    def find_device(device_identifier: str | int) -> Dict[str, Any]:
        """
        Find a device by name (fuzzy search) or index.

        Args:
            device_identifier: Device name (string) or index (integer)

        Returns:
            Device dictionary

        Raises:
            DeviceNotFoundError: If device is not found
        """
        if isinstance(device_identifier, int):
            device = DeviceManager.get_device_by_index(device_identifier)
            if device is None:
                raise DeviceNotFoundError(f"Device with index {device_identifier} not found")
            return device

        # Otherwise, try to find the device by name
        try:
            devices: sd.DeviceList = sd.query_devices()  # type: ignore

            candidates = [{"index": i, **device} for i, device in enumerate(devices)]

            result = find_best_match(device_identifier, candidates)
            if result:
                return result[1]

            raise DeviceNotFoundError(f"Device '{device_identifier}' not found")
        except Exception as e:
            raise RuntimeError(f"Failed to query devices: {e}")

    @staticmethod
    def get_default_device() -> Optional[Dict[str, Any]]:
        """
        Get the default input device.

        Returns:
            Default device dictionary or None if no default device
        """
        try:
            default_device: dict[str, Any] = sd.query_devices(kind="input")  # type: ignore
            devices = DeviceManager.get_devices()

            # Find the default device in our list
            for device in devices:
                if device["name"] == default_device["name"]:
                    return device

            return None
        except Exception:
            return None

    @staticmethod
    def get_device_info(device_identifier: str | int) -> DeviceInfo:
        """
        Get detailed information about a specific device.

        Args:
            device_identifier: Device name or index

        Returns:
            Detailed device information
        """
        device = DeviceManager.find_device(device_identifier)

        # Add additional info
        info = device.copy()
        info["supported_samplerates"] = DeviceManager._get_supported_samplerates(device["index"])

        return info

    @staticmethod
    def _get_supported_samplerates(device_index: int) -> List[int]:
        """Get supported sample rates for a device."""
        common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
        supported = []

        for rate in common_rates:
            try:
                sd.check_input_settings(device=device_index, samplerate=rate)
                supported.append(rate)
            except sd.PortAudioError:
                continue

        return supported
