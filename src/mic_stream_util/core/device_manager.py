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

    DEVICE_IGNORE_LIST = [
        "sysdefault",
        "default",
        "spdif",
        "hdmi",
        "iec958",
        "dmix",
        "dsnoop",
        "null",
        "monitor",
        "pulse",
    ]

    @staticmethod
    def get_devices(
        refresh: bool = False,
        ignore_list: List[str] = DEVICE_IGNORE_LIST,
        check_openable: bool = True,
        include_unavailable: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get a list of available audio input devices.

        Args:
            refresh: Force refresh of device cache
            ignore_list: List of device names to ignore. Defaults to DEVICE_IGNORE_LIST.
            check_openable: Whether to verify devices can be opened
            include_unavailable: Include devices with 0 input channels for debugging
        Returns:
            List of device dictionaries with index and device info
        """
        if DeviceManager._devices_cache is None or refresh:
            try:
                DeviceManager._devices_cache = []
                devices: sd.DeviceList = sd.query_devices()  # type: ignore

                for device in devices:
                    try:
                        index = device["index"]
                        name = device["name"]

                        inchannels = device.get("max_input_channels", 0)

                        # Skip devices with 0 input channels unless debugging
                        if inchannels <= 0 and not include_unavailable:
                            continue

                        if name in ignore_list:
                            continue

                        # Add device status information
                        device_info = {"index": index, **device}

                        # Check if device is currently in use
                        try:
                            import os

                            if os.path.exists(f"/proc/asound/card{index}/pcm0c/sub0/status"):
                                with open(f"/proc/asound/card{index}/pcm0c/sub0/status", "r") as f:
                                    status_content = f.read()
                                    if "state: RUNNING" in status_content:
                                        device_info["status"] = "in_use"
                                        device_info["status_details"] = status_content
                        except Exception:
                            device_info["status"] = "unknown"

                        # Optionally verify the device can be opened with basic settings.
                        # This may trigger ALSA/PortAudio errors if a stream is already open,
                        # so allow callers (e.g., periodic discovery) to disable this check.
                        if check_openable and inchannels > 0:
                            try:
                                sd.check_input_settings(device=index, samplerate=16000, channels=1)
                                device_info["openable"] = True
                            except sd.PortAudioError as e:
                                device_info["openable"] = False
                                device_info["open_error"] = str(e)
                        else:
                            device_info["openable"] = inchannels > 0

                        DeviceManager._devices_cache.append(device_info)
                    except Exception as e:
                        # Log the error but continue with other devices
                        print(f"Error processing device {device.get('name', 'unknown')}: {e}")
                        continue

            except Exception as e:
                raise RuntimeError(f"Failed to query devices: {e}")

        return DeviceManager._devices_cache.copy()

    @staticmethod
    def print_devices(refresh: bool = False, include_unavailable: bool = False) -> None:
        """
        Print a formatted list of available audio input devices.

        Args:
            refresh: Force refresh of device cache
            include_unavailable: Include devices with 0 input channels for debugging
        """
        devices = DeviceManager.get_devices(refresh, include_unavailable=include_unavailable)

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
            status = device.get("status", "unknown")
            openable = device.get("openable", False)

            print(f"[{index:2d}] {name}")
            print(f"     Inputs: {max_inputs}, Default Sample Rate: {default_samplerate}")
            print(f"     Status: {status}, Openable: {openable}")

            # Show additional info if available
            if "hostapi" in device:
                print(f"     Host API: {device['hostapi']}")

            if "open_error" in device:
                print(f"     Open Error: {device['open_error']}")

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

    @staticmethod
    def diagnose_device_issues() -> Dict[str, Any]:
        """
        Diagnose common issues with audio devices.

        Returns:
            Dictionary with diagnostic information
        """
        import os
        import subprocess

        diagnostics = {"alsa_devices": [], "processes_using_audio": [], "device_status": {}, "recommendations": []}

        try:
            # Get ALSA device list
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
            diagnostics["alsa_devices"] = result.stdout.splitlines()
        except Exception as e:
            diagnostics["alsa_devices"] = [f"Error: {e}"]

        try:
            # Find processes using audio devices
            if os.path.exists("/dev/snd"):
                result = subprocess.run(["lsof", "/dev/snd/*"], capture_output=True, text=True)
                diagnostics["processes_using_audio"] = result.stdout.splitlines()
        except Exception as e:
            diagnostics["processes_using_audio"] = [f"Error: {e}"]

        # Check device status
        devices = DeviceManager.get_devices(refresh=True, include_unavailable=True)
        for device in devices:
            diagnostics["device_status"][device["name"]] = {
                "max_input_channels": device.get("max_input_channels", 0),
                "status": device.get("status", "unknown"),
                "openable": device.get("openable", False),
            }

        # Generate recommendations
        if not devices:
            diagnostics["recommendations"].append("No audio input devices detected. Check hardware connections and drivers.")

        in_use_devices = [d for d in devices if d.get("status") == "in_use"]
        if in_use_devices:
            diagnostics["recommendations"].append(f"{len(in_use_devices)} device(s) are currently in use by other applications.")

        unavailable_devices = [d for d in devices if d.get("max_input_channels", 0) == 0]
        if unavailable_devices:
            diagnostics["recommendations"].append(f"{len(unavailable_devices)} device(s) show 0 input channels - they may be in use or have configuration issues.")

        return diagnostics
