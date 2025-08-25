"""Sounddevice backend for audio device management."""

from __future__ import annotations

from typing import Any

import sounddevice as sd

from mic_stream_util.backends.base_backend import DeviceBackend, DeviceInfo, SampleSpecification


class SounddeviceBackend(DeviceBackend):
    """Sounddevice backend implementation for audio device management."""

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
        # "pulse",
    ]

    def get_backend_name(self) -> str:
        """Get the name of the backend."""
        return "sounddevice"

    def backend_is_available(self) -> bool:
        """Check if the sounddevice backend is available."""
        try:
            sd.query_devices()
            return True
        except Exception:
            return False

    def refresh_devices(self) -> list[DeviceInfo]:
        """Refresh the device cache using sounddevice."""
        self.device_cache.clear()

        try:
            devices: sd.DeviceList = sd.query_devices()  # type: ignore

            for device in devices:
                try:
                    index = device["index"]
                    name = device["name"]

                    # Skip devices with 0 input channels or in ignore list
                    inchannels = device.get("max_input_channels", 0)
                    if inchannels <= 0 or name in self.DEVICE_IGNORE_LIST:
                        continue

                    # Skip output-only devices (we only want input devices)
                    outchannels = device.get("max_output_channels", 0)
                    if inchannels == 0 and outchannels > 0:
                        continue

                    # Skip if we already have this device (avoid duplicates)
                    if index in self.device_cache:
                        continue

                    # Convert sounddevice device info to our DeviceInfo format
                    device_info = self._convert_sd_device_to_device_info(device)
                    self.device_cache[index] = device_info
                    self.device_cache[name] = device_info

                except Exception as e:
                    # Log the error but continue with other devices
                    print(f"Error processing sounddevice {device.get('name', 'unknown')}: {e}")
                    continue

        except Exception as e:
            raise RuntimeError(f"Failed to query sounddevice devices: {e}")

        return list(self.device_cache.values())

    def _convert_sd_device_to_device_info(self, sd_device: dict[str, Any]) -> DeviceInfo:
        """Convert sounddevice device info to our DeviceInfo format."""
        # Parse sample specification
        sample_spec = self._parse_sample_specification(sd_device)

        # Parse volume information
        volume_info = self._parse_volume_info(sd_device)

        return DeviceInfo(
            index=sd_device["index"],
            name=sd_device["name"],
            description=sd_device.get("name", ""),  # sounddevice doesn't have separate description
            driver=self._get_hostapi_name(sd_device.get("hostapi", 0)),
            sample_specification=sample_spec,
            channel_map=self._get_channel_map(sd_device),
            owner_module=0,  # sounddevice doesn't provide this
            mute=False,  # sounddevice doesn't provide mute status
            volume=volume_info["volume"],
            balance=volume_info["balance"],
            base_volume=volume_info["base_volume"],
            monitor_of_sink=None,  # sounddevice doesn't provide this
            latency=sd_device.get("default_low_input_latency", 0),
            flags=self._get_device_flags(sd_device),
        )

    def _parse_sample_specification(self, sd_device: dict[str, Any]) -> SampleSpecification:
        """Parse sample specification from sounddevice device info."""
        # sounddevice doesn't provide detailed sample format info
        # We'll use default values and let the user check supported formats
        return SampleSpecification(
            sample_format="s16le",  # Default assumption
            sample_rate_hz=sd_device.get("default_samplerate", 44100),
            channels=sd_device.get("max_input_channels", 1),
        )

    def _parse_volume_info(self, sd_device: dict[str, Any]) -> dict[str, Any]:
        """Parse volume information from sounddevice device info."""
        # sounddevice doesn't provide volume information
        # Return default values
        return {
            "volume": {"front-left": 65536, "front-right": 65536},  # 100% volume
            "balance": 0.0,
            "base_volume": 65536,
        }

    def _get_channel_map(self, sd_device: dict[str, Any]) -> list[str]:
        """Get channel map for the device."""
        channels = sd_device.get("max_input_channels", 1)
        if channels == 1:
            return ["mono"]
        elif channels == 2:
            return ["front-left", "front-right"]
        else:
            # For multi-channel devices, create generic channel names
            return [f"channel-{i}" for i in range(channels)]

    def _get_hostapi_name(self, hostapi_id: int) -> str:
        """Get host API name from ID."""
        try:
            hostapis = sd.query_hostapis()
            if 0 <= hostapi_id < len(hostapis):
                return hostapis[hostapi_id]["name"]
        except Exception:
            pass
        return "Unknown"

    def _get_device_flags(self, sd_device: dict[str, Any]) -> list[str]:
        """Get device flags from sounddevice device info."""
        flags = []

        # Check if device is hardware
        if sd_device.get("max_input_channels", 0) > 0:
            flags.append("HARDWARE")

        # Check if device is default
        try:
            default_device = sd.query_devices(kind="input")
            if sd_device["name"] == default_device["name"]:
                flags.append("DEFAULT")
        except Exception:
            pass

        return flags

    def get_supported_sample_rates(self, device: int | str) -> list[int]:
        """Get supported sample rates for a specific device."""
        device_info = self.get_device_info(device)
        device_index = device_info["index"]

        common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
        supported = []

        for rate in common_rates:
            try:
                sd.check_input_settings(device=device_index, samplerate=rate)
                supported.append(rate)
            except sd.PortAudioError:
                continue

        return supported

    def check_device_openable(self, device: int | str, sample_rate: int = 16000, channels: int = 1) -> bool:
        """Check if a device can be opened with specific settings."""
        device_info = self.get_device_info(device)
        device_index = device_info["index"]

        try:
            sd.check_input_settings(device=device_index, samplerate=sample_rate, channels=channels)
            return True
        except sd.PortAudioError:
            return False

    def prepare_device_for_streaming(self, device_identifier: int | str) -> dict:
        """
        Prepare a device for streaming with sounddevice.
        For sounddevice, we just return the device index.
        """
        device_info = self.get_device_info(device_identifier)
        return {"device": device_info["index"]}
