"""Pipewire backend for audio device management."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from mic_stream_util.backends.base_backend import DeviceBackend, DeviceInfo, SampleSpecification


class PipewireBackend(DeviceBackend):
    """Pipewire backend implementation for audio device management."""

    def get_backend_name(self) -> str:
        """Get the name of the backend."""
        return "pipewire"

    def backend_is_available(self) -> bool:
        """Check if the pipewire backend is available."""
        try:
            result = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and "PipeWire" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def refresh_devices(self, skip_monitor_sources: bool = True) -> list[DeviceInfo]:
        """Refresh the device cache using pactl command."""
        self.device_cache.clear()

        try:
            # Get list of sources (input devices)
            result = subprocess.run(["pactl", "list", "sources"], capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                raise RuntimeError(f"pactl list sources failed: {result.stderr}")

            # Parse the output
            source_blocks = self._parse_pactl_output(result.stdout)

            for source_block in source_blocks:
                try:
                    device_info = self._parse_source_block(source_block)
                    if device_info:
                        # Skip monitor sources (we only want input devices)
                        if device_info["name"] and ".monitor" in device_info["name"]:
                            if skip_monitor_sources:
                                continue
                            else:
                                device_info["is_monitor"] = True

                        # Skip if we already have this device (avoid duplicates)
                        if device_info["index"] not in self.device_cache:
                            self.device_cache[device_info["index"]] = device_info
                            self.device_cache[device_info["name"]] = device_info

                except Exception as e:
                    print(f"Error processing pipewire source: {e}")
                    continue

        except Exception as e:
            raise RuntimeError(f"Failed to query pipewire devices: {e}")

        return list(self.device_cache.values())

    def _parse_pactl_output(self, output: str) -> list[str]:
        """Parse pactl output into individual source blocks."""
        # Split by "Source #" to get individual source blocks
        source_blocks = []
        current_block = ""

        for line in output.split("\n"):
            if line.strip().startswith("Source #"):
                if current_block.strip():
                    source_blocks.append(current_block.strip())
                current_block = line
            else:
                current_block += "\n" + line

        # Add the last block
        if current_block.strip():
            source_blocks.append(current_block.strip())

        return source_blocks

    def _parse_source_block(self, block: str) -> DeviceInfo | None:
        """Parse a single source block into DeviceInfo."""
        lines = block.split("\n")

        # Extract basic info
        index_match = re.search(r"Source #(\d+)", lines[0])
        if not index_match:
            return None

        index = int(index_match.group(1))

        # Parse all the fields
        name = self._extract_field(block, "Name:")
        description = self._extract_field(block, "Description:")
        driver = self._extract_field(block, "Driver:")
        sample_spec = self._extract_field(block, "Sample Specification:")
        channel_map = self._extract_field(block, "Channel Map:")
        owner_module = self._extract_field(block, "Owner Module:")
        mute = self._extract_field(block, "Mute:")
        volume = self._extract_field(block, "Volume:")
        balance = self._extract_field(block, "balance")
        base_volume = self._extract_field(block, "Base Volume:")
        monitor_of_sink = self._extract_field(block, "Monitor of Sink:")
        latency = self._extract_field(block, "Latency:")
        flags = self._extract_field(block, "Flags:")

        # Convert to proper types
        sample_specification = self._parse_sample_specification(sample_spec)
        channel_map_list = self._parse_channel_map(channel_map)
        mute_bool = mute.lower() == "yes" if mute else False
        volume_dict = self._parse_volume(volume)
        balance_float = float(balance) if balance else 0.0
        base_volume_int = self._parse_volume_value(base_volume)
        latency_int = self._parse_latency(latency)
        flags_list = self._parse_flags(flags)

        return DeviceInfo(
            index=index,
            name=name or f"Source #{index}",
            description=description or name or f"Source #{index}",
            driver=driver or "PipeWire",
            sample_specification=sample_specification,
            channel_map=channel_map_list,
            owner_module=int(owner_module) if owner_module else 0,
            mute=mute_bool,
            volume=volume_dict,
            balance=balance_float,
            base_volume=base_volume_int,
            monitor_of_sink=monitor_of_sink if monitor_of_sink != "n/a" else None,
            latency=latency_int,
            flags=flags_list,
        )

    def _extract_field(self, block: str, field_name: str) -> str | None:
        """Extract a field value from the block."""
        pattern = rf"{re.escape(field_name)}\s*(.+)"
        match = re.search(pattern, block, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _parse_sample_specification(self, sample_spec: str | None) -> SampleSpecification:
        """Parse sample specification string."""
        if not sample_spec:
            return SampleSpecification(sample_format="s16le", sample_rate_hz=44100, channels=1)

        # Example: "s16le 3ch 16000Hz"
        pattern = r"(\w+)\s+(\d+)ch\s+(\d+)Hz"
        match = re.search(pattern, sample_spec)

        if match:
            return SampleSpecification(sample_format=match.group(1), sample_rate_hz=int(match.group(3)), channels=int(match.group(2)))

        # Fallback
        return SampleSpecification(sample_format="s16le", sample_rate_hz=44100, channels=1)

    def _parse_channel_map(self, channel_map: str | None) -> list[str]:
        """Parse channel map string."""
        if not channel_map:
            return ["mono"]

        # Example: "front-left,front-right,lfe"
        return [ch.strip() for ch in channel_map.split(",")]

    def _parse_volume(self, volume: str | None) -> dict[str, int]:
        """Parse volume string."""
        if not volume:
            return {"front-left": 65536, "front-right": 65536}

        volume_dict = {}
        # Example: "front-left: 65536 / 100% / 0.00 dB, front-right: 65536 / 100% / 0.00 dB"
        pattern = r"(\w+(?:-\w+)*):\s*(\d+)"
        matches = re.findall(pattern, volume)

        for channel, value in matches:
            volume_dict[channel] = int(value)

        return volume_dict if volume_dict else {"front-left": 65536, "front-right": 65536}

    def _parse_volume_value(self, volume_str: str | None) -> int:
        """Parse volume value string."""
        if not volume_str:
            return 65536

        # Example: "65536 / 100% / 0.00 dB"
        pattern = r"(\d+)"
        match = re.search(pattern, volume_str)
        return int(match.group(1)) if match else 65536

    def _parse_latency(self, latency: str | None) -> int:
        """Parse latency string."""
        if not latency:
            return 0

        # Example: "0 usec, configured 0 usec"
        pattern = r"(\d+)\s+usec"
        match = re.search(pattern, latency)
        return int(match.group(1)) if match else 0

    def _parse_flags(self, flags: str | None) -> list[str]:
        """Parse flags string."""
        if not flags:
            return []

        # Example: "HARDWARE DECIBEL_VOLUME LATENCY"
        return [flag.strip() for flag in flags.split()]

    def get_source_info(self, source_index: int) -> dict[str, Any]:
        """Get detailed information about a specific source using pactl."""
        try:
            result = subprocess.run(["pactl", "list", "sources"], capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                raise RuntimeError(f"pactl list sources {source_index} failed: {result.stderr}")

            # Parse the output and find the specific source
            source_blocks = self._parse_pactl_output(result.stdout)

            for source_block in source_blocks:
                device_info = self._parse_source_block(source_block)
                if device_info and device_info["index"] == source_index:
                    return dict(device_info)

            raise ValueError(f"Source {source_index} not found")

        except Exception as e:
            raise RuntimeError(f"Failed to get source info for {source_index}: {e}")

    def set_source_volume(self, source_index: int, volume_percent: float) -> bool:
        """Set the volume of a source."""
        try:
            # Convert percentage to pipewire volume (0-65536)
            volume = int((volume_percent / 100.0) * 65536)

            result = subprocess.run(["pactl", "set-source-volume", str(source_index), str(volume)], capture_output=True, text=True, timeout=5)

            return result.returncode == 0

        except Exception as e:
            print(f"Failed to set source volume: {e}")
            return False

    def set_source_mute(self, source_index: int, mute: bool) -> bool:
        """Set the mute state of a source."""
        try:
            mute_arg = "1" if mute else "0"

            result = subprocess.run(["pactl", "set-source-mute", str(source_index), mute_arg], capture_output=True, text=True, timeout=5)

            return result.returncode == 0

        except Exception as e:
            print(f"Failed to set source mute: {e}")
            return False

    def set_default_source(self, source_index: int) -> bool:
        """Set a source as the default input device."""
        try:
            # Get the source name
            source_info = self.get_source_info(source_index)
            source_name = source_info["name"]

            # Set as default source
            result = subprocess.run(["pactl", "set-default-source", source_name], capture_output=True, text=True, timeout=5)

            return result.returncode == 0

        except Exception as e:
            print(f"Failed to set default source: {e}")
            return False

    def get_pipewire_device_index(self) -> int | None:
        """Get the sounddevice index for the 'pipewire' device."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device["name"] == "pipewire":
                    return i

            return None

        except Exception as e:
            print(f"Failed to get pipewire device index: {e}")
            return None

    def route_source_to_default(self, source_index: int) -> bool:
        """Route a specific source to be the default input for sounddevice."""
        try:
            # First, set this source as the default in Pipewire
            if not self.set_default_source(source_index):
                return False

            # Get the pipewire device index for sounddevice
            pipewire_device_index = self.get_pipewire_device_index()
            if pipewire_device_index is None:
                print("Warning: Could not find 'pipewire' device in sounddevice")
                return True  # Still return True as the source is set as default

            return True

        except Exception as e:
            print(f"Failed to route source to default: {e}")
            return False

    def prepare_device_for_streaming(self, device_identifier: int | str) -> dict:
        """
        Prepare a device for streaming with pipewire.
        For pipewire, we route the source to default and return the pipewire device index.
        """
        device_info = self.get_device_info(device_identifier)
        source_index = device_info["index"]

        # Route the source to be the default
        if not self.route_source_to_default(source_index):
            raise RuntimeError(f"Failed to route source {source_index} to default")

        # Get the pipewire device index for sounddevice
        pipewire_device_index = self.get_pipewire_device_index()
        if pipewire_device_index is None:
            raise RuntimeError("Could not find 'pipewire' device in sounddevice")

        return {"device": pipewire_device_index}
