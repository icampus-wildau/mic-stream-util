from abc import ABC, abstractmethod
from typing import TypedDict

from mic_stream_util.util.fuzzy_match import FuzzySearch


class SampleSpecification(TypedDict):
    sample_format: str
    sample_rate_hz: int
    channels: int


class DeviceInfo(TypedDict):
    index: int
    name: str
    description: str
    driver: str
    sample_specification: SampleSpecification
    channel_map: list[str]
    owner_module: int
    mute: bool
    volume: dict[str, int]
    balance: float
    base_volume: int
    monitor_of_sink: str | None
    latency: int
    flags: list[str]


class BaseDeviceBackend(ABC):
    device_cache: dict[int | str, DeviceInfo] = {}
    """
    A dictionary of devices, indexed by their name or index.
    This cache should be at least contain the devices mapped by their index, the name is optional.
    """

    def __init__(self):
        self.device_cache = {}
        self.refresh_devices()

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
        return list(self.device_cache.values())

    def get_device_info(self, device: int | str, refresh: bool = False) -> DeviceInfo:
        """Get information about a specific device."""
        if refresh:
            self.refresh_devices()

        if device in self.device_cache:
            return self.device_cache[device]

        if isinstance(device, str):
            devices = self.get_all_devices()
            device_names = [device["name"] for device in devices]
            index, name = FuzzySearch.find_best_match(device, device_names)
            if index is not None:
                return devices[index]

        raise ValueError(f"Device {device} not found")

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
