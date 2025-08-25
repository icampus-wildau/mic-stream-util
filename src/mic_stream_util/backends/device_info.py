from __future__ import annotations

from typing import TypedDict


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
