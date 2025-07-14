#!/usr/bin/env python3
"""Main CLI entry point for microphone-util library."""

from __future__ import annotations

import click
from click_aliases import ClickAliasedGroup

from .commands import cpu_usage, device_info, devices, latency_test, loopback, memory_usage, monitor, record, spectrum, vad

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(cls=ClickAliasedGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option()
def main():
    """Microphone Stream Util CLI - Multiprocessing microphone helper with VAD support."""
    pass


# Add all commands to the main group
main.add_command(devices.devices, aliases=["d", "ls", "list"])
main.add_command(device_info.device_info, aliases=["i", "info"])
main.add_command(record.record, aliases=["r", "rec"])
main.add_command(monitor.monitor, aliases=["m", "mon"])
main.add_command(vad.vad, aliases=["v", "vad"])
main.add_command(spectrum.spectrum, aliases=["s", "spec"])
main.add_command(loopback.loopback)
main.add_command(cpu_usage.cpu_usage)
main.add_command(memory_usage.memory_usage)
main.add_command(latency_test.latency_test)


if __name__ == "__main__":
    main()
