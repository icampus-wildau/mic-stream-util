"""Devices command module."""

from __future__ import annotations

import json
from typing import Optional

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.option("--threshold", "-t", type=int, default=70, help="Fuzzy search threshold (0-100, default: 70)")
@click.argument("filter", required=False)
def devices(json_output: bool, filter: Optional[str], threshold: int):
    """List available audio input devices."""
    # Dynamic imports for better response time
    try:
        from mic_stream_util.core.device_manager import DeviceManager

        devices = DeviceManager.get_devices()

        # Apply fuzzy filter if provided
        if filter:
            from fuzzywuzzy import fuzz

            filtered_devices = []
            for device in devices:
                score = max(fuzz.ratio(filter.lower(), device["name"].lower()), fuzz.partial_ratio(filter.lower(), device["name"].lower()))
                if score >= threshold:
                    device["match_score"] = score
                    filtered_devices.append(device)

            # Sort by match score (highest first)
            filtered_devices.sort(key=lambda x: x["match_score"], reverse=True)
            devices = filtered_devices

            if not devices:
                click.echo(f"No devices found matching '{filter}' with threshold {threshold}")
                return

        if json_output:
            # Clean up devices for JSON output
            json_devices = []
            for device in devices:
                json_device = {
                    "index": device["index"],
                    "name": device["name"],
                    "max_input_channels": device["max_input_channels"],
                    "default_samplerate": device.get("default_samplerate", "Unknown"),
                    "hostapi": device.get("hostapi", "Unknown"),
                }
                if "match_score" in device:
                    json_device["match_score"] = device["match_score"]
                json_devices.append(json_device)

            click.echo(json.dumps(json_devices, indent=2))
        else:
            if filter:
                click.echo(f"\nFiltered Audio Input Devices (filter: '{filter}', threshold: {threshold}):")
            else:
                click.echo(f"\nAvailable Audio Input Devices ({len(devices)} found):")
            click.echo("-" * 80)

            for device in devices:
                index = device["index"]
                name = device["name"]
                max_inputs = device["max_input_channels"]
                default_samplerate = device.get("default_samplerate", "Unknown")

                if "match_score" in device:
                    click.echo(f"[{index:2d}] {name} (score: {device['match_score']})")
                else:
                    click.echo(f"[{index:2d}] {name}")
                click.echo(f"     Inputs: {max_inputs}, Default Sample Rate: {default_samplerate}")

                # Show additional info if available
                if "hostapi" in device:
                    click.echo(f"     Host API: {device['hostapi']}")

                click.echo()

    except Exception as e:
        click.echo(f"Error listing devices: {e}", err=True)
        raise click.Abort()
