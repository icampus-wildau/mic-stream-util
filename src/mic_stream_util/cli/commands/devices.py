"""Devices command module."""

from __future__ import annotations

import json
from typing import Optional

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.option("--threshold", "-t", type=int, default=70, help="Fuzzy search threshold (0-100, default: 70)")
@click.option("--include-unavailable", is_flag=True, help="Include devices with 0 input channels for debugging")
@click.argument("filter", required=False)
def devices(json_output: bool, filter: Optional[str], threshold: int, include_unavailable: bool):
    """List available audio input devices."""
    # Dynamic imports for better response time
    try:
        from mic_stream_util.core.device_manager import DeviceManager

        devices = DeviceManager.get_devices(include_unavailable=include_unavailable)

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
                    "status": device.get("status", "unknown"),
                    "openable": device.get("openable", False),
                }
                if "match_score" in device:
                    json_device["match_score"] = device["match_score"]
                if "open_error" in device:
                    json_device["open_error"] = device["open_error"]
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
                status = device.get("status", "unknown")
                openable = device.get("openable", False)

                if "match_score" in device:
                    click.echo(f"[{index:2d}] {name} (score: {device['match_score']})")
                else:
                    click.echo(f"[{index:2d}] {name}")
                click.echo(f"     Inputs: {max_inputs}, Default Sample Rate: {default_samplerate}")
                click.echo(f"     Status: {status}, Openable: {openable}")

                # Show additional info if available
                if "hostapi" in device:
                    click.echo(f"     Host API: {device['hostapi']}")

                if "open_error" in device:
                    click.echo(f"     Open Error: {device['open_error']}")

                click.echo()

    except Exception as e:
        click.echo(f"Error listing devices: {e}", err=True)
        raise click.Abort()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def diagnose(json_output: bool):
    """Diagnose audio device issues."""
    try:
        from mic_stream_util.core.device_manager import DeviceManager

        diagnostics = DeviceManager.diagnose_device_issues()

        if json_output:
            click.echo(json.dumps(diagnostics, indent=2))
        else:
            click.echo("Audio Device Diagnostics")
            click.echo("=" * 50)

            click.echo("\nALSA Devices:")
            click.echo("-" * 20)
            for line in diagnostics["alsa_devices"]:
                click.echo(line)

            click.echo("\nProcesses Using Audio:")
            click.echo("-" * 25)
            if diagnostics["processes_using_audio"]:
                for line in diagnostics["processes_using_audio"]:
                    click.echo(line)
            else:
                click.echo("No processes currently using audio devices")

            click.echo("\nDevice Status:")
            click.echo("-" * 15)
            for device_name, status in diagnostics["device_status"].items():
                click.echo(f"{device_name}:")
                click.echo(f"  Input Channels: {status['max_input_channels']}")
                click.echo(f"  Status: {status['status']}")
                click.echo(f"  Openable: {status['openable']}")
                click.echo()

            click.echo("\nRecommendations:")
            click.echo("-" * 15)
            if diagnostics["recommendations"]:
                for i, rec in enumerate(diagnostics["recommendations"], 1):
                    click.echo(f"{i}. {rec}")
            else:
                click.echo("No issues detected")

    except Exception as e:
        click.echo(f"Error running diagnostics: {e}", err=True)
        raise click.Abort()
