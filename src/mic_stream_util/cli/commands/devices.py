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
        from mic_stream_util.backends import DeviceBackend

        backend = DeviceBackend.get_backend()
        devices = backend.get_all_devices()

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
                    "description": device.get("description", ""),
                    "driver": device.get("driver", "Unknown"),
                    "sample_specification": device.get("sample_specification", {}),
                    "channels": device.get("sample_specification", {}).get("channels", 0),
                    "sample_rate": device.get("sample_specification", {}).get("sample_rate_hz", 0),
                    "flags": device.get("flags", []),
                }
                if "match_score" in device:
                    json_device["match_score"] = device["match_score"]
                json_devices.append(json_device)

            click.echo(json.dumps(json_devices, indent=2))
        else:
            if filter:
                click.echo(f"\nFiltered Audio Input Devices (filter: '{filter}', threshold: {threshold}):")
            else:
                click.echo(f"\nAvailable Audio Input Devices ({len(devices)} found) - Backend: {backend.get_backend_name()}")
            click.echo("-" * 80)

            for device in devices:
                index = device["index"]
                name = device["name"]
                description = device.get("description", "")
                driver = device.get("driver", "Unknown")
                sample_spec = device.get("sample_specification", {})
                channels = sample_spec.get("channels", 0)
                sample_rate = sample_spec.get("sample_rate_hz", 0)
                flags = device.get("flags", [])

                if "match_score" in device:
                    click.echo(f"[{index:2d}] {name} (score: {device['match_score']})")
                else:
                    click.echo(f"[{index:2d}] {name}")

                if description and description != name:
                    click.echo(f"     Description: {description}")
                click.echo(f"     Driver: {driver}")
                click.echo(f"     Sample Rate: {sample_rate} Hz, Channels: {channels}")
                if flags:
                    click.echo(f"     Flags: {', '.join(flags)}")

                click.echo()

    except Exception as e:
        click.echo(f"Error listing devices: {e}", err=True)
        raise click.Abort()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def diagnose(json_output: bool):
    """Diagnose audio device issues."""
    try:
        from mic_stream_util.backends import DeviceBackend

        backend = DeviceBackend.get_backend()
        devices = backend.get_all_devices()

        diagnostics = {"backend": backend.get_backend_name(), "backend_available": backend.backend_is_available(), "devices_found": len(devices), "recommendations": []}

        if not backend.backend_is_available():
            diagnostics["recommendations"].append(f"{backend.get_backend_name()} backend is not available")

        if not devices:
            diagnostics["recommendations"].append("No audio input devices detected")

        if json_output:
            click.echo(json.dumps(diagnostics, indent=2))
        else:
            click.echo("Audio Device Diagnostics")
            click.echo("=" * 50)

            click.echo("\nBackend Information:")
            click.echo("-" * 20)
            click.echo(f"Backend: {diagnostics['backend']}")
            click.echo(f"Available: {diagnostics['backend_available']}")
            click.echo(f"Devices Found: {diagnostics['devices_found']}")

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
