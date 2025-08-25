"""Device info command module."""

from __future__ import annotations

import json

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("device_identifier")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def device_info(device_identifier: str, json_output: bool):
    """Get detailed information about a specific device."""
    # Dynamic imports for better response time
    from mic_stream_util.backends import DeviceBackend
    from mic_stream_util.exceptions import DeviceNotFoundError

    try:
        backend = DeviceBackend.get_backend()
        device_info = backend.get_device_info(device_identifier)

        if json_output:
            click.echo(json.dumps(device_info, indent=2))
        else:
            click.echo(f"\nDevice Information for '{device_identifier}' (Backend: {backend.__class__.__name__}):")
            click.echo("-" * 50)
            click.echo(f"Index: {device_info['index']}")
            click.echo(f"Name: {device_info['name']}")
            click.echo(f"Description: {device_info.get('description', 'N/A')}")
            click.echo(f"Driver: {device_info.get('driver', 'Unknown')}")

            sample_spec = device_info.get("sample_specification", {})
            click.echo(f"Sample Format: {sample_spec.get('sample_format', 'Unknown')}")
            click.echo(f"Sample Rate: {sample_spec.get('sample_rate_hz', 'Unknown')} Hz")
            click.echo(f"Channels: {sample_spec.get('channels', 'Unknown')}")

            click.echo(f"Channel Map: {device_info.get('channel_map', [])}")
            click.echo(f"Flags: {device_info.get('flags', [])}")

            if device_info.get("mute") is not None:
                click.echo(f"Mute: {device_info['mute']}")

            if device_info.get("volume"):
                click.echo(f"Volume: {device_info['volume']}")

    except (DeviceNotFoundError, ValueError) as e:
        if "not found" in str(e).lower():
            click.echo(f"Device not found: {e}", err=True)
        else:
            click.echo(f"Error getting device info: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error getting device info: {e}", err=True)
        raise click.Abort()
