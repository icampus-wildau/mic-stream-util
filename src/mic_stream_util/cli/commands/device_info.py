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
    from mic_stream_util.core.device_manager import DeviceManager
    from mic_stream_util.exceptions import DeviceNotFoundError

    try:
        try:
            id = int(device_identifier)
        except ValueError:
            id = device_identifier
        device_info = DeviceManager.get_device_info(id)

        if json_output:
            click.echo(json.dumps(device_info, indent=2))
        else:
            click.echo(f"\nDevice Information for '{device_identifier}':")
            click.echo("-" * 50)
            click.echo(f"Index: {device_info['index']}")
            click.echo(f"Name: {device_info['name']}")
            click.echo(f"Max Input Channels: {device_info['max_input_channels']}")
            click.echo(f"Default Sample Rate: {device_info.get('default_samplerate', 'Unknown')}")
            click.echo(f"Host API: {device_info.get('hostapi', 'Unknown')}")
            click.echo(f"Supported Sample Rates: {device_info.get('supported_samplerates', [])}")

    except DeviceNotFoundError as e:
        click.echo(f"Device not found: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error getting device info: {e}", err=True)
        raise click.Abort()
