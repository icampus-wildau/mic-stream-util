"""Route command module for Pipewire device management."""

from __future__ import annotations

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("device_identifier", required=False)
@click.option("--list", "list_devices", is_flag=True, help="List available devices")
def route(device_identifier: str | None, list_devices: bool):
    """Route a specific device to be the default input (Pipewire backend only)."""
    # Dynamic imports for better response time
    from mic_stream_util.backends import DeviceBackend

    try:
        backend = DeviceBackend.get_backend()

        if list_devices:
            click.echo("Available devices for routing:")
            devices = backend.get_all_devices()
            for device in devices:
                click.echo(f"  [{device['index']}] {device['name']}")
            return

        if not device_identifier:
            click.echo("Please provide a device identifier or use --list to see available devices")
            return

        # Check if we're using Pipewire backend
        if "Pipewire" not in backend.__class__.__name__:
            click.echo("❌ Device routing is only available with Pipewire backend", err=True)
            click.echo(f"Current backend: {backend.__class__.__name__}")
            return

        # Get device info
        try:
            device_info = backend.get_device_info(device_identifier)
        except Exception as e:
            click.echo(f"❌ Device not found: {e}", err=True)
            return

        click.echo(f"Routing device: {device_info['name']} (index: {device_info['index']})")

        # Route the device
        if backend.route_source_to_default(device_info["index"]):
            click.echo("✅ Device successfully routed to default input")
            click.echo("Note: You may need to restart your audio applications for changes to take effect")
        else:
            click.echo("❌ Failed to route device", err=True)

    except Exception as e:
        click.echo(f"Error routing device: {e}", err=True)
        raise click.Abort()
