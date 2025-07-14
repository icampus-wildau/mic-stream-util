#!/usr/bin/env python3
"""CLI tool for microphone-util library."""

from __future__ import annotations

import json
import time
from typing import Optional

import click
from click_aliases import ClickAliasedGroup

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(cls=ClickAliasedGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option()
def main():
    """Microphone Stream Util CLI - Multiprocessing microphone helper with VAD support."""
    pass


@main.command(aliases=["ls", "l"], context_settings=CONTEXT_SETTINGS)
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


@main.command(aliases=["i", "info"], context_settings=CONTEXT_SETTINGS)
@click.argument("device_identifier")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def device_info(device_identifier: str, json_output: bool):
    """Get detailed information about a specific device."""
    # Dynamic imports for better response time
    from mic_stream_util.core.device_manager import DeviceManager
    from mic_stream_util.exceptions import DeviceNotFoundError

    try:
        device_info = DeviceManager.get_device_info(device_identifier)

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


@main.command(aliases=["rec", "r"], context_settings=CONTEXT_SETTINGS)
@click.option("--device", "-d", help="Device name or index (default: default device)")
@click.option("--sample-rate", "-r", type=int, default=16000, help="Sample rate (default: 16000)")
@click.option("--channels", "-c", type=int, default=1, help="Number of channels (default: 1)")
@click.option("--blocksize", "-b", type=int, default=1024, help="Block size (default: 1024)")
@click.option("--latency", type=click.Choice(["low", "high"]), default="low", help="Latency setting (default: low)")
@click.option("--output", "-o", type=click.Path(), help="Save audio to file (WAV format)")
def record(device: Optional[str], sample_rate: int, channels: int, blocksize: int, latency: str, output: Optional[str]):
    """Record audio from microphone until Ctrl+C."""
    # Dynamic imports for better response time
    import numpy as np

    from mic_stream_util.core.audio_config import AudioConfig
    from mic_stream_util.core.microphone_manager import MicrophoneStream

    try:
        config = AudioConfig(
            sample_rate=sample_rate,
            channels=channels,
            blocksize=blocksize,
            latency=latency,
            device_name=device,
            dtype="float32",
            buffer_size=sample_rate * 2,
        )
        click.echo("Recording... Press Ctrl+C to stop.")
        click.echo(f"Device: {device or 'default'}")
        click.echo(f"Sample Rate: {sample_rate} Hz")
        click.echo(f"Channels: {channels}")
        click.echo(f"Block Size: {blocksize}")
        click.echo("-" * 40)
        audio_chunks = []
        start_time = time.time()
        chunk_count = 0
        try:
            mic = MicrophoneStream(config)
            with mic.stream():
                while True:
                    chunk = mic.read(blocksize)
                    audio_chunks.append(chunk)
                    chunk_count += 1
                    rms = np.sqrt(np.mean(chunk**2))
                    level_bars = int(rms * 50)
                    bar = "█" * min(level_bars, 50)
                    click.echo(f"\rLevel: [{bar:<50}] {rms:.6f} | Chunks: {chunk_count}", nl=False)
        except KeyboardInterrupt:
            click.echo("\nRecording stopped by user.")
        click.echo()  # New line after progress
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            click.echo(f"Recorded {len(audio_chunks)} chunks, total samples: {len(full_audio)}")
            if output:
                import soundfile as sf

                sf.write(output, full_audio, sample_rate)
                click.echo(f"Audio saved to: {output}")
            rms = np.sqrt(np.mean(full_audio**2))
            peak = np.max(np.abs(full_audio))
            duration = len(full_audio) / sample_rate
            click.echo("Audio Statistics:")
            click.echo(f"  RMS Level: {rms:.6f}")
            click.echo(f"  Peak Level: {peak:.6f}")
            click.echo(f"  Duration: {duration:.2f} seconds")
        else:
            click.echo("No audio recorded")
    except Exception as e:
        click.echo(f"Error during recording: {e}", err=True)
        raise click.Abort()
