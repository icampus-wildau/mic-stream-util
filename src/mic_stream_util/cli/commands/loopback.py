"""Loopback command module."""

from __future__ import annotations

import time
from typing import Optional

import click
import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--device", "-d", help="Device name or index (default: default device)")
@click.option("--sample-rate", "-r", type=int, default=16000, help="Sample rate (default: 16000)")
@click.option("--channels", "-c", type=int, default=1, help="Number of channels (default: 1)")
@click.option("--blocksize", "-b", type=int, default=1024, help="Block size (default: 1024)")
@click.option("--latency", type=click.Choice(["low", "high"]), default="low", help="Latency setting (default: low)")
@click.option("--delay", type=float, default=0.1, help="Delay in seconds before playback (default: 0.1)")
@click.option("--gain", type=float, default=1.0, help="Playback gain (default: 1.0)")
def loopback(device: Optional[str], sample_rate: int, channels: int, blocksize: int, latency: str, delay: float, gain: float):
    """Record audio and immediately play it back with configurable delay."""
    # Dynamic imports for better response time
    from mic_stream_util.core.audio_config import AudioConfig
    from mic_stream_util.core.microphone_manager import MicrophoneStream

    console = Console()

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

        console.print("[bold blue]Audio Loopback Test[/bold blue]")
        console.print(f"Device: {device or 'default'}")
        console.print(f"Sample Rate: {sample_rate} Hz")
        console.print(f"Channels: {channels}")
        console.print(f"Block Size: {blocksize}")
        console.print(f"Delay: {delay:.3f}s")
        console.print(f"Gain: {gain:.2f}")
        console.print("Press Ctrl+C to stop")
        console.print("-" * 40)

        mic = MicrophoneStream(config)

        def create_loopback_display(input_rms: float, output_rms: float, chunk_count: int, latency_ms: float) -> Panel:
            """Create the loopback display panel."""
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="bold")
            table.add_column("Bar", style="bold")

            # Input level
            input_bars = int(input_rms * 50)
            input_bar = "█" * min(input_bars, 50)
            input_color = "red" if input_rms > 0.5 else "yellow" if input_rms > 0.1 else "green"
            table.add_row("Input", f"{input_rms:.6f}", f"[{input_color}]{input_bar:<50}[/{input_color}]")

            # Output level
            output_bars = int(output_rms * 50)
            output_bar = "█" * min(output_bars, 50)
            output_color = "red" if output_rms > 0.5 else "yellow" if output_rms > 0.1 else "green"
            table.add_row("Output", f"{output_rms:.6f}", f"[{output_color}]{output_bar:<50}[/{output_color}]")

            # Statistics
            table.add_row("Chunks", str(chunk_count), "")
            table.add_row("Latency", f"{latency_ms:.1f}ms", "")

            return Panel(table, title="[bold]Loopback Monitor[/bold]", border_style="blue")

        with mic.stream():
            chunk_count = 0
            last_playback_time = 0

            with Live(create_loopback_display(0.0, 0.0, 0, 0.0), refresh_per_second=10) as live:
                while True:
                    try:
                        chunk = mic.read(blocksize)
                        chunk_count += 1

                        # Calculate input RMS
                        input_rms = np.sqrt(np.mean(chunk**2))

                        # Apply gain and prepare for playback
                        playback_chunk = chunk * gain

                        # Calculate current time
                        current_time = time.time()

                        # Play back with delay
                        if current_time - last_playback_time >= delay:
                            sd.play(playback_chunk, sample_rate, blocking=False)
                            last_playback_time = current_time

                        # Calculate output RMS (approximate)
                        output_rms = input_rms * gain

                        # Calculate latency
                        latency_ms = delay * 1000

                        # Update display
                        live.update(create_loopback_display(input_rms, output_rms, chunk_count, latency_ms))

                    except KeyboardInterrupt:
                        break

        console.print("\n[bold green]Loopback test stopped[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during loopback test: {e}[/bold red]")
        raise click.Abort()
