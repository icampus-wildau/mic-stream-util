"""Latency test command module."""

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
@click.option("--duration", type=float, default=10.0, help="Test duration in seconds (default: 10.0)")
@click.option("--tone-freq", type=float, default=440.0, help="Test tone frequency in Hz (default: 440.0)")
def latency_test(device: Optional[str], sample_rate: int, channels: int, blocksize: int, latency: str, duration: float, tone_freq: float):
    """Test audio system latency by measuring round-trip time."""
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

        console.print("[bold blue]Audio Latency Test[/bold blue]")
        console.print(f"Device: {device or 'default'}")
        console.print(f"Sample Rate: {sample_rate} Hz")
        console.print(f"Channels: {channels}")
        console.print(f"Block Size: {blocksize}")
        console.print(f"Test Duration: {duration:.1f}s")
        console.print(f"Tone Frequency: {tone_freq:.1f} Hz")
        console.print("Press Ctrl+C to stop")
        console.print("-" * 40)

        # Generate test tone
        t = np.linspace(0, blocksize / sample_rate, blocksize, False)
        test_tone = 0.3 * np.sin(2 * np.pi * tone_freq * t)

        mic = MicrophoneStream(config)

        def create_latency_display(avg_latency: float, min_latency: float, max_latency: float, test_count: int, current_latency: float) -> Panel:
            """Create the latency test display panel."""
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="bold")
            table.add_column("Bar", style="bold")

            # Current latency
            latency_bars = int(min(current_latency * 10, 50))  # Scale for display
            latency_bar = "█" * min(latency_bars, 50)
            latency_color = "red" if current_latency > 100 else "yellow" if current_latency > 50 else "green"
            table.add_row("Current", f"{current_latency:.1f}ms", f"[{latency_color}]{latency_bar:<50}[/{latency_color}]")

            # Statistics
            table.add_row("Average", f"{avg_latency:.1f}ms", "")
            table.add_row("Minimum", f"{min_latency:.1f}ms", "")
            table.add_row("Maximum", f"{max_latency:.1f}ms", "")
            table.add_row("Tests", str(test_count), "")

            return Panel(table, title="[bold]Latency Test[/bold]", border_style="blue")

        with mic.stream():
            latencies = []
            test_count = 0
            start_time = time.time()

            with Live(create_latency_display(0.0, 0.0, 0.0, 0, 0.0), refresh_per_second=10) as live:
                while time.time() - start_time < duration:
                    try:
                        # Record start time
                        send_time = time.time()

                        # Play test tone
                        sd.play(test_tone, sample_rate, blocking=True)

                        # Read audio and detect tone
                        chunk = mic.read(blocksize)

                        # Simple tone detection (cross-correlation with test tone)
                        correlation = np.correlate(chunk.flatten(), test_tone)
                        if np.max(correlation) > 0.1:  # Threshold for detection
                            receive_time = time.time()
                            latency_ms = (receive_time - send_time) * 1000
                            latencies.append(latency_ms)
                            test_count += 1

                            # Calculate statistics
                            avg_latency = np.mean(latencies) if latencies else 0.0
                            min_latency = np.min(latencies) if latencies else 0.0
                            max_latency = np.max(latencies) if latencies else 0.0

                            # Update display
                            live.update(create_latency_display(avg_latency, min_latency, max_latency, test_count, latency_ms))

                        time.sleep(0.1)  # Small delay between tests

                    except KeyboardInterrupt:
                        break

        # Final results
        if latencies:
            console.print("\n[bold green]Latency Test Results:[/bold green]")
            console.print(f"Tests performed: {test_count}")
            console.print(f"Average latency: {np.mean(latencies):.1f}ms")
            console.print(f"Minimum latency: {np.min(latencies):.1f}ms")
            console.print(f"Maximum latency: {np.max(latencies):.1f}ms")
            console.print(f"Standard deviation: {np.std(latencies):.1f}ms")
        else:
            console.print("\n[bold yellow]No latency measurements recorded[/bold yellow]")

        console.print("\n[bold green]Latency test completed[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during latency test: {e}[/bold red]")
        raise click.Abort()
