"""Monitor command module."""

from __future__ import annotations

from typing import Optional

import click
import numpy as np
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
def monitor(device: Optional[str], sample_rate: int, channels: int, blocksize: int, latency: str):
    """Monitor microphone input in real-time with audio level display."""
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

        console.print("[bold blue]Real-time Microphone Monitor[/bold blue]")
        console.print(f"Device: {device or 'default'}")
        console.print(f"Sample Rate: {sample_rate} Hz")
        console.print(f"Channels: {channels}")
        console.print(f"Block Size: {blocksize}")
        console.print("Press Ctrl+C to stop")
        console.print("-" * 40)

        mic = MicrophoneStream(config)

        def create_monitor_display(rms: float, peak: float, chunk_count: int) -> Panel:
            """Create the monitor display panel."""
            # Create visual level meters
            rms_bars = int(rms * 100)
            peak_bars = int(peak * 100)

            rms_bar = "█" * min(rms_bars, 50)
            peak_bar = "█" * min(peak_bars, 50)

            # Color coding based on levels
            if rms > 0.5:
                rms_color = "red"
                status = "🔴 HIGH"
            elif rms > 0.1:
                rms_color = "yellow"
                status = "🟡 MED"
            else:
                rms_color = "green"
                status = "🟢 LOW"

            if peak > 0.8:
                peak_color = "red"
            elif peak > 0.5:
                peak_color = "yellow"
            else:
                peak_color = "green"

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="bold")
            table.add_column("Meter", style="bold")

            table.add_row("RMS", f"{rms:.6f}", f"[{rms_color}]{rms_bar:<50}[/{rms_color}]")
            table.add_row("Peak", f"{peak:.6f}", f"[{peak_color}]{peak_bar:<50}[/{peak_color}]")
            table.add_row("Status", status, "")
            table.add_row("Chunks", str(chunk_count), "")

            return Panel(table, title="[bold]Audio Levels[/bold]", border_style="blue")

        with mic.stream():
            chunk_count = 0
            with Live(create_monitor_display(0.0, 0.0, 0), refresh_per_second=10) as live:
                while True:
                    try:
                        chunk = mic.read(blocksize)
                        chunk_count += 1

                        # Calculate audio statistics
                        rms = np.sqrt(np.mean(chunk**2))
                        peak = np.max(np.abs(chunk))

                        # Update display
                        live.update(create_monitor_display(rms, peak, chunk_count))

                    except KeyboardInterrupt:
                        break

        console.print("\n[bold green]Monitor stopped[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during monitoring: {e}[/bold red]")
        raise click.Abort()
