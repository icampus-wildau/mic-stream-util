"""Spectrum command module."""

from __future__ import annotations

from typing import Optional

import click
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# TODO: The display could be improved a lot, but it's working for now.


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--device", "-d", help="Device name or index (default: default device)")
@click.option("--sample-rate", "-r", type=int, default=16000, help="Sample rate (default: 16000)")
@click.option("--channels", "-c", type=int, default=1, help="Number of channels (default: 1)")
@click.option("--blocksize", "-b", type=int, default=1024, help="Block size (default: 1024)")
@click.option("--latency", type=click.Choice(["low", "high"]), default="low", help="Latency setting (default: low)")
@click.option("--fft-size", type=int, default=512, help="FFT size for spectrum analysis (default: 512)")
@click.option("--freq-range", type=str, default="0-8000", help="Frequency range to display in Hz (default: 0-8000)")
def spectrum(device: Optional[str], sample_rate: int, channels: int, blocksize: int, latency: str, fft_size: int, freq_range: str):
    """Display real-time frequency spectrum of microphone input."""
    # Dynamic imports for better response time
    from mic_stream_util.core.audio_config import AudioConfig
    from mic_stream_util.core.microphone_manager import MicrophoneStream

    console = Console()

    try:
        # Parse frequency range
        try:
            freq_min, freq_max = map(int, freq_range.split("-"))
        except ValueError:
            console.print("[bold red]Invalid frequency range format. Use 'min-max' (e.g., '0-8000')[/bold red]")
            raise click.Abort()

        config = AudioConfig(
            sample_rate=sample_rate,
            channels=channels,
            blocksize=blocksize,
            latency=latency,
            device_name=device,
            dtype="float32",
            buffer_size=sample_rate * 2,
        )

        console.print("[bold blue]Real-time Frequency Spectrum[/bold blue]")
        console.print(f"Device: {device or 'default'}")
        console.print(f"Sample Rate: {sample_rate} Hz")
        console.print(f"FFT Size: {fft_size}")
        console.print(f"Frequency Range: {freq_min}-{freq_max} Hz")
        console.print("Press Ctrl+C to stop")
        console.print("-" * 40)

        mic = MicrophoneStream(config)

        def create_spectrum_display(magnitudes: np.ndarray, frequencies: np.ndarray, rms: float) -> Panel:
            """Create the spectrum display panel."""
            # Filter frequencies to display range
            mask = (frequencies >= freq_min) & (frequencies <= freq_max)
            display_freqs = frequencies[mask]
            display_mags = magnitudes[mask]

            # Create visual spectrum bars
            max_bars = 50
            if len(display_mags) > 0:
                # Normalize magnitudes to 0-1 range
                mag_norm = display_mags / (np.max(display_mags) + 1e-10)

                # Create spectrum visualization
                spectrum_bars = []
                for i, mag in enumerate(mag_norm):
                    bars = int(mag * max_bars)
                    if bars > 0:
                        bar_str = "█" * bars
                        # Color based on frequency (low=blue, mid=green, high=red)
                        if i < len(display_mags) // 3:
                            color = "blue"
                        elif i < 2 * len(display_mags) // 3:
                            color = "green"
                        else:
                            color = "red"
                        spectrum_bars.append(f"[{color}]{bar_str}[/{color}]")
                    else:
                        spectrum_bars.append(" ")

                # Create frequency labels
                freq_labels = []
                for freq in display_freqs:
                    if freq < 1000:
                        freq_labels.append(f"{freq}Hz")
                    else:
                        freq_labels.append(f"{freq / 1000:.1f}k")

                # Create table
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Freq", style="bold")
                table.add_column("Spectrum", style="bold")
                table.add_column("Level", style="bold")

                # Add spectrum data
                for i, (freq_label, bar) in enumerate(zip(freq_labels, spectrum_bars)):
                    level = f"{display_mags[i]:.2f}" if display_mags[i] > 0.01 else "0.00"
                    table.add_row(freq_label, bar, level)

                # Add RMS info
                table.add_row("RMS", f"{rms:.4f}", "")

            else:
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Info", style="bold")
                table.add_row("No data in frequency range")

            return Panel(table, title="[bold]Frequency Spectrum[/bold]", border_style="blue")

        with mic.stream():
            with Live(create_spectrum_display(np.array([]), np.array([]), 0.0), refresh_per_second=10) as live:
                while True:
                    try:
                        chunk = mic.read(blocksize)

                        # Convert to mono if stereo (take mean across channels)
                        if chunk.ndim > 1:
                            chunk = np.mean(chunk, axis=1)

                        # Calculate RMS
                        rms = np.sqrt(np.mean(chunk**2))

                        # Apply window function
                        window = np.hanning(len(chunk))
                        windowed_chunk = chunk * window

                        # Perform FFT
                        fft_result = np.fft.fft(windowed_chunk, n=fft_size)
                        magnitudes = np.abs(fft_result[: fft_size // 2])
                        frequencies = np.fft.fftfreq(fft_size, 1 / sample_rate)[: fft_size // 2]

                        # Update display
                        live.update(create_spectrum_display(magnitudes, frequencies, rms))

                    except KeyboardInterrupt:
                        break

        console.print("\n[bold green]Spectrum analysis stopped[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during spectrum analysis: {e}[/bold red]")
        raise e
        # raise click.Abort()
