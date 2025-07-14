"""Record command module."""

from __future__ import annotations

import time
from typing import Optional

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
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
