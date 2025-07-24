"""VAD command module."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--device", "-d", help="Device name or index (default: default device)")
@click.option("--sample-rate", "-r", type=int, default=16000, help="Sample rate (default: 16000)")
@click.option("--threshold", "-t", type=float, default=0.5, help="VAD threshold (0.0-1.0, default: 0.5)")
# @click.option("--frame-length", type=int, default=30, help="Frame length in ms (default: 30)")
@click.option("--padding-before", type=int, default=300, help="Padding before speech in ms (default: 300)")
@click.option("--padding-after", type=int, default=300, help="Padding after speech in ms (default: 300)")
@click.option("--max-silence", type=int, default=1000, help="Max silence duration in ms (default: 1000)")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False), help="Directory to save speech segments")
@click.option("--model-path", type=click.Path(exists=True), help="Path to Silero VAD model")
def vad(
    device: Optional[str],
    sample_rate: int,
    threshold: float,
    # frame_length: int,
    padding_before: int,
    padding_after: int,
    max_silence: int,
    output_dir: Optional[str],
    model_path: Optional[str],
):
    """Run Voice Activity Detection on microphone input."""
    # Check if VAD functionality is available
    try:
        from mic_stream_util import VAD_AVAILABLE
        if not VAD_AVAILABLE:
            raise ImportError("VAD functionality not available")
    except ImportError:
        click.echo("❌ VAD functionality requires additional dependencies.")
        click.echo("Install with: pip install mic-stream-util[vad] or uv add mic-stream-util[vad]")
        return

    # Dynamic imports for better response time
    try:
        from mic_stream_util.core.audio_config import AudioConfig
        from mic_stream_util.speech import SpeechManager, VADConfig
    except ImportError as e:
        click.echo(f"❌ Failed to import VAD dependencies: {e}")
        click.echo("Install with: pip install mic-stream-util[vad] or uv add mic-stream-util[vad]")
        return

    try:
        # Create configurations
        audio_config = AudioConfig(
            sample_rate=sample_rate,
            device_name=device,
            dtype="float32",
            num_samples=512,  # Convert ms to samples
            # num_samples=sample_rate * frame_length // 1000,  # Convert ms to samples
        )
        vad_config = VADConfig(
            threshold=threshold,
            padding_before_ms=padding_before,
            padding_after_ms=padding_after,
            max_silence_ms=max_silence,
        )

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        click.echo("Starting Voice Activity Detection...")
        click.echo(f"Device: {device or 'default'}")
        click.echo(f"Sample Rate: {sample_rate} Hz")
        click.echo(f"VAD Threshold: {threshold}")
        # click.echo(f"Frame Length: {frame_length} ms")
        click.echo("Press Ctrl+C to stop")
        click.echo("-" * 40)

        speech_count = 0

        def on_speech_start(timestamp: float):
            nonlocal speech_count
            speech_count += 1
            click.echo(f"\n[SPEECH START #{speech_count}] at {timestamp:.2f}s")

        def on_speech_ended(speech_chunk):
            click.echo(f"[SPEECH END #{speech_count}] Duration: {speech_chunk.duration:.2f}s, Samples: {len(speech_chunk.audio_chunk)}")

            # Save audio if output directory specified
            if output_dir and len(speech_chunk.audio_chunk) > 0:
                import soundfile as sf

                output_file = Path(output_dir) / f"speech_{speech_count:03d}.wav"
                sf.write(str(output_file), speech_chunk.audio_chunk, sample_rate)
                click.echo(f"  Saved to: {output_file}")

        # Start speech detection
        speech_manager = SpeechManager(audio_config=audio_config, vad_config=vad_config)
        speech_manager.set_callbacks(on_speech_start=on_speech_start, on_speech_ended=on_speech_ended)

        try:
            with speech_manager.stream_context():
                # Run speech detection
                while True:
                    time.sleep(0.1)  # Keep the main thread alive

        except KeyboardInterrupt:
            click.echo("\nVAD stopped by user")
        finally:
            speech_manager.stop_stream()

    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return
