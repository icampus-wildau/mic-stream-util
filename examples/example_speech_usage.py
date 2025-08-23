#!/usr/bin/env python3
"""Example usage of the SpeechManager for voice activity detection."""

import time

from mic_stream_util.core.audio_config import AudioConfig
from mic_stream_util.speech import SpeechChunk, SpeechManager, VADConfig


def main():
    """Demonstrate speech manager usage."""

    # Configure audio settings
    audio_config = AudioConfig(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        num_samples=512,
        # device_name="default",  # Use default microphone
    )

    # Configure VAD settings
    vad_config = VADConfig(
        threshold=0.5,  # VAD threshold (0.0 to 1.0)
        padding_before_ms=300,  # Audio padding before speech
        padding_after_ms=300,  # Audio padding after speech
        max_silence_ms=1000,  # Max silence before ending speech
        min_speech_duration_ms=250,  # Minimum speech duration
        max_speech_duration_s=30.0,  # Maximum speech duration
    )

    # Create speech manager
    speech_manager = SpeechManager(audio_config, vad_config)

    # Define callback functions
    def on_vad_changed(vad_score: float) -> None:
        """Called when VAD score changes."""
        print(f"VAD Score: {vad_score:.3f}")

    def on_speech_start(start_time: float) -> None:
        """Called when speech starts."""
        print(f"Speech started at {start_time:.2f}s")

    def on_speech_chunk(chunk: SpeechChunk, vad_score: float) -> None:
        """Called for each speech chunk during speech."""
        print(f"Speech chunk: {chunk.duration:.3f}s, VAD: {vad_score:.3f}")

    def on_speech_ended(speech_chunk: SpeechChunk) -> None:
        """Called when speech ends."""
        print(f"Speech ended: duration={speech_chunk.duration:.2f}s, start={speech_chunk.start_time:.2f}s, end={speech_chunk.end_time:.2f}s")
        print(f"Audio shape: {speech_chunk.audio_chunk.shape}")

    def on_audio_chunk(audio_chunk, timestamp: float) -> None:
        """Called for all audio chunks (optional)."""
        # Uncomment to see all audio chunks
        # print(f"Audio chunk at {timestamp:.2f}s: shape={audio_chunk.shape}")
        pass

    # Set callbacks
    speech_manager.set_callbacks(
        on_vad_changed=on_vad_changed,
        on_speech_start=on_speech_start,
        on_speech_chunk=on_speech_chunk,
        on_audio_chunk=on_audio_chunk,
        on_speech_ended=on_speech_ended,
    )

    print("Starting speech detection...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print("-" * 50)

    try:
        # Start the stream
        speech_manager.start_stream()

        # Keep running
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping speech detection...")
    finally:
        speech_manager.stop_stream()
        print("Speech detection stopped.")


if __name__ == "__main__":
    main()
