#!/usr/bin/env python3
"""
Simple test script for the microphone stream functionality.
"""

import time

from src.mic_stream_util.core.audio_config import AudioConfig
from src.mic_stream_util.core.microphone_manager import MicrophoneStream


def main():
    """Test the microphone stream functionality."""
    print("Testing Microphone Stream...")

    # Create a custom configuration
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        buffer_size=16000,  # 1 second buffer
        num_samples=512,
    )

    # Create microphone stream
    mic_stream = MicrophoneStream(config)

    try:
        print("Starting microphone stream...")

        # Use context manager for automatic cleanup
        with mic_stream.stream():
            print("Stream is active. Recording for 5 seconds...")

            # Record for 5 seconds
            start_time = time.time()
            samples_collected = 0

            while time.time() - start_time < 5.0:
                # Read 512 samples (about 32ms of audio)
                audio_data = mic_stream.read(512)
                samples_collected += len(audio_data)

                # Print some basic stats
                if samples_collected % 16000 == 0:  # Every second
                    print(f"Collected {samples_collected} samples...")
                    print(f"Audio shape: {audio_data.shape}")
                    print(f"Audio range: {audio_data.min():.3f} to {audio_data.max():.3f}")
                    print(f"Audio mean: {audio_data.mean():.3f}")
                    print()

        print("Recording completed!")
        print(f"Total samples collected: {samples_collected}")

    except Exception as e:
        print(f"Error during recording: {e}")

    print("Test completed.")


if __name__ == "__main__":
    main()
