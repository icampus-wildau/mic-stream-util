#!/usr/bin/env python3
"""
Example demonstrating the callback functionality of MicrophoneStream.

This example shows how to use the callback mode where audio data is automatically
processed in a separate thread instead of manually calling read().
"""

import time

import numpy as np

from mic_stream_util.core.audio_config import AudioConfig
from mic_stream_util.core.microphone_manager import MicrophoneStream


def audio_callback(audio_data: np.ndarray) -> None:
    """
    Callback function that gets called with audio data.

    Parameters
    ----------
    audio_data : np.ndarray
        Audio data with shape (num_samples, channels)
    """
    # Calculate RMS (Root Mean Square) as a simple audio level indicator
    rms = np.sqrt(np.mean(audio_data**2))

    # Print audio level (you could do any processing here)
    print(f"Audio level: {rms:.4f} | Shape: {audio_data.shape} | Max: {np.max(audio_data):.4f}")


def main():
    """Main function demonstrating callback usage."""

    # Create audio configuration
    config = AudioConfig(sample_rate=16000, channels=1, dtype="float32", num_samples=1024)

    # Create microphone stream
    mic_stream = MicrophoneStream(config)

    # Set the callback function
    mic_stream.set_callback(audio_callback)

    print("Starting microphone stream with callback...")
    print("Press Ctrl+C to stop")

    try:
        # Start the stream - the callback will be called automatically
        with mic_stream.stream():
            # Keep the main thread alive
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping stream...")

    print("Stream stopped.")


def example_without_callback():
    """Example showing how to use the stream without callback (traditional way)."""

    config = AudioConfig(sample_rate=16000, channels=1, dtype="float32", num_samples=1024)

    mic_stream = MicrophoneStream(config)

    print("Starting microphone stream without callback...")
    print("Press Ctrl+C to stop")

    try:
        with mic_stream.stream():
            while True:
                # Manually read audio data
                audio_data = mic_stream.read()
                rms = np.sqrt(np.mean(audio_data**2))
                print(f"Audio level: {rms:.4f} | Shape: {audio_data.shape}")

    except KeyboardInterrupt:
        print("\nStopping stream...")


if __name__ == "__main__":
    print("=== MicrophoneStream Callback Example ===\n")

    # Run the callback example
    main()

    # Uncomment the line below to see the traditional usage
    # example_without_callback()
