#!/usr/bin/env python3
"""
Comprehensive example of using the microphone stream utility.

This example demonstrates:
1. Basic microphone streaming
2. Reading raw bytes vs numpy arrays
3. Different audio configurations
4. Error handling
5. Manual stream control
"""

import time

import numpy as np

from mic_stream_util.core.device_manager import DeviceManager
from src.mic_stream_util.core.audio_config import AudioConfig
from src.mic_stream_util.core.microphone_manager import MicrophoneStream


def example_basic_streaming():
    """Example 1: Basic microphone streaming with context manager."""
    print("=== Example 1: Basic Streaming ===")

    print(DeviceManager.print_devices())

    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        buffer_size=16000,  # 1 second buffer
        device_name="Microphone Array (Intel",
    )
    print(DeviceManager.find_device(config.device_name))

    mic_stream = MicrophoneStream(config)

    # Print all available devices

    try:
        with mic_stream.stream():
            print("Streaming for 3 seconds...")
            start_time = time.time()

            while time.time() - start_time < 3.0:
                # Read audio as numpy array
                # print("Reading audio...")
                audio_data = mic_stream.read(1024)
                print(f"Read {len(audio_data)} samples, shape: {audio_data.shape}")
                # time.sleep(0.1)
                # Print all data
                print(audio_data)
                break

    except Exception as e:
        print(f"Error: {e}")


def example_raw_bytes():
    """Example 2: Reading raw bytes from the stream."""
    print("\n=== Example 2: Raw Bytes Reading ===")

    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        dtype="float32",
        buffer_size=8000,  # 0.5 second buffer
    )

    mic_stream = MicrophoneStream(config)

    try:
        with mic_stream.stream():
            print("Reading raw bytes for 2 seconds...")
            start_time = time.time()

            while time.time() - start_time < 2.0:
                # Read raw bytes
                raw_data = mic_stream.read_raw(512)
                print(f"Read {len(raw_data)} bytes")

                # Convert to numpy array manually if needed
                audio_array = np.frombuffer(raw_data, dtype=config.dtype)
                print(f"Converted to {len(audio_array)} samples")
                # time.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")


def example_manual_control():
    """Example 3: Manual stream control without context manager."""
    print("\n=== Example 3: Manual Control ===")

    config = AudioConfig(
        sample_rate=22050,
        channels=1,
        dtype="int16",
        buffer_size=22050,  # 1 second buffer
    )

    mic_stream = MicrophoneStream(config)

    try:
        # Start stream manually
        mic_stream.start_stream()
        print("Stream started manually")

        # Check if streaming
        if mic_stream.is_streaming():
            print("Stream is active")

            # Read some audio
            for i in range(5):
                audio_data = mic_stream.read(256)
                print(f"Read chunk {i + 1}: {len(audio_data)} samples")
                time.sleep(0.2)

        # Stop stream manually
        mic_stream.stop_stream()
        print("Stream stopped manually")

    except Exception as e:
        print(f"Error: {e}")
        mic_stream.stop_stream()  # Ensure cleanup


def example_different_formats():
    """Example 4: Testing different audio formats."""
    print("\n=== Example 4: Different Formats ===")

    formats = [
        ("float32", np.float32),
        ("int16", np.int16),
        ("int32", np.int32),
    ]

    for dtype_str, numpy_dtype in formats:
        print(f"\nTesting {dtype_str} format...")

        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            dtype=dtype_str,
            buffer_size=16000,
        )

        mic_stream = MicrophoneStream(config)

        try:
            with mic_stream.stream():
                # Read a small amount of audio
                audio_data = mic_stream.read(256)
                print(f"  Shape: {audio_data.shape}")
                print(f"  Dtype: {audio_data.dtype}")
                print(f"  Range: {audio_data.min()} to {audio_data.max()}")
                print(f"  Mean: {audio_data.mean():.3f}")

        except Exception as e:
            print(f"  Error with {dtype_str}: {e}")


def example_error_handling():
    """Example 5: Error handling scenarios."""
    print("\n=== Example 5: Error Handling ===")

    # Test with invalid configuration
    try:
        config = AudioConfig(
            sample_rate=16000,
            channels=1,
            dtype="float32",
            device=99999,  # Invalid device
        )

        mic_stream = MicrophoneStream(config)
        with mic_stream.stream():
            pass

    except Exception as e:
        print(f"Expected error with invalid device: {e}")

    # Test reading from inactive stream
    try:
        mic_stream = MicrophoneStream()
        audio_data = mic_stream.read(256)  # Should fail

    except RuntimeError as e:
        print(f"Expected error reading from inactive stream: {e}")


def main():
    """Run all examples."""
    print("Microphone Stream Utility Examples")
    print("=" * 40)

    try:
        example_basic_streaming()
        example_raw_bytes()
        example_manual_control()
        example_different_formats()
        example_error_handling()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
