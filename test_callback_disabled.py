#!/usr/bin/env python3
"""
Test script to verify that read methods are disabled when callback mode is active.
"""

import numpy as np

from mic_stream_util.core.audio_config import AudioConfig
from mic_stream_util.core.microphone_manager import MicrophoneStream


def test_callback_disabled_read():
    """Test that read methods are disabled when callback is active."""

    config = AudioConfig(sample_rate=16000, channels=1, dtype="float32", num_samples=1024)

    mic_stream = MicrophoneStream(config)

    # Set a callback
    def dummy_callback(audio_data: np.ndarray) -> None:
        pass

    mic_stream.set_callback(dummy_callback)

    # Start the stream
    mic_stream.start_stream()

    try:
        # Try to read - this should raise an error
        print("Testing read() method with callback active...")
        mic_stream.read()
        print("ERROR: read() should have raised an exception!")
        return False
    except RuntimeError as e:
        print(f"✓ read() correctly raised: {e}")

    try:
        # Try to read_raw - this should raise an error
        print("Testing read_raw() method with callback active...")
        mic_stream.read_raw(1024)
        print("ERROR: read_raw() should have raised an exception!")
        return False
    except RuntimeError as e:
        print(f"✓ read_raw() correctly raised: {e}")

    # Stop the stream
    mic_stream.stop_stream()

    # Clear callback and test that read works again
    mic_stream.clear_callback()
    mic_stream.start_stream()

    try:
        # This should work now
        print("Testing read() method after clearing callback...")
        audio_data = mic_stream.read()
        print(f"✓ read() works after clearing callback: {audio_data.shape}")
    except Exception as e:
        print(f"ERROR: read() failed after clearing callback: {e}")
        return False

    mic_stream.stop_stream()
    return True


if __name__ == "__main__":
    print("=== Testing Callback Mode Restrictions ===\n")

    success = test_callback_disabled_read()

    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
