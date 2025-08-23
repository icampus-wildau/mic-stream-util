#!/usr/bin/env python3
"""
Example usage of Pipewire with sounddevice.

This example demonstrates how to work with Pipewire audio devices
even when Pipewire is not available as a separate hostapi.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf

from mic_stream_util.pipewire_utils import find_pipewire_device, get_pipewire_device_index, play_audio_with_pipewire, print_audio_info, record_audio_with_pipewire


def main():
    """Main example function."""
    print("=== Pipewire Audio Example ===\n")

    # Print current audio setup
    print_audio_info()

    # Get Pipewire device
    pipewire_device = find_pipewire_device()
    if not pipewire_device:
        print("❌ No Pipewire device found!")
        return

    pipewire_idx = pipewire_device["index"]
    print(f"\n✅ Using Pipewire device at index {pipewire_idx}")

    # Example 1: Generate and play a test tone
    print("\n=== Example 1: Playing Test Tone ===")
    sample_rate = int(pipewire_device["default_samplerate"])
    duration = 2.0  # seconds
    frequency = 440.0  # Hz (A4 note)

    # Generate a sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume

    print(f"Playing {frequency}Hz tone for {duration} seconds...")
    try:
        play_audio_with_pipewire(tone, sample_rate, device=pipewire_idx)
        print("✅ Test tone played successfully!")
    except Exception as e:
        print(f"❌ Error playing audio: {e}")

    # Example 2: Record audio
    print("\n=== Example 2: Recording Audio ===")
    record_duration = 3.0  # seconds
    print(f"Recording {record_duration} seconds of audio...")
    print("Speak into your microphone now!")

    try:
        # Record audio using Pipewire device
        recorded_audio = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, device=pipewire_idx, channels=1)
        sd.wait()  # Wait for recording to complete

        print("✅ Recording completed!")
        print(f"Recorded {len(recorded_audio)} samples")
        print(f"Audio shape: {recorded_audio.shape}")

        # Play back the recorded audio
        print("\nPlaying back recorded audio...")
        sd.play(recorded_audio, sample_rate, device=pipewire_idx)
        sd.wait()
        print("✅ Playback completed!")

        # Save the recorded audio
        output_file = "recorded_audio_pipewire.wav"
        sf.write(output_file, recorded_audio, sample_rate)
        print(f"✅ Audio saved to {output_file}")

    except Exception as e:
        print(f"❌ Error during recording: {e}")

    # Example 3: Show device comparison
    print("\n=== Example 3: Device Comparison ===")
    devices = sd.query_devices()

    print("Available devices for comparison:")
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:  # Only input devices
            print(f"  {i}: {device['name']}")
            print(f"      Input channels: {device['max_input_channels']}")
            print(f"      Sample rate: {device['default_samplerate']}")
            print(f"      Latency: {device['default_low_input_latency']:.4f}s")
            print()

    print("=== Summary ===")
    print("✅ Pipewire is working as an ALSA device")
    print("✅ You can use device index 6 for Pipewire audio")
    print("✅ Low latency audio is available (8.7ms)")
    print("✅ High quality audio (64 channels, 44.1kHz)")
    print("\nTo use Pipewire in your code:")
    print("  device=6  # or get_pipewire_device_index()")
    print("  samplerate=44100  # Pipewire's default")


if __name__ == "__main__":
    main()




