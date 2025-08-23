#!/usr/bin/env python3
"""Example usage of the audio device backends."""

from mic_stream_util.backends import PipewireBackend, SounddeviceBackend


def main():
    """Demonstrate backend usage."""
    print("Audio Device Backend Examples")
    print("=" * 50)

    # Test Sounddevice Backend
    print("\n1. Sounddevice Backend")
    print("-" * 30)

    sd_backend = SounddeviceBackend()
    if sd_backend.backend_is_available():
        print("✓ Sounddevice backend is available")

        devices = sd_backend.get_all_devices()
        print(f"Found {len(devices)} input devices:")

        # Show unique devices by index
        unique_devices = {}
        for device in devices:
            unique_devices[device["index"]] = device

        for device in unique_devices.values():
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Driver: {device['driver']}")
            print(f"      Sample Rate: {device['sample_specification']['sample_rate_hz']} Hz")
            print(f"      Channels: {device['sample_specification']['channels']}")
            print(f"      Format: {device['sample_specification']['sample_format']}")
            print()
    else:
        print("✗ Sounddevice backend not available")

    # Test Pipewire Backend
    print("\n2. Pipewire Backend")
    print("-" * 30)

    pw_backend = PipewireBackend()
    if pw_backend.backend_is_available():
        print("✓ Pipewire backend is available")

        devices = pw_backend.get_all_devices()
        print(f"Found {len(devices)} input devices:")

        # Show unique devices by index
        unique_devices = {}
        for device in devices:
            unique_devices[device["index"]] = device

        for device in unique_devices.values():
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Description: {device['description']}")
            print(f"      Driver: {device['driver']}")
            print(f"      Sample Rate: {device['sample_specification']['sample_rate_hz']} Hz")
            print(f"      Channels: {device['sample_specification']['channels']}")
            print(f"      Format: {device['sample_specification']['sample_format']}")
            print(f"      Mute: {device['mute']}")
            print(f"      Volume: {device['volume']}")
            print()
    else:
        print("✗ Pipewire backend not available")

    # Demonstrate device lookup
    print("\n3. Device Lookup Examples")
    print("-" * 30)

    backends = []
    if sd_backend.backend_is_available():
        backends.append(("Sounddevice", sd_backend))
    if pw_backend.backend_is_available():
        backends.append(("Pipewire", pw_backend))

    for name, backend in backends:
        print(f"\n{name} Backend:")
        devices = backend.get_all_devices()

        if devices:
            # Lookup by index
            first_device = devices[0]
            try:
                found = backend.get_device_info(first_device["index"])
                print(f"  Lookup by index {first_device['index']}: ✓ {found['name']}")
            except Exception as e:
                print(f"  Lookup by index {first_device['index']}: ✗ {e}")

            # Lookup by name
            try:
                found = backend.get_device_info(first_device["name"])
                print(f"  Lookup by name: ✓ {found['name']}")
            except Exception as e:
                print(f"  Lookup by name: ✗ {e}")

            # Fuzzy lookup
            try:
                # Try to find a device with partial name
                partial_name = first_device["name"].split()[0]  # First word
                found = backend.get_device_info(partial_name)
                print(f"  Fuzzy lookup '{partial_name}': ✓ {found['name']}")
            except Exception as e:
                print(f"  Fuzzy lookup '{partial_name}': ✗ {e}")

    # Demonstrate backend-specific features
    print("\n4. Backend-Specific Features")
    print("-" * 30)

    if pw_backend.backend_is_available():
        print("\nPipewire Backend Features:")
        devices = pw_backend.get_all_devices()

        if devices:
            # Test volume control (read-only for safety)
            first_device = devices[0]
            print(f"  Device {first_device['index']} volume: {first_device['volume']}")

            # Test getting detailed source info
            try:
                detailed_info = pw_backend.get_source_info(first_device["index"])
                print("  Detailed info available: ✓")
                print(f"    Owner Module: {detailed_info['owner_module']}")
                print(f"    Latency: {detailed_info['latency']} usec")
            except Exception as e:
                print(f"  Detailed info: ✗ {e}")

    if sd_backend.backend_is_available():
        print("\nSounddevice Backend Features:")
        devices = sd_backend.get_all_devices()

        if devices:
            first_device = devices[0]

            # Test supported sample rates
            try:
                supported_rates = sd_backend.get_supported_sample_rates(first_device["index"])
                print(f"  Supported sample rates for device {first_device['index']}: {supported_rates}")
            except Exception as e:
                print(f"  Sample rate check: ✗ {e}")

            # Test device openability
            try:
                openable = sd_backend.check_device_openable(first_device["index"], 16000, 1)
                print(f"  Device {first_device['index']} openable at 16kHz mono: {'✓' if openable else '✗'}")
            except Exception as e:
                print(f"  Device openability check: ✗ {e}")


if __name__ == "__main__":
    main()
