# Microphone Stream Utility

A Python utility for managing microphone streams with support for both manual reading and callback-based processing, plus optional Voice Activity Detection (VAD).

## Features

- **Multi-process audio capture**: Audio is captured in a separate process to avoid blocking the main thread
- **Shared memory buffer**: Efficient data transfer between processes using shared memory
- **Flexible audio configuration**: Configurable sample rate, channels, data type, and buffer settings
- **Callback support**: Process audio data automatically in a separate thread
- **Manual reading**: Traditional read-based approach for custom processing
- **Device management**: Automatic device detection and selection
- **Context manager support**: Easy stream lifecycle management
- **Voice Activity Detection (VAD)**: Optional speech detection using Silero VAD (requires additional dependencies)

## Installation

### Basic Installation (Core Features Only)

```bash
# Clone the repository
git clone <repository-url>
cd mic-stream-util

# Install core dependencies only
uv sync
```

### With Voice Activity Detection (VAD)

```bash
# Install with VAD support (includes torch and silero-vad)
uv add mic-stream-util[vad]

# Or if installing from source
uv sync --extra vad
```

### All Features

```bash
# Install with all optional features
uv add mic-stream-util[all]
```

## Quick Start

### Basic Usage (Manual Reading)

```python
from mic_stream_util.core.microphone_manager import MicrophoneStream
from mic_stream_util.core.audio_config import AudioConfig
import numpy as np

# Create configuration
config = AudioConfig(
    sample_rate=16000,
    channels=1,
    dtype="float32",
    num_samples=1024
)

# Create and use microphone stream
mic_stream = MicrophoneStream(config)

with mic_stream.stream():
    while True:
        # Read audio data manually
        audio_data = mic_stream.read()
        print(f"Audio shape: {audio_data.shape}")
        # Process audio_data as needed
```

### Callback Mode

```python
from mic_stream_util.core.microphone_manager import MicrophoneStream
from mic_stream_util.core.audio_config import AudioConfig
import numpy as np

def audio_callback(audio_data: np.ndarray) -> None:
    """Process audio data automatically."""
    rms = np.sqrt(np.mean(audio_data**2))
    print(f"Audio level: {rms:.4f}")

# Create configuration
config = AudioConfig(
    sample_rate=16000,
    channels=1,
    dtype="float32",
    num_samples=1024
)

# Create microphone stream
mic_stream = MicrophoneStream(config)

# Set callback function
mic_stream.set_callback(audio_callback)

# Start streaming - callback will be called automatically
with mic_stream.stream():
    # Keep main thread alive
    import time
    while True:
        time.sleep(0.1)
```

### Voice Activity Detection (VAD)

```python
from mic_stream_util import SpeechManager, VADConfig, AudioConfig

# Check if VAD is available
from mic_stream_util import VAD_AVAILABLE
if not VAD_AVAILABLE:
    print("VAD requires additional dependencies. Install with: pip install mic-stream-util[vad]")
    exit(1)

# Create configurations
audio_config = AudioConfig(sample_rate=16000, dtype="float32", num_samples=512)
vad_config = VADConfig(threshold=0.5, padding_before_ms=300, padding_after_ms=300)

# Create speech manager
speech_manager = SpeechManager(audio_config=audio_config, vad_config=vad_config)

def on_speech_start(timestamp: float):
    print(f"Speech started at {timestamp:.2f}s")

def on_speech_ended(speech_chunk):
    print(f"Speech ended, duration: {speech_chunk.duration:.2f}s")

# Set callbacks
speech_manager.set_callbacks(
    on_speech_start=on_speech_start,
    on_speech_ended=on_speech_ended
)

# Start VAD
with speech_manager.stream_context():
    import time
    while True:
        time.sleep(0.1)
```

## Command Line Interface

The package includes a CLI with various commands:

```bash
# List audio devices
mic devices

# Monitor audio levels
mic monitor

# Record audio
mic record --output recording.wav

# Voice Activity Detection (requires VAD dependencies)
mic vad --threshold 0.5

# Test latency
mic latency-test

# CPU usage monitoring
mic cpu-usage
```

## API Reference

### Core Classes

#### MicrophoneStream

Main class for managing microphone streams.

#### Constructor

```python
MicrophoneStream(config: AudioConfig | None = None)
```

- `config`: Audio configuration. If None, uses default configuration.

#### Methods

##### `set_callback(callback: Callable[[np.ndarray], None] | None)`

Set a callback function to be called when audio data is available.

- `callback`: Function that accepts a numpy array with shape (num_samples, channels)
- If `None`, callback mode is disabled

##### `clear_callback()`

Clear the callback function and disable callback mode.

##### `has_callback() -> bool`

Check if a callback function is set.

##### `start_stream()`

Start the microphone stream in a separate process.

##### `stop_stream()`

Stop the microphone stream and clean up resources.

##### `stream()`

Context manager for automatic stream start/stop.

##### `is_streaming() -> bool`

Check if the stream is currently active.

##### `read_raw(num_samples: int) -> bytes`

Read raw audio data from the stream buffer.

**Note**: This method is disabled when callback mode is active.

##### `read(num_samples: int | None = None) -> np.ndarray`

Read audio data from the stream buffer.

**Note**: This method is disabled when callback mode is active.

### AudioConfig

Configuration class for audio settings.

#### Constructor

```python
AudioConfig(
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "float32",
    blocksize: int = None,
    buffer_size: int | None = None,
    device: int | None = None,
    device_name: str | None = None,
    latency: str = "low",
    num_samples: int = 512
)
```

#### Parameters

- `sample_rate`: Sample rate in Hz
- `channels`: Number of audio channels
- `dtype`: Data type ("float32", "int32", "int16", "int8", "uint8")
- `blocksize`: Audio block size (defaults to sample_rate // 10)
- `buffer_size`: Buffer size in samples (defaults to sample_rate * 10)
- `device`: Device index
- `device_name`: Device name (will be used to find device index)
- `latency`: Latency setting ("low" or "high")
- `num_samples`: Number of samples to process at a time

### Speech Classes (VAD Dependencies Required)

#### SpeechManager

Main class for Voice Activity Detection.

#### Constructor

```python
SpeechManager(audio_config: AudioConfig, vad_config: VADConfig)
```

#### VADConfig

Configuration for Voice Activity Detection.

```python
VADConfig(
    threshold: float = 0.5,
    padding_before_ms: int = 300,
    padding_after_ms: int = 300,
    max_silence_ms: int = 1000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf")
)
```

## Examples

See the example files for complete demonstrations:
- `example_usage.py` - Basic microphone usage
- `example_callback_usage.py` - Callback-based processing
- `example_speech_usage.py` - Voice Activity Detection

## Important Notes

### Optional Dependencies

- **Core functionality**: Works without any additional dependencies
- **VAD functionality**: Requires `torch` and `silero-vad` (install with `[vad]` extra)
- **Check availability**: Use `from mic_stream_util import VAD_AVAILABLE` to check if VAD is available

### Callback Mode vs Manual Reading

- **Callback Mode**: Audio data is automatically processed in a separate thread. The `read()` and `read_raw()` methods are disabled.
- **Manual Reading**: You must manually call `read()` or `read_raw()` to get audio data.

### Thread Safety

- Callback functions are called in a separate thread, so ensure thread-safe operations
- The callback function should handle exceptions gracefully as they won't stop the stream

### Resource Management

- Always use the context manager (`with mic_stream.stream():`) or call `stop_stream()` to clean up resources
- The stream uses shared memory, so proper cleanup is important

## Development

```bash
# Run tests
uv run pytest

# Run example
uv run example_callback_usage.py

# Install development dependencies
uv sync --extra vad
```
