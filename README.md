# Microphone Stream Utility

A Python utility for managing microphone streams with support for both manual reading and callback-based processing.

## Features

- **Multi-process audio capture**: Audio is captured in a separate process to avoid blocking the main thread
- **Shared memory buffer**: Efficient data transfer between processes using shared memory
- **Flexible audio configuration**: Configurable sample rate, channels, data type, and buffer settings
- **Callback support**: Process audio data automatically in a separate thread
- **Manual reading**: Traditional read-based approach for custom processing
- **Device management**: Automatic device detection and selection
- **Context manager support**: Easy stream lifecycle management

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mic-stream-util

# Install dependencies
uv sync
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

## API Reference

### MicrophoneStream

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

## Examples

See `example_callback_usage.py` for a complete example demonstrating both callback and manual reading modes.

## Important Notes

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
```
