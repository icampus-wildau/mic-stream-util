from __future__ import annotations

from .core.microphone_manager import AudioConfig, MicrophoneStream

# Core functionality is always available
__all__ = ["AudioConfig", "MicrophoneStream", "VAD_AVAILABLE"]

# For backward compatibility, also export DeviceInfo from backends
from .backends import DeviceInfo

__all__.append("DeviceInfo")

# Try to import speech functionality
VAD_AVAILABLE = False
try:
    from .speech import (
        CallbackEvent,
        CallbackEventType,
        CallbackProcessor,
        SpeechChunk,
        SpeechManager,
        VADConfig,
    )

    __all__.extend(
        [
            "CallbackEvent",
            "CallbackEventType",
            "CallbackProcessor",
            "SpeechChunk",
            "SpeechManager",
            "VADConfig",
        ]
    )
    VAD_AVAILABLE = True
except ImportError:
    # Speech functionality not available - users will get ImportError when trying to use it
    pass
