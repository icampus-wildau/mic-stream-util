from __future__ import annotations

from .core.device_manager import DeviceInfo, DeviceManager
from .core.microphone_manager import AudioConfig, MicrophoneStream

# Core functionality is always available
__all__ = ["AudioConfig", "DeviceInfo", "DeviceManager", "MicrophoneStream", "VAD_AVAILABLE"]

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
    __all__.extend([
        "CallbackEvent",
        "CallbackEventType", 
        "CallbackProcessor",
        "SpeechChunk",
        "SpeechManager",
        "VADConfig",
    ])
    VAD_AVAILABLE = True
except ImportError:
    # Speech functionality not available - users will get ImportError when trying to use it
    pass
