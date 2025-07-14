"""Speech processing module for voice activity detection."""

from .speech_manager import (
    CallbackEvent,
    CallbackEventType,
    CallbackProcessor,
    SpeechChunk,
    SpeechManager,
    VADConfig,
)

__all__ = [
    "CallbackEvent",
    "CallbackEventType",
    "CallbackProcessor",
    "SpeechChunk",
    "SpeechManager",
    "VADConfig",
]
