"""Speech processing module for voice activity detection."""

try:
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
except ImportError as e:
    if "torch" in str(e) or "silero_vad" in str(e):
        raise ImportError(
            "Speech/VAD functionality requires additional dependencies. "
            "Install with: pip install mic-stream-util[vad] or uv add mic-stream-util[vad]"
        ) from e
    raise
