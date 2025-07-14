"""Custom exceptions for the microphone utility library."""


class MicrophoneError(Exception):
    """Base exception for all microphone-related errors."""
    pass


class DeviceNotFoundError(MicrophoneError):
    """Raised when a requested audio device cannot be found."""
    pass


class StreamError(MicrophoneError):
    """Raised when there's an error with audio streaming."""
    pass


class VADError(MicrophoneError):
    """Raised when there's an error with Voice Activity Detection."""
    pass


class ConfigurationError(MicrophoneError):
    """Raised when there's an error with audio configuration."""
    pass 