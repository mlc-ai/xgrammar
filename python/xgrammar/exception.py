"""Exceptions in XGrammar."""

from typing import TYPE_CHECKING

from .base import _core

if TYPE_CHECKING or isinstance(_core, str):

    class DeserializeFormatError(RuntimeError):
        """Raised when the deserialization format is invalid."""

    class DeserializeVersionError(RuntimeError):
        """Raised when the serialization format is invalid."""

    class InvalidJSONError(RuntimeError):
        """Raised when the JSON is invalid."""

else:
    # real implementation here
    DeserializeFormatError = _core.DeserializeFormatError
    DeserializeVersionError = _core.DeserializeVersionError
    InvalidJSONError = _core.InvalidJSONError
