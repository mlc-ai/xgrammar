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

    class InvalidStructuralTagError(RuntimeError):
        """Raised when the structural tag is invalid."""

else:
    # real implementation here
    DeserializeFormatError = _core.exception.DeserializeFormatError
    DeserializeVersionError = _core.exception.DeserializeVersionError
    InvalidJSONError = _core.exception.InvalidJSONError
    InvalidStructuralTagError = _core.exception.InvalidStructuralTagError
