"""Exceptions in XGrammar."""

from tvm_ffi import register_error


class DeserializeFormatError(RuntimeError):
    """Raised when the deserialization format is invalid."""


class DeserializeVersionError(RuntimeError):
    """Raised when the serialization format is invalid."""


class InvalidJSONError(RuntimeError):
    """Raised when the JSON is invalid."""


class InvalidStructuralTagError(RuntimeError):
    """Raised when the structural tag is invalid."""


register_error("DeserializeFormatError", DeserializeFormatError)
register_error("DeserializeVersionError", DeserializeVersionError)
register_error("InvalidJSONError", InvalidJSONError)
register_error("InvalidStructuralTagError", InvalidStructuralTagError)
