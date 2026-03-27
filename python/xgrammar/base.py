"""This module provides classes to handle C++ objects via tvm_ffi."""

import os
from typing import Any, Union

if os.environ.get("XGRAMMAR_BUILD_DOCS") != "1":
    from tvm_ffi import Object as _ffi_Object

    from .tvm_ffi_binding import _ffi_api as _core
    from .tvm_ffi_binding import config as _config_ffi
    from .tvm_ffi_binding.kernels import _ffi_api as _kernels_ffi
    from .tvm_ffi_binding.testing import _ffi_api as _testing_ffi
    from .tvm_ffi_binding.testing.grammar_functor import _ffi_api as _grammar_functor_ffi

    _core.testing = _testing_ffi
    _core.testing.grammar_functor = _grammar_functor_ffi
    _core.kernels = _kernels_ffi
    _core.config = _config_ffi
else:
    _ffi_Object: Any = None  # type: ignore[misc, assignment]
    _core: Any = None


class XGRObject:
    """The base class for all objects in XGrammar. This class provides methods to handle the
    C++ object through a tvm_ffi Object (or its derived class) held by each instance.

    In subclasses, the FFI object should be initialized via _create_from_handle, or via
    _init_handle called within the __init__ method, and should not be modified afterwards.
    Subclasses should use the _handle property to access the underlying FFI object. When comparing
    two objects, equality is checked by comparing the underlying FFI objects.

    For performance considerations, objects in XGrammar should be lightweight and only maintain
    a handle to the C++ objects. Heavy operations should be performed on the C++ side.
    """

    @classmethod
    def _create_from_handle(cls, handle: Union["_ffi_Object", Any]) -> "XGRObject":
        """Construct an object of the class from an FFI object (tvm_ffi Object or derived).

        Parameters
        ----------
        cls
            The class of the object.

        handle
            The FFI object (e.g. from _core.Grammar, _core.CompiledGrammar, etc.).

        Returns
        -------
        obj : XGRObject
            An object of type cls.
        """
        obj = cls.__new__(cls)
        obj.__handle = handle
        return obj

    def _init_handle(self, handle: Union["_ffi_Object", Any]) -> None:
        """Initialize an object with an FFI handle. This method should be called in the __init__
        method of the subclasses of XGRObject to initialize the underlying FFI object.

        Parameters
        ----------
        handle
            The FFI object (e.g. from _core.GrammarCompiler, _core.GrammarMatcher, etc.).
        """
        self.__handle = handle

    @property
    def _handle(self) -> Union["_ffi_Object", Any]:
        """Get the underlying FFI object (tvm_ffi Object or derived).

        Returns
        -------
        handle
            The FFI object used for C++ communication.
        """
        return self.__handle

    def __eq__(self, other: object) -> bool:
        """Compare two XGrammar objects by comparing their underlying FFI objects.

        Parameters
        ----------
        other : object
            The other object to compare with.

        Returns
        -------
        equal : bool
            Whether the two objects have the same underlying FFI object.
        """
        if not isinstance(other, XGRObject):
            return NotImplemented
        return self._handle == other._handle
