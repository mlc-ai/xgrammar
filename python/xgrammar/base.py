"""This module provides classes to handle C++ objects from nanobind."""

import os
from typing import Any

if os.environ.get("XGRAMMAR_BUILD_DOCS") != "1":
    from . import xgrammar_bindings as _core
else:
    _core: Any = "dummy namespace"


class XGRObject:
    """The base class for all objects in XGrammar. This class provides methods to handle the
    C++ handle from nanobind.

    In subclasses, the handle should be initialized via the the _create_from_handle, or via
    the _init_handle method called within the __init__ method, and should not be modified
    afterwards. Subclasses should use the _handle property to access the handle. When comparing
    two objects, the equality is checked by comparing the C++ handles.

    For performance considerations, objects in XGrammar should be lightweight and only maintain
    a handle to the C++ objects. Heavy operations should be performed on the C++ side.
    """

    @classmethod
    def _create_from_handle(cls, handle) -> "XGRObject":
        """Construct an object of the class from a C++ handle.

        Parameters
        ----------
        cls
            The class of the object.

        handle
            The C++ handle.

        Returns
        -------
        obj : XGRObject
            An object of type cls.
        """
        obj = cls.__new__(cls)
        obj.__handle = handle
        return obj

    def _init_handle(self, handle):
        """Initialize an object with a handle. This method should be called in the __init__
        method of the subclasses of XGRObject to initialize the C++ handle.

        Parameters
        ----------
        handle
            The C++ handle.
        """
        self.__handle = handle

    @property
    def _handle(self):
        """Get the C++ handle of the object.

        Returns
        -------
        handle
            The C++ handle.
        """
        return self.__handle

    def __eq__(self, other: object) -> bool:
        """Compare two XGrammar objects by comparing their C++ handles.

        Parameters
        ----------
        other : object
            The other object to compare with.

        Returns
        -------
        equal : bool
            Whether the two objects have the same C++ handle.
        """
        if not isinstance(other, XGRObject):
            return NotImplemented
        return self._handle == other._handle
