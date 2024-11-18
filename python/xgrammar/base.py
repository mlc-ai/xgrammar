# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The main functionality of XGrammar. The functions here are Python bindings of the C++ logic."""

from . import xgrammar_bindings as _core


class XGObject:
    """The base class for all objects in XGrammar. This class provides methods to handle the
    interaction between Python and C++ objects.
    """

    @classmethod
    def from_handle(cls, handle) -> "XGObject":
        """Construct an object of the class from a C++ handle.

        Parameters
        ----------
        cls
            The class of the object.

        handle
            The C++ handle.

        Returns
        -------
        obj : XGObject
            An object of type cls.
        """
        obj = cls.__new__(cls)
        obj._handle = handle
        return obj

    def init_with_handle(self, handle):
        """Initialize an object with a handle. Used in the __init__ method of the class.

        Parameters
        ----------
        handle
            The C++ handle.
        """
        self._handle = handle

    @property
    def handle(self):
        """Get the C++ handle of the object.

        Returns
        -------
        handle
            The C++ handle.
        """
        return self._handle

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
        if not isinstance(other, XGObject):
            return NotImplemented
        return self._handle == other._handle
