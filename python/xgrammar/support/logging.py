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
"""
Logging support for XGrammar. It derives from Python's logging module, and in the future,
it can be easily replaced by other logging modules such as structlog.
"""

import logging


def enable_logging():
    """Enable XGrammar's default logging formpat"""
    logging.basicConfig(
        level=logging.INFO,
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[{asctime}] {levelname} {filename}:{lineno}: {message}",
    )


def getLogger(name: str):  # pylint: disable=invalid-name
    """Get a logger according to the given name"""
    return logging.getLogger(name)
