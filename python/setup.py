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
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import List

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

CONDA_BUILD = os.getenv("CONDA_BUILD") is not None
CURRENT_DIR = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def get_version():
    with open("xgrammar/version.py") as f:
        code = compile(f.read(), "xgrammar/version.py", "exec")
    loc = {}
    exec(code, loc)
    if "__version__" not in loc:
        raise RuntimeError("Version info is not found in xgrammar/version.py")
    print("version:", loc["__version__"])
    return loc["__version__"]


def parse_requirements(filename: os.PathLike):
    with open(filename) as f:
        requirements = f.read().splitlines()

        def extract_url(line):
            return next(filter(lambda x: x[0] != "-", line.split()))

        extra_URLs = []
        deps = []
        for line in requirements:
            if line.startswith("#") or line.startswith("-r"):
                continue

            # handle -i and --extra-index-url options
            if "-i " in line or "--extra-index-url" in line:
                extra_URLs.append(extract_url(line))
            else:
                deps.append(line)
    return deps, extra_URLs


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        """Return True for binary distribution."""
        return True

    def is_pure(self):
        """Return False for binary distribution."""
        return False


def get_env_paths(env_var, splitter):
    """Get path in env variable"""
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(splitter)]
    return []


def get_dll_directories():
    """Get extra mlc llm dll directories"""
    source_dir = os.path.dirname(CURRENT_DIR)
    dll_path = [
        CURRENT_DIR,
        os.path.join(source_dir, "build"),
        os.path.join(source_dir, "build", "Release"),
    ]
    if "CONDA_PREFIX" in os.environ:
        dll_path.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(get_env_paths("LD_LIBRARY_PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(get_env_paths("DYLD_LIBRARY_PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(get_env_paths("PATH", ";"))
    return [os.path.abspath(p) for p in dll_path if os.path.isdir(p)]


def get_xgrammar_libs() -> List[str]:
    dll_paths = get_dll_directories()
    if platform.system() == "Windows":
        lib_glob = "xgrammar_bindings.*.pyd"
    else:
        lib_glob = "xgrammar_bindings.*.so"
    candidates = [os.path.join(p, lib_glob) for p in dll_paths]
    lib_paths = sum((glob.glob(p) for p in candidates), [])
    if len(lib_paths) == 0 or not os.path.isfile(lib_paths[0]):
        raise RuntimeError(
            "Cannot find xgrammar bindings library. Please build the library first. List of "
            f"candidates: {candidates}"
        )
    return lib_paths[:1]


def remove_path(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def main():
    setup_kwargs = {}
    lib_list = get_xgrammar_libs()

    if not CONDA_BUILD:
        with open("MANIFEST.in", "w", encoding="utf-8") as fo:
            for path in lib_list:
                if os.path.isfile(path):
                    shutil.copy(path, os.path.join(CURRENT_DIR, "xgrammar"))
                    _, libname = os.path.split(path)
                    fo.write(f"include xgrammar/{libname}\n")
        setup_kwargs = {"include_package_data": True}

    setup(
        name="xgrammar",
        version=get_version(),
        author="MLC Team",
        description="Cross-platform Near-zero Overhead Grammar-guided Generation for LLMs",
        long_description=open("../README.md").read(),
        licence="Apache 2.0",
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
        ],
        keywords="machine learning inference",
        zip_safe=False,
        install_requires=parse_requirements("requirements.txt")[0],
        python_requires=">=3.7, <4",
        url="https://github.com/mlc-ai/xgrammar",
        packages=find_packages(),
        distclass=BinaryDistribution,
        **setup_kwargs,
    )

    if not CONDA_BUILD:
        # Wheel cleanup
        os.remove("MANIFEST.in")
        for path in lib_list:
            _, libname = os.path.split(path)
            remove_path(os.path.join(CURRENT_DIR, "xgrammar", libname))


main()
