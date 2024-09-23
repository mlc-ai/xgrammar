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

from setuptools import find_packages, setup
from setuptools.dist import Distribution

CONDA_BUILD = os.getenv("CONDA_BUILD") is not None
PYTHON_SRC_DIR = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
PROJECT_DIR = os.path.dirname(PYTHON_SRC_DIR)


def get_version() -> str:
    version_path = os.path.join(PYTHON_SRC_DIR, "xgrammar", "version.py")
    if not os.path.exists(version_path) or not os.path.isfile(version_path):
        msg = f"Version file not found: {version_path}"
        raise RuntimeError(msg)
    with open(version_path) as f:
        code = compile(f.read(), version_path, "exec")
    loc = {}
    exec(code, loc)
    if "__version__" not in loc:
        msg = "Version info is not found in xgrammar/version.py"
        raise RuntimeError(msg)
    return loc["__version__"]


def parse_requirements(filename: os.PathLike) -> list[str]:
    with open(filename) as f:
        requirements = f.read().splitlines()

        def extract_url(line):
            return next(filter(lambda x: x[0] != "-", line.split()))

        extra_URLs = []
        deps = []
        for line in requirements:
            if line.startswith(("#", "-r")):
                continue

            # handle -i and --extra-index-url options
            if "-i " in line or "--extra-index-url" in line:
                extra_URLs.append(extract_url(line))
            else:
                deps.append(line)
    return deps, extra_URLs


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self) -> bool:
        """Return True for binary distribution."""
        return True

    def is_pure(self) -> bool:
        """Return False for binary distribution."""
        return False


def get_xgrammar_lib() -> str:
    if platform.system() == "Windows":
        lib_glob = "xgrammar_bindings.*.pyd"
    else:
        lib_glob = "xgrammar_bindings.*.so"
    lib_glob = os.path.join(PYTHON_SRC_DIR, "xgrammar", lib_glob)

    lib_paths = glob.glob(lib_glob)
    if len(lib_paths) == 0 or not os.path.isfile(lib_paths[0]):
        msg = (
            "Cannot find xgrammar bindings library. Please build the library first. Search path: "
            f"{lib_glob}"
        )
        raise RuntimeError(msg)
    if len(lib_paths) > 1:
        msg = (
            f"Found multiple xgrammar bindings libraries: {lib_paths}. "
            "Please remove the extra ones."
        )
        raise RuntimeError(msg)

    return lib_paths[0]


def main() -> None:
    xgrammar_lib_path = get_xgrammar_lib()

    setup(
        name="xgrammar",
        version=get_version(),
        author="MLC Team",
        description="Cross-platform Near-zero Overhead Grammar-guided Generation for LLMs",
        long_description=open(os.path.join(PROJECT_DIR, "README.md")).read(),
        licence="Apache 2.0",
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
        ],
        keywords="machine learning inference",
        packages=find_packages(),
        package_data={"xgrammar": [xgrammar_lib_path]},
        zip_safe=False,
        install_requires=parse_requirements("requirements.txt")[0],
        python_requires=">=3.7, <4",
        url="https://github.com/mlc-ai/xgrammar",
        distclass=BinaryDistribution,
    )


main()
