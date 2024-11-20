.. _installation:

Installation
============

.. contents:: Table of Contents
    :local:
    :depth: 2

XGrammar Python Package can be installed directly from a prebuilt developer package,
or built from source.


.. _installation_prebuilt_package:

Option 1. Prebuilt Package
--------------------------

We provide nightly built pip wheels for XGrammar via pip.

.. note::
    ❗ Whenever using Python, it is highly recommended to use **conda** to manage an isolated Python environment to avoid missing dependencies, incompatible versions, and package conflicts.
    Please make sure your conda environment has Python and pip installed.

.. code-block:: bash

    conda activate your-environment
    python -m pip install xgrammar


Then you can verify installation in command line:

.. code-block:: bash

    python -c "import xgrammar; print(xgrammar)"
    # Prints out: <module 'xgrammar' from '/path-to-env/lib/python3.11/site-packages/xgrammar/__init__.py'>


CUDA Dependency
~~~~~~~~~~~~~~~

When using NVIDIA GPUs, please also install these extra
dependencies to enable CUDA support for applying bitmasks:

.. code-block:: bash

    python -m pip install cuda-python nvidia-cuda-nvrtc-cu12

|

.. _installation_build_from_source:

Option 2. Build from Source
---------------------------

We also provide options to build XGrammar from source.
This step is useful when you want to make modification or obtain a specific version of XGrammar.


**Step 1. Set up build dependency.** To build from source, you need to ensure that the following build dependencies are satisfied:

* CMake >= 3.18
* Git

.. code-block:: bash

    # make sure to start with a fresh environment
    conda env remove -n xgrammar-venv
    # create the conda environment with build dependency
    conda create -n xgrammar-venv -c conda-forge \
        "cmake>=3.18" \
        git \
        python=3.11
    # enter the build environment
    conda activate xgrammar-venv
    # install Python dependency
    python3 -m pip install ninja pybind11 torch


**Step 2. Configure, build and install.** A standard git-based workflow is recommended to download XGrammar.

.. code-block:: bash

    # 1. clone from GitHub
    git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
    # 2. build XGrammar core and Python bindings
    mkdir build && cd build
    cmake .. -G Ninja
    ninja
    # 3. install the Python package
    cd ../python
    python3 -m pip install .
    # 4. (optional) add the python directory to PATH
    echo "export PATH=\"$(pwd):\$PATH\"" >> ~/.bashrc

**Step 3. Validate installation.** You may validate if XGrammar is compiled successfully in command line.
You should see the path you used to build from source with:

.. code:: bash

   python -c "import xgrammar; print(xgrammar)"
