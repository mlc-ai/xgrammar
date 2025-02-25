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


**Step 1. Set up build environment.** To build from source, you need to ensure that the following build dependencies are satisfied:

* CMake >= 3.18
* Git
* C++ Compiler (e.g. apt-get install build-essential)

.. code-block:: bash

    # Using conda
    # make sure to start with a fresh environment
    conda env remove -n xgrammar-venv
    # create the conda environment with build dependency
    conda create -n xgrammar-venv -c conda-forge \
        "cmake>=3.18" \
        git \
        python=3.11 \
        ninja
    # enter the build environment
    conda activate xgrammar-venv

    # Using pip (you will need to install git seperately)
    python -m venv .venv
    source .venv/bin/activate


**Step 2. Configure, build and install.** A standard git-based workflow is recommended to download XGrammar.

.. code-block:: bash

    # 1. clone from GitHub
    git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
    # 2. Install pre-commit hooks (optional, recommended for contributing to XGrammar)
    pre-commit install
    # 3. build and install XGrammar core and Python bindings
    python3 -m pip install .

**Step 3. Validate installation.** You may validate if XGrammar is compiled successfully in command line.
You should see the path you used to build from source with:

.. code:: bash

   python -c "import xgrammar; print(xgrammar)"

**Step 4. (Optional) Run Python Tests.** You will need a HuggingFace token and access to gated models to run the tests that have gated models.

.. code:: bash

    # Install the test dependencies
    python3 -m pip install ".[test]"

    # To run all tests including the ones that have gated models, you will need a HuggingFace token.
    huggingface-cli login --token YOUR_HF_TOKEN
    python3 -m pytest tests/python

    # To run a subset of tests that do not require gated models, you can skip the tests with:
    python3 -m pytest tests/python -m "not hf_token_required"
