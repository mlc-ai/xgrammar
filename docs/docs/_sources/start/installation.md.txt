# Installation

XGrammar Python Package can be installed directly from a prebuilt package or built from source.

## Method 1: Prebuilt Package

XGrammar supports various platforms:
* Operating Systems: Linux, macOS, and Windows
* Hardware: CPU, NVIDIA GPUs, AMD GPUs, Apple Silicon, TPU, etc.
* Python: 3.9 and later

We provide Python wheels for XGrammar via pip.

```bash
python -m pip install xgrammar
```

We also provide conda packages for XGrammar:

```bash
conda install -c conda-forge xgrammar
```

Use the following command to verify installation:

```bash
python -c "import xgrammar; print(xgrammar)"
# Prints: <module 'xgrammar' from '/path-to-env/lib/python3.11/site-packages/xgrammar/__init__.py'>
```

## Method 2: Build XGrammar Python Package from Source

This option is useful when you want to make modification or obtain a specific version of XGrammar.

```bash
git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
pre-commit install
# Copy cmake config. You can update the config if needed.
cp cmake/config.cmake .
python3 -m pip install --no-build-isolation -e .
```

XGrammar is a library written in C++ and Python. The editable install will automatically rebuild
the package when XGrammar is imported in Python.

### Optional: Run Python Tests

```bash
# Install the test dependencies
python3 -m pip install ".[test]"

# If you have a HuggingFace token, you can run all tests including the ones that have gated models.
huggingface-cli login --token YOUR_HF_TOKEN
python3 -m pytest

# If you do not have a HuggingFace token, you can run a subset of tests that do not require gated models.
python3 -m pytest -m "not hf_token_required"
```

## Method 3: Build XGrammar C++ Library Only

XGrammar can also be build as a C++ library. This is useful for using XGrammar in C++ or Rust projects.

XGrammar uses CMake and Ninja to build the C++ library.

```bash
git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
# Copy cmake config. You can update the config if needed.
cp cmake/config.cmake .
mkdir build && cd build
cmake -G Ninja ..
ninja
```

### Optional: Run C++ Tests

```bash
# Run all tests
bash scripts/run_ctest.sh

# Run a subset of tests whose name contains "test_name"
bash scripts/run_ctest.sh test_name
```
