#!/usr/bin/env bash

source /multibuild/manylinux_utils.sh
source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12

function usage() {
    echo "Usage: $0"
}

function in_array() {
    KEY=$1
    ARRAY=$2
    for e in ${ARRAY[*]}; do
        if [[ "$e" == "$1" ]]; then
            return 0
        fi
    done
    return 1
}

function build_xgrammar_wheel() {
    python_dir=$1
    PYTHON_BIN="${python_dir}/bin/python"

    cd "${XGRAMMAR_PYTHON_DIR}" &&
        ${PYTHON_BIN} setup.py bdist_wheel
}

function audit_xgrammar_wheel() {
    python_version_str=$1

    cd "${XGRAMMAR_PYTHON_DIR}" &&
        mkdir -p repaired_wheels &&
        auditwheel repair ${AUDITWHEEL_OPTS} dist/*cp${python_version_str}*.whl

    rm -rf ${XGRAMMAR_PYTHON_DIR}/dist/ \
        ${XGRAMMAR_PYTHON_DIR}/build/ \
        ${XGRAMMAR_PYTHON_DIR}/*.egg-info
}

XGRAMMAR_PYTHON_DIR="/workspace/python"
PYTHON_VERSIONS_CPU=("3.9" "3.10" "3.11" "3.12" "3.13")

while [[ $# -gt 0 ]]; do
    arg="$1"
    case $arg in
    -h | --help)
        usage
        exit -1
        ;;
    *) # unknown option
        echo "Unknown argument: $arg"
        echo
        usage
        exit -1
        ;;
    esac
done

echo "Building XGrammar for CPU only"
PYTHON_VERSIONS=${PYTHON_VERSIONS_CPU[*]}

AUDITWHEEL_OPTS="--plat ${AUDITWHEEL_PLAT} -w repaired_wheels/"
AUDITWHEEL_OPTS="--exclude libtorch --exclude libtorch_cpu --exclude libtorch_python ${AUDITWHEEL_OPTS}"

UNICODE_WIDTH=32 # Dummy value, irrelevant for Python 3

source /opt/conda/etc/profile.d/conda.sh

# Not all manylinux Docker images will have all Python versions,
# so check the existing python versions before generating packages
for python_version in ${PYTHON_VERSIONS[*]}; do
    echo "> Looking for Python ${python_version}."
    # Remove the . in version string, e.g. "3.8" turns into "38"
    python_version_str="$(echo "${python_version}" | sed -r 's/\.//g')"
    cpython_dir="/opt/conda/envs/py${python_version_str}/"

    # compile xgrammar on the particular python version
    conda activate py${python_version_str}

    # setup config.cmake
    cd /workspace
    rm -rf ${XGRAMMAR_PYTHON_DIR}/xgrammar/*.so
    rm -rf config.cmake
    echo set\(XGRAMMAR_BUILD_PYTHON_BINDINGS ON\) >>config.cmake
    echo set\(XGRAMMAR_BUILD_CXX_TESTS OFF\) >>config.cmake

    python3 -m pip install nanobind
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make -j4
    find . -type d -name 'CMakeFiles' -exec rm -rf {} +

    # For compatibility in environments where Conda is not installed,
    # revert back to previous method of locating cpython_dir.
    if ! [ -d "${cpython_dir}" ]; then
        cpython_dir=$(cpython_path "${python_version}" "${UNICODE_WIDTH}" 2>/dev/null)
    fi

    if [ -d "${cpython_dir}" ]; then
        echo "Generating package for Python ${python_version}."
        build_xgrammar_wheel ${cpython_dir}

        echo "Running auditwheel on package for Python ${python_version}."
        audit_xgrammar_wheel ${python_version_str}
    else
        echo "Python ${python_version} not found. Skipping."
    fi

done
