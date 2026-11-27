#!/usr/bin/env bash
# Linux x86_64, CPython 3.12, CUDA 12. The Mamba wheels match PyTorch's C++ ABI.
set -euo pipefail
cd "$(dirname "$0")/.."
venv=.venvs/speech-v0
if [[ ! -x "$venv/bin/python" ]]; then
    uv venv --python 3.12 "$venv"
fi
reference=${SPEECH_REFERENCE_PYTHON:-../audio-db/.venvs/asr_guided_se/bin/python}
wheel_dir="$venv/native-wheels"
tag=cp312-cp312-linux_x86_64.whl
mamba="$wheel_dir/mamba_ssm-2.3.2.post1+mamba1-$tag"
causal="$wheel_dir/causal_conv1d-1.7.0-$tag"
if [[ ! -f "$mamba" || ! -f "$causal" ]]; then
    if [[ -x "$reference" ]]; then
        "$reference" scripts/speech_native_wheels.py "$wheel_dir"
    else
        # Published wheels for newer Linux systems, with the PyTorch 2.6 C++ ABI.
        abi=cu12torch2.6cxx11abiFALSE-$tag
        mamba="https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2B$abi"
        causal="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2B$abi"
    fi
fi
uv pip install --python "$venv/bin/python" --torch-backend cu124 -e '.[speech-v0,test]' "$mamba" "$causal"

# Expose the wheel-provided NVVM and runtime in the layout Numba discovers.
nvidia_libs=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia")')
mkdir -p "$venv/cuda/nvvm/lib64"
ln -sfn "$nvidia_libs/cuda_nvcc/nvvm/lib64/libnvvm.so" "$venv/cuda/nvvm/lib64/libnvvm.so.4"
ln -sfn "$nvidia_libs/cuda_nvcc/nvvm/libdevice" "$venv/cuda/nvvm/libdevice"
ln -sfn "$nvidia_libs/cuda_runtime/lib" "$venv/cuda/lib64"
"$venv/bin/python" -c 'import torch, causal_conv1d_cuda, selective_scan_cuda; from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn'
