"""Package the reference's PyTorch 2.6 CUDA binaries for reuse on the same platform."""

import sys
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED

from wheel.wheelfile import WheelFile


# SEMamba++ uses Mamba-1. Mamba-3's TileLang and Triton dependencies need a
# different PyTorch ABI. Include the upstream source files needed by Mamba-1.
MAMBA_MODULES = {
    "modules/mamba_simple.py", "ops/selective_scan_interface.py", "ops/triton/layer_norm.py",
    "ops/triton/selective_state_update.py", "ops/triton/softplus.py", "utils/determinism.py", "utils/torch.py",
}


def export_wheel(directory, package, release, extension):
    dist = metadata.distribution(package)
    if dist.version != release:
        raise ValueError(f"Expected {package} {release}, found {dist.version}")
    mamba = package == "mamba_ssm"
    version = release + "+mamba1" if mamba else release
    info = f"{package}-{version}.dist-info"
    filename = f"{package}-{version}-cp312-cp312-linux_x86_64.whl"
    dest = directory / filename
    if dest.is_file():
        return
    with TemporaryDirectory(dir=directory) as tmp:
        archive = Path(tmp) / filename
        with WheelFile(archive, "w", compression=ZIP_STORED) as wheel:
            for entry in dist.files:
                root, *parts = entry.parts
                if root.startswith(extension + ".") and root.endswith(".so"):
                    wheel.write(dist.locate_file(entry), str(entry))
                elif root.endswith(".dist-info") and parts[0] == "licenses":
                    wheel.write(dist.locate_file(entry), "/".join([info, *parts]))
                elif root == package and entry.suffix == ".py":
                    if not mamba or "/".join(parts) in MAMBA_MODULES or parts[-1] == "__init__.py" and len(parts) > 1:
                        wheel.write(dist.locate_file(entry), str(entry))
            wheel.writestr(f"{info}/WHEEL", dist.read_text("WHEEL"))
            if mamba:
                wheel.writestr("mamba_ssm/__init__.py", f'__version__ = "{version}"\n'
                               "from .ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn\n"
                               "from .modules.mamba_simple import Mamba\n")
                wheel.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: mamba-ssm\nVersion: {version}\n"
                               "Summary: Mamba-1 selective scan and normalization from the audio-db build\n"
                               "License: Apache-2.0\nRequires-Dist: torch==2.6.0\nRequires-Dist: triton==3.2.0\n"
                               "Requires-Dist: einops\nRequires-Dist: packaging\nRequires-Dist: causal-conv1d==1.7.0\n")
            else:
                wheel.writestr(f"{info}/METADATA", dist.read_text("METADATA"))
        archive.replace(dest)


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 12) or metadata.version("torch") != "2.6.0+cu124":
        raise SystemExit("Expected a CPython 3.12 reference environment with PyTorch 2.6.0+cu124")
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    export_wheel(directory, "causal_conv1d", "1.7.0", "causal_conv1d_cuda")
    export_wheel(directory, "mamba_ssm", "2.3.2.post1", "selective_scan_cuda")
