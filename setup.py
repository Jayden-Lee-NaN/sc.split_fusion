from pathlib import Path

from setuptools import find_packages, setup


# Single source of truth for the package version: read from splitfuser/__version__.py
here = Path(__file__).resolve().parent
version_ns = {}
version_file = here / "splitfuser" / "__version__.py"

with version_file.open(encoding="utf-8") as vf:
    exec(vf.read(), version_ns)

version = version_ns.get("__version__")


setup(
    name="splitfuser",
    version=version,
    packages=find_packages(),
)
from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="splitfuser",
        version="0.1.0",
        packages=find_packages(),
        install_requires=["torch>=2.2", "diffusers==0.24.0", "transformers==4.29.2", "huggingface-hub==0.23.0", "tqdm"],
        python_requires=">=3.10",
        include_package_data=True,
    )
