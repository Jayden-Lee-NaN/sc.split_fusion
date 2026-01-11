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
