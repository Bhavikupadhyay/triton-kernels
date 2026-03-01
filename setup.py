from setuptools import setup, find_packages

setup(
    name="triton-kernels",
    version="0.1.0",
    description="Progressive Triton GPU kernel implementations with PyTorch benchmarks",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "numpy>=1.26.0",
    ],
    extras_require={
        "colab": [
            "triton>=2.3.0",
            "matplotlib>=3.8.0",
            "pandas>=2.0.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "ruff>=0.4.0",
        ],
    },
)
