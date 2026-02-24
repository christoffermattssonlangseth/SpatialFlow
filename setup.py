from setuptools import setup, find_packages

setup(
    name="spatialflow",
    version="4.0.0",
    description=(
        "EAE-SpatialFlow v4 — Physics-informed spatio-temporal GNN for "
        "trajectory-divergence analysis in EAE/MS spatial transcriptomics"
    ),
    long_description=open("README.md").read() if __import__("pathlib").Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="EAE-SpatialFlow Team",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "scripts"]),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "scikit-image>=0.20.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "pyarrow>=12.0.0",
    ],
    extras_require={
        "full": [
            "torch-geometric>=2.3.0",
            "captum>=0.7.0",
            "matplotlib>=3.7.0",
            "umap-learn>=0.5.3",
            "anndata>=0.9.0",
        ],
        "leiden": [
            "leidenalg>=0.10.0",
            "igraph>=0.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    entry_points={
        "console_scripts": [
            "spatialflow-train=scripts.train:main",
            "spatialflow-evaluate=scripts.evaluate:main",
        ],
    },
)
