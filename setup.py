#!/usr/bin/env python3
"""
Setup script for guidedLP package.

This setup.py provides a traditional installation method for the 
Guided Label Propagation library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    """Read README.md for long description."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Large-scale network analysis with Guided Label Propagation"

# Read version from __init__.py
def get_version():
    """Extract version from src/__init__.py."""
    version = {}
    try:
        with open("src/__init__.py", "r") as f:
            exec(f.read(), version)
        return version.get("__version__", "0.1.0")
    except FileNotFoundError:
        return "0.1.0"

setup(
    name="guidedLP",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="Large-scale network analysis with Guided Label Propagation for computational social science",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/guided-label-propagation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Sociology",
    ],
    python_requires=">=3.9",
    install_requires=[
        "networkit>=11.0",
        "polars>=0.20.0", 
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.1.0",
            "black>=23.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "viz": [
            "matplotlib>=3.6",
            "plotly>=5.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)