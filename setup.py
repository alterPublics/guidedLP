#!/usr/bin/env python3
"""
Fallback setup.py for maximum compatibility with older pip/setuptools.

This ensures the package installs correctly even on systems with older tools.
"""

from setuptools import setup

# Use setuptools to read pyproject.toml
setup()