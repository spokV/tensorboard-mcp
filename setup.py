#!/usr/bin/env python3
"""Setup script for TensorBoard MCP Server."""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tensorboard-mcp-server",
    version="1.0.0",
    description="TensorBoard Model Context Protocol (MCP) server for analyzing training runs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAIL Team",
    author_email="sail@stanford.edu",
    url="https://github.com/your-org/tensorboard-mcp",

    packages=find_packages(),

    python_requires=">=3.10",
    install_requires=[
        "mcp>=0.1.0",
        "tensorboard>=2.0.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ]
    },

    entry_points={
        "console_scripts": [
            "tensorboard-mcp-server=tb_mcp_server:main",
            "tensorboard-mcp=tb_mcp_server:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    keywords="tensorboard mcp model-context-protocol machine-learning ai",

    project_urls={
        "Bug Reports": "https://github.com/your-org/tensorboard-mcp/issues",
        "Source": "https://github.com/your-org/tensorboard-mcp",
        "Documentation": "https://github.com/your-org/tensorboard-mcp#readme",
    },
)