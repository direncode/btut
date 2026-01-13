from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="btut-sdk",
    version="1.0.0",
    author="BTUT Team",
    author_email="contact@btut.ai",
    description="Python SDK for BTUT multi-agent simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/btut/python-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0"],
    },
)
