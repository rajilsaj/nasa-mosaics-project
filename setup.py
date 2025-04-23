from setuptools import setup, find_packages

setup(
    name="vortexdetect",
    version="0.1.0",
    description="Utilities for vortex detection and analysis in time-series pressure data",
    author="Vembe Sajila Rajil",
    packages=find_packages(include=["vortexdetect", "vortexdetect.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "xgboost",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
