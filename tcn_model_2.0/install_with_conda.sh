#!/bin/bash
# Install PyTorch and dependencies using conda

# Activate conda base environment
source /opt/miniconda3/bin/activate

# Install PyTorch with conda (CPU version)
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
conda install -y numpy pandas matplotlib scikit-learn
pip install xarray

echo "Installation complete! Test with: python -c 'import torch; print(torch.__version__)'"
