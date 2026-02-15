# Install PyTorch and dependencies using conda on Windows (PowerShell)

Write-Host "Installing PyTorch and dependencies with conda..." -ForegroundColor Green

# Install PyTorch with conda (CPU version)
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
conda install -y numpy pandas matplotlib scikit-learn
pip install xarray

Write-Host ""
Write-Host "Installation complete! Test with: python -c 'import torch; print(torch.__version__)'" -ForegroundColor Green
