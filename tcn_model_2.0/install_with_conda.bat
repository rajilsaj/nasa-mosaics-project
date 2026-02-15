@echo off
REM Install PyTorch and dependencies using conda on Windows

echo Installing PyTorch and dependencies with conda...

REM Install PyTorch with conda (CPU version)
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

REM Install other dependencies
conda install -y numpy pandas matplotlib scikit-learn
pip install xarray

echo.
echo Installation complete! Test with: python -c "import torch; print(torch.__version__)"
pause
