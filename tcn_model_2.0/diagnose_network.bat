@echo off
REM Network diagnostic script for PyTorch installation issues on Windows

echo === Network Diagnostic for PyTorch Installation ===
echo.

echo 1. Checking internet connectivity...
ping -n 1 8.8.8.8 >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ Can reach Google DNS (8.8.8.8) - basic connectivity OK
) else (
    echo    ✗ Cannot reach 8.8.8.8 - no internet connection
    echo    → Fix: Check your network connection
    pause
    exit /b 1
)

echo.
echo 2. Checking DNS resolution...
ping -n 1 pypi.org >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✓ Can resolve pypi.org - DNS working
) else (
    echo    ✗ Cannot resolve pypi.org - DNS issue
    echo    → Fix: Try using Google DNS (8.8.8.8) or restart network
    echo.
    echo    To fix DNS on Windows:
    echo    1. Settings → Network ^& Internet → Change adapter options
    echo    2. Right-click your connection → Properties → IPv4 → Properties
    echo    3. Use the following DNS: 8.8.8.8 and 8.8.4.4
    echo    4. Apply and restart network
)

echo.
echo 3. Checking Python versions...
python --version 2>nul
if %errorlevel% neq 0 (
    echo    ✗ Python not found in PATH
) else (
    python --version
)

echo.
echo 4. Checking if PyTorch is already installed...
python -c "import torch; print('   ✓ PyTorch is installed:', torch.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo    ✗ PyTorch is NOT installed
)

echo.
echo === Recommendations ===
echo.
echo If DNS is working, try:
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo.
echo If DNS is NOT working, you need to:
echo   1. Fix your network/DNS settings (see above)
echo   2. Or download PyTorch wheels manually from another machine
echo   3. Or wait until network is restored
echo.
pause
