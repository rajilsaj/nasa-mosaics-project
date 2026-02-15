# Network diagnostic script for PyTorch installation issues on Windows

Write-Host "=== Network Diagnostic for PyTorch Installation ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Checking internet connectivity..." -ForegroundColor Yellow
$pingResult = Test-Connection -ComputerName "8.8.8.8" -Count 1 -Quiet
if ($pingResult) {
    Write-Host "   ✓ Can reach Google DNS (8.8.8.8) - basic connectivity OK" -ForegroundColor Green
} else {
    Write-Host "   ✗ Cannot reach 8.8.8.8 - no internet connection" -ForegroundColor Red
    Write-Host "   → Fix: Check your network connection" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "2. Checking DNS resolution..." -ForegroundColor Yellow
try {
    $dnsResult = Resolve-DnsName -Name "pypi.org" -ErrorAction Stop
    Write-Host "   ✓ Can resolve pypi.org - DNS working" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Cannot resolve pypi.org - DNS issue" -ForegroundColor Red
    Write-Host "   → Fix: Try using Google DNS (8.8.8.8) or restart network" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   To fix DNS on Windows:" -ForegroundColor Yellow
    Write-Host "   1. Settings → Network & Internet → Change adapter options" -ForegroundColor Yellow
    Write-Host "   2. Right-click your connection → Properties → IPv4 → Properties" -ForegroundColor Yellow
    Write-Host "   3. Use the following DNS: 8.8.8.8 and 8.8.4.4" -ForegroundColor Yellow
    Write-Host "   4. Apply and restart network" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "3. Checking Python versions..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "   Python: $pythonVersion" -ForegroundColor Cyan

if (Get-Command python3.12 -ErrorAction SilentlyContinue) {
    $python312Version = python3.12 --version 2>&1
    Write-Host "   Python 3.12: $python312Version" -ForegroundColor Cyan
}
if (Get-Command python3.11 -ErrorAction SilentlyContinue) {
    $python311Version = python3.11 --version 2>&1
    Write-Host "   Python 3.11: $python311Version" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "4. Checking if PyTorch is already installed..." -ForegroundColor Yellow
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    Write-Host "   ✓ PyTorch is installed: $torchVersion" -ForegroundColor Green
} catch {
    Write-Host "   ✗ PyTorch is NOT installed" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Recommendations ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "If DNS is working, try:" -ForegroundColor Yellow
Write-Host "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor White
Write-Host ""
Write-Host "If DNS is NOT working, you need to:" -ForegroundColor Yellow
Write-Host "  1. Fix your network/DNS settings (see above)" -ForegroundColor White
Write-Host "  2. Or download PyTorch wheels manually from another machine" -ForegroundColor White
Write-Host "  3. Or wait until network is restored" -ForegroundColor White
Write-Host ""
