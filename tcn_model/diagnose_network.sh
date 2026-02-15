#!/bin/bash
# Network diagnostic script for PyTorch installation issues

echo "=== Network Diagnostic for PyTorch Installation ==="
echo ""

echo "1. Checking internet connectivity..."
if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
    echo "   ✓ Can reach Google DNS (8.8.8.8) - basic connectivity OK"
else
    echo "   ✗ Cannot reach 8.8.8.8 - no internet connection"
    echo "   → Fix: Check your network connection"
    exit 1
fi

echo ""
echo "2. Checking DNS resolution..."
if ping -c 1 pypi.org > /dev/null 2>&1; then
    echo "   ✓ Can resolve pypi.org - DNS working"
else
    echo "   ✗ Cannot resolve pypi.org - DNS issue"
    echo "   → Fix: Try using Google DNS (8.8.8.8) or restart network"
    echo ""
    echo "   To fix DNS on macOS:"
    echo "   1. System Settings → Network → Wi‑Fi/Ethernet → Details → DNS"
    echo "   2. Add 8.8.8.8 and 8.8.4.4"
    echo "   3. Apply and restart network"
fi

echo ""
echo "3. Checking Python versions..."
echo "   Python 3.14: $(python3 --version 2>&1)"
if command -v python3.12 &> /dev/null; then
    echo "   Python 3.12: $(python3.12 --version 2>&1)"
fi
if command -v python3.11 &> /dev/null; then
    echo "   Python 3.11: $(python3.11 --version 2>&1)"
fi

echo ""
echo "4. Checking if PyTorch is already installed..."
if python3 -c "import torch" 2>/dev/null; then
    echo "   ✓ PyTorch is installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    echo "   ✗ PyTorch is NOT installed"
fi

echo ""
echo "=== Recommendations ==="
echo ""
echo "If DNS is working, try:"
echo "  pip install torch torchvision torchaudio"
echo ""
echo "If DNS is NOT working, you need to:"
echo "  1. Fix your network/DNS settings (see above)"
echo "  2. Or download PyTorch wheels manually from another machine"
echo "  3. Or wait until network is restored"
echo ""
