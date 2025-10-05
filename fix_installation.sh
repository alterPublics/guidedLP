#!/bin/bash
# Fix script for guidedLP installation issues

echo "=== Fixing guidedLP Installation Issues ==="
echo ""

echo "1. Checking current versions..."
python3 --version
pip3 --version
pip3 show setuptools | grep Version:

echo ""
echo "2. Upgrading build tools..."

# Upgrade setuptools and build tools
pip3 install --upgrade --user setuptools>=61.0 wheel build

echo ""
echo "3. Clearing pip cache..."
pip3 cache purge

echo ""
echo "4. Installing guidedLP..."
pip3 install --user --no-cache-dir git+https://github.com/alterPublics/guidedLP.git

echo ""
echo "5. Verifying installation..."
pip3 show guidedLP

echo ""
echo "6. Testing import..."
python3 -c "import guidedLP; print(f'✅ Successfully imported guidedLP version {guidedLP.__version__}')"

echo ""
echo "=== Installation complete! ==="