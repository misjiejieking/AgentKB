#!/usr/bin/env bash
set -e

echo "========================================"
echo "  AgentKB - Personal Knowledge Agent"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.11+."
    exit 1
fi

# Create data directories
mkdir -p data/uploads data/vectors data/logs

# Install dependencies
echo "[1/2] Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null || echo "[WARN] pip install failed, trying to continue..."

# Run
echo "[2/2] Starting AgentKB..."
export PYTHONPATH="src:$PYTHONPATH"
python3 -m agentkb.main
