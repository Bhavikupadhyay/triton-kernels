#!/usr/bin/env bash
# setup_colab.sh — Run this in a Colab cell to set up the environment
# Usage: !bash scripts/setup_colab.sh  (from repo root)
#   or:  !bash /content/triton-kernels/scripts/setup_colab.sh

set -e

echo "=== Triton Kernels: Colab Setup ==="

# 1. Upgrade triton to a compatible version
echo "[1/4] Installing extra dependencies..."
pip install -q --upgrade triton>=2.3.0
pip install -q matplotlib>=3.8.0 pandas>=2.0.0

# 2. Install the kernels package in editable mode
echo "[2/4] Installing kernels package..."
# Determine repo root (works whether run from root or scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -q -e "$REPO_ROOT"

# 3. Verify GPU is available
echo "[3/4] Verifying GPU..."
python - <<'EOF'
import torch
if not torch.cuda.is_available():
    print("WARNING: No GPU detected. Benchmarks will not work correctly.")
else:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  SM count: {props.multi_processor_count}")
    print(f"  Compute capability: {props.major}.{props.minor}")
EOF

# 4. Print environment report
echo "[4/4] Environment report..."
python - <<'EOF'
import sys, torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("Triton: NOT INSTALLED")
try:
    import kernels
    print("kernels package: OK")
except ImportError as e:
    print(f"kernels package: FAILED ({e})")
EOF

echo ""
echo "=== Setup complete. Ready to run kernels. ==="
