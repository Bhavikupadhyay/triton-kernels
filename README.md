# Triton Kernels

22 GPU kernels written in [Triton](https://triton-lang.org), organized into 7 phases from elementwise ops to Flash Attention v2. Each kernel includes a correctness test against the PyTorch equivalent and a benchmark reporting GB/s (memory-bound) or TFLOPS (compute-bound).

All benchmarks run on NVIDIA Tesla T4 (Google Colab Pro).

---

## Kernels

### Phase 1 — Elementwise

| Kernel | File | PyTorch equivalent | Peak (GB/s) |
|---|---|---|---|
| Vector add | `kernels/elementwise/vector_add.py` | `a + b` | 241 |
| Activations | `kernels/elementwise/activations.py` | `F.relu`, `F.gelu`, `F.silu` | 235–238 |
| Fused bias + activation | `kernels/elementwise/fused_elementwise.py` | `F.relu(x + bias)` | 245 fused / 145 unfused |

Fused bias+activation eliminates one HBM round-trip: 1.69× measured speedup, matching the theoretical 5/3 ratio.

---

### Phase 2 — Reductions

| Kernel | File | PyTorch equivalent | Peak (GB/s) |
|---|---|---|---|
| Reduce sum | `kernels/reductions/reduce_sum.py` | `x.sum()` | 286 |
| Argmax / Argmin | `kernels/reductions/max_min.py` | `x.argmax()` | 275 |
| Softmax | `kernels/reductions/softmax.py` | `F.softmax` | 222 |
| Layer norm | `kernels/reductions/layer_norm.py` | `F.layer_norm` | 222 |

reduce_sum exceeds the 241 GB/s elementwise ceiling because it is read-only (no HBM write; scratch stays in L2). Layer norm fuses mean, variance, normalise, and affine transform in a single pass — 40% ahead of PyTorch at N≥4096 where PyTorch's multi-pass implementation spills from L2.

---

### Phase 3 — Scanning

| Kernel | File | PyTorch equivalent | Notes |
|---|---|---|---|
| Prefix sum | `kernels/scanning/prefix_sum.py` | `x.cumsum(0)` | 158 GB/s at n=512K; 2× gap at large n vs PyTorch |
| Cummax | `kernels/scanning/cummax.py` | `x.cummax(0)` | ~50× ahead of PyTorch at n=16M |

Both use three kernel launches communicating through global memory — a consequence of Triton's single-tier model (no cross-block sync within one launch). `torch.cummax` uses a near-sequential CUDA kernel; the parallel scan is ~50× faster at large n.

---

### Phase 4 — Matrix Multiply

| Kernel | File | Notes |
|---|---|---|
| Naive matmul | `kernels/matmul/naive_matmul.py` | No tiling; baseline showing what tiling buys |
| Tiled matmul | `kernels/matmul/tiled_matmul.py` | SRAM tiling + L2-reuse group ordering + autotuning |
| Batched matmul | `kernels/matmul/batched_matmul.py` | Batch dim over tiled_matmul |

**Results (fp32, square M=N=K):**

| N | naive_matmul | tiled_matmul | cuBLAS |
|---|---|---|---|
| 512 | 1.06 TFLOPS | ~3.5 TFLOPS | ~4.1 TFLOPS |
| 1024 | 2.48 TFLOPS | ~3.8 TFLOPS | ~4.5 TFLOPS |
| 4096 | 1.77 TFLOPS | 3.84 TFLOPS | 3.89 TFLOPS |

tiled_matmul matches cuBLAS within 1% at N=4096. batched_matmul (B=16) is 10–13% ahead of `torch.bmm` at N=4096. T4 tensor cores only support FP16/INT8; all fp32 runs on CUDA cores (8.1 TFLOPS theoretical).

---

### Phase 5 — FFT

| Kernel | File | Notes |
|---|---|---|
| Cooley-Tukey FFT | `kernels/fft/fft_kernel.py` | Radix-2 DIT; in-SRAM butterfly for N≤8192 |

~1200 GFLOPS at N=4096 vs ~420 GFLOPS (`torch.fft.fft`). Single-pass in-SRAM execution avoids the HBM round-trips that cuFFT incurs for multi-pass plans at small N.

---

### Phase 6 — Attention

| Kernel | File | Notes |
|---|---|---|
| Naive attention | `kernels/attention/naive_attention.py` | O(N²) memory; baseline |
| SDPA | `kernels/attention/sdpa.py` | Causal masking + K/V tile skipping for masked region |
| Multi-head attention | `kernels/attention/multi_head_attention.py` | Packed (B, N, H×d) layout |
| Flash Attention v1 | `kernels/attention/flash_attention_v1.py` | Online softmax; O(N·d) HBM |
| Flash Attention v2 | `kernels/attention/flash_attention_v2.py` | Q-block parallelism; autotuned tiles; split causal/full K/V loop |

Flash Attention v2 keeps the running (max, sum) state in registers, accumulating partial output tiles without materialising the N×N attention matrix in HBM. This is the algorithmic advance from [Dao et al. 2022](https://arxiv.org/abs/2205.14135) and [2023](https://arxiv.org/abs/2307.08691).

---

### Phase 7 — Convolution

| Kernel | File | PyTorch equivalent | Notes |
|---|---|---|---|
| Conv1d | `kernels/convolution/conv1d.py` | `F.conv1d` | Implicit GEMM; 9% gap vs cuDNN at N=131072 |
| Conv2d | `kernels/convolution/conv2d.py` | `F.conv2d` | 2D spatial tiling; weight transposed to (C_out, K, K, C_in) for contiguous loads |
| Depthwise conv2d | `kernels/convolution/depthwise_conv2d.py` | `F.conv2d(..., groups=C)` | Per-channel sliding window; within 15% of PyTorch everywhere |

**Conv2d results (B=1, C_in=C_out=64, K=3, fp32):**

| H | Triton | PyTorch |
|---|---|---|
| 128 | 4.27 TFLOPS | 6.79 TFLOPS |
| 256 | 4.32 TFLOPS | 5.96 TFLOPS |
| 512 | 3.97 TFLOPS | 5.14 TFLOPS |

The 1.3–1.6× gap at H≥128 is algorithmic: cuDNN selects Winograd F(2,3) for K=3 stride-1, reducing the multiply count ~2.25× vs direct convolution. The TFLOPS metric counts standard direct-conv FLOPs for both providers, so cuDNN's fewer operations appear as higher throughput.

Depthwise conv has arithmetic intensity ~1.8 FLOPs/byte (K=3, fp32) — well below T4's ridge point of 25 FLOPs/byte — so both Triton and PyTorch are HBM-bandwidth-limited. The 0.27 TFLOPS plateau corresponds to ~150 GB/s effective bandwidth (~63% of the elementwise ceiling). The low TFLOPS number is an artefact of the metric, not a sign of inefficiency.

---

## Project Structure

```
kernels/           # Triton kernel source (importable package)
├── elementwise/
├── reductions/
├── scanning/
├── matmul/
├── fft/
├── attention/
└── convolution/
notebooks/         # One Jupyter notebook per phase — run on Colab
tests/             # pytest tests (require CUDA — run on Colab)
benchmarks/
└── results/       # PNGs and CSVs saved after Colab runs
scripts/
└── setup_colab.sh
```

---

## Running on Colab

Each notebook has a setup cell that mounts Google Drive, clones or updates the repo, and verifies the GPU. Run it first, then **File → Revert to saved** before running any other cells (forces Colab to re-read the notebook from disk after `git reset --hard`).

```python
import os
from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/Bhavikupadhyay/triton-kernels.git"
REPO_DIR = "/content/drive/MyDrive/triton-kernels"

if os.path.exists(REPO_DIR):
    !git -C {REPO_DIR} fetch --all
    !git -C {REPO_DIR} checkout -f main
    !git -C {REPO_DIR} reset --hard origin/main
else:
    !git clone {REPO_URL} {REPO_DIR}

os.chdir(REPO_DIR)
!bash scripts/setup_colab.sh
```

---

## Local Development

Triton requires CUDA and is not installable on macOS or Windows without a GPU. Local development is limited to writing and linting.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

make lint    # ruff check
make format  # ruff format
```
