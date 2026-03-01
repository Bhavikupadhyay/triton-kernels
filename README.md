# Triton Kernels

Progressive Triton GPU kernel implementations benchmarked against PyTorch equivalents.

## Goal

Implement 15–22 kernels progressing from simple elementwise ops to Flash Attention v2, each with:
- A correct Triton implementation
- A PyTorch baseline for comparison
- Benchmark reporting GB/s (memory-bound) or TFLOPS (compute-bound)

## Kernel Roadmap

| Phase | Kernel | Status |
|---|---|---|
| 1 — Elementwise | `vector_add` | [ ] |
| 1 — Elementwise | `activations` (ReLU, GELU, SiLU) | [ ] |
| 1 — Elementwise | `fused_elementwise` | [ ] |
| 2 — Reductions | `vector_sum` | [ ] |
| 2 — Reductions | `max_min` | [ ] |
| 2 — Reductions | `softmax` | [ ] |
| 2 — Reductions | `layer_norm` | [ ] |
| 3 — Scanning | `prefix_sum` | [ ] |
| 3 — Scanning | `cummax` | [ ] |
| 4 — Matmul | `naive_matmul` | [ ] |
| 4 — Matmul | `tiled_matmul` | [ ] |
| 4 — Matmul | `batched_matmul` | [ ] |
| 5 — FFT | `fft_kernel` | [ ] |
| 6 — Attention | `naive_attention` | [ ] |
| 6 — Attention | `sdpa` | [ ] |
| 6 — Attention | `multi_head_attention` | [ ] |
| 6 — Attention | `flash_attention_v1` | [ ] |
| 6 — Attention | `flash_attention_v2` | [ ] |

## Setup

### Local (Mac, no GPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

# Run tests with Triton interpreter (no GPU needed)
make test-local
```

### Google Colab

```python
# In the first cell of any notebook:
!bash scripts/setup_colab.sh
```

Or manually:

```bash
!pip install -q -r requirements-colab.txt
!pip install -q -e .
```

## Workflow

1. Write kernel locally → run `make test-local` for logic validation
2. Push to GitHub
3. Open the relevant notebook in Colab Pro
4. Run correctness + benchmark cells
5. Pull benchmark results back, commit with `bench:` prefix

## Project Structure

```
kernels/       # Triton kernel source (importable package)
notebooks/     # One notebook per category, designed for Colab
benchmarks/    # Benchmark results (CSV + PNG) after Colab runs
tests/         # pytest tests, run locally with TRITON_INTERPRET=1
scripts/       # Colab setup script
```


