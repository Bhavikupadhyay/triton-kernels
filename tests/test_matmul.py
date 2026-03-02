from kernels.matmul.naive_matmul import test_naive_matmul
from kernels.matmul.tiled_matmul import test_tiled_matmul

# from kernels.matmul.batched_matmul import test_batched_matmul


def test_naive_matmul_correctness():
    test_naive_matmul()


def test_tiled_matmul_correctness():
    test_tiled_matmul()

# def test_batched_matmul_correctness():
#     test_batched_matmul()
