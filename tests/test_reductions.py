from kernels.reductions.reduce_sum import test_reduce_sum
from kernels.reductions.max_min import test_max_min


def test_reduce_sum_correctness():
    test_reduce_sum()


def test_max_min_correctness():
    test_max_min()
