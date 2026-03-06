from kernels.scanning.prefix_sum import test_prefix_sum
from kernels.scanning.cummax import test_cummax


def test_prefix_sum_correctness():
    test_prefix_sum()


def test_cummax_correctness():
    test_cummax()
