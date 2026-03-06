from kernels.reductions.reduce_sum import test_reduce_sum
from kernels.reductions.max_min import test_max_min
from kernels.reductions.softmax import test_softmax
from kernels.reductions.layer_norm import test_layer_norm


def test_reduce_sum_correctness():
    test_reduce_sum()


def test_max_min_correctness():
    test_max_min()


def test_softmax_correctness():
    test_softmax()


def test_layer_norm_correctness():
    test_layer_norm()
