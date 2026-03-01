from kernels.elementwise.vector_add import test_vector_add
from kernels.elementwise.activations import test_activations
from kernels.elementwise.fused_elementwise import test_fused_elementwise


def test_vector_add_correctness():
    test_vector_add()


def test_activations_correctness():
    test_activations()


def test_fused_elementwise_correctness():
    test_fused_elementwise()
