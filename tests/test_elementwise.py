from kernels.elementwise.vector_add import test_vector_add
from kernels.elementwise.activations import test_activations


def test_vector_add_correctness():
    test_vector_add()


def test_activations_correctness():
    test_activations()
