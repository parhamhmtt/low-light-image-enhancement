import numpy as np
from llie.classical import METHODS


def test_classical_methods_return_valid_image():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 80, size=(48, 64, 3), dtype=np.uint8)
    for name, method in METHODS.items():
        output = method(image)
        assert output.shape == image.shape, name
        assert output.dtype == np.uint8, name
        assert output.min() >= 0 and output.max() <= 255, name
