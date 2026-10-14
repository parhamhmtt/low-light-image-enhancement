import torch
from llie.model import LowLightUNet


def test_model_output_shape_and_range():
    model = LowLightUNet(base_channels=4).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 64, 80))
    assert output.shape == (1, 3, 64, 80)
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_model_handles_odd_spatial_dimensions():
    model = LowLightUNet(base_channels=4).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 65, 79))
    assert output.shape == (1, 3, 65, 79)
