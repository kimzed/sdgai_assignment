import numpy as np
import torch

from utils.deep_learning_functions import tensor_to_image


def test_tensor_to_image_returns_numpy_array():

    # arrange
    tensor = torch.rand(3, 100, 100)

    # act
    image = tensor_to_image(tensor)

    # assert
    assert isinstance(image, np.ndarray)
    assert image.shape == (100, 100, 3)
