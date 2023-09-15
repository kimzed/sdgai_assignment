import numpy as np
from torch import Tensor


def tensor_to_image(tensor: Tensor) -> np.ndarray:
    """
    Converts a tensor to a numpy array.

    Parameters:
    - tensor: Tensor to be converted.

    Returns:
    - A numpy array.
    """

    if len(tensor.shape) != 3:
        raise ValueError("The tensor should be 3D")
    if tensor.shape[0] != 3:
        raise ValueError("The 1st dimension should be 3")

    # we remove potential tracking of gradients, and move the tensor to the cpu
    image = tensor.cpu().detach().numpy().transpose(1, 2, 0)

    return image
