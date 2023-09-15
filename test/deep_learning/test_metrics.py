import numpy as np

from deep_learning.metrics import confusion_matrix_visualization
import pytest


def test_confusion_matrix_mismatched_labels_returns_error():
    predictions = [1, 2, 3]
    ground_truth = [1, 2, 3]
    labels = ["Class A", "Class B"]
    with pytest.raises(ValueError):
        confusion_matrix_visualization(predictions, ground_truth, labels)
