from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def confusion_matrix_visualization(
    predictions: List[int],
    ground_truth: List[int],
    labels: List[str],
    save_path: Optional[Path] = None,
) -> None:
    """
    Generates a confusion matrix from predictions and ground truth labels
    and maps integer classes to their labels.
    """

    classes = np.unique(np.concatenate((predictions, ground_truth)))
    if len(classes) != len(labels):
        raise ValueError(
            "Number of labels must match the number of "
            "unique classes in predictions and ground truth."
        )

    confusion_matrix_data = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix_data,
        index=labels,
        columns=labels,
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix_df, annot=True)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    if save_path:
        plt.savefig(save_path)
    plt.show()
