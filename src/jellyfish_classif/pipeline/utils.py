from typing import Dict, List
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def compute_class_weights(labels: List[int]) -> Dict[int, float]:
    """Compute class weights to handle class imbalance.

    Args:
        labels (List[int]): List of class labels for the dataset.

    Returns:
        Dict[int, float]: Mapping class_index -> weight
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    class_weights = {int(cls): float(w) for cls, w in zip(classes, weights)}
    return class_weights
