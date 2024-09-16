from __future__ import annotations

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.class_weight import compute_class_weight


def discretize_and_classify(array, bins=2):
    """
    Discretizes and classifies a 2D array into a specified number of bins.
    Parameters:
    - array (ndarray): Input array with shape (n, 2), where n is the number of samples.
    - bins (int): Number of bins to discretize the array into. Default is 2.
    Returns:
    - class_index (ndarray): Array of class indices corresponding to each sample in the input array.
    Raises:
    - AssertionError: If the input array does not have 2 columns.
    - AssertionError: If the number of bins is not positive.
    - AssertionError: If the values in the array are not in the range [-1, 1].
    """
    assert array.shape[1] == 2, "Input array must have 2 columns"
    assert bins > 0, "Number of bins m must be positive"
    assert np.all(np.abs(array) <= 1), "Values in array must be in the range [-1, 1]"
    discretizer = KBinsDiscretizer(n_bins=[bins, bins], encode="ordinal", strategy="uniform")
    discretizer.fit(np.array([[1.0, 1.0], [-1.0, -1.0]]))
    binned = discretizer.transform(array)
    class_index = binned[:, 0] * bins + binned[:, 1]  # class = x * bins + y, steering * bins + throttle  # type: ignore
    return class_index.astype(int)


def discretize_and_compute_balancing_weights(array, bins=2):
    """
    Discretizes the given array into specified number of bins and computes balancing weights for each class.
    Parameters:
    - array: The input array to be discretized and balanced.
    - bins: The number of bins to discretize the array into. Default is 2.
    Returns:
    - weights: The balancing weights for each element in the input array.
    """
    classes = discretize_and_classify(array, bins=bins)
    unique_classes = np.unique(classes)
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=classes)
    print(f"Unique classes: {unique_classes}, Class weights: {class_weights}")
    weight_lookup = np.zeros(np.max(unique_classes) + 1)
    weight_lookup[unique_classes] = class_weights
    weights = weight_lookup[classes]
    return weights
