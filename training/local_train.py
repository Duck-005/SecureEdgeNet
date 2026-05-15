from __future__ import annotations

import numpy as np


def positive_class_weight(y: np.ndarray) -> float:
    positives = max(int(np.sum(y == 1)), 1)
    negatives = max(int(np.sum(y == 0)), 1)
    return float(negatives / positives)
