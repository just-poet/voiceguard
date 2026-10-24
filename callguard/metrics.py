import numpy as np
from sklearn.metrics import roc_curve


def eer(labels, scores):
    """ROC crossing with linear interpolation; higher score means synthetic."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('EER requires finite scores and both genuine and synthetic examples.')
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    index = np.flatnonzero(fpr >= fnr)[0]
    left = max(0, index - 1)
    gap_left = fpr[left] - fnr[left]
    gap_right = fpr[index] - fnr[index]
    weight = -gap_left / (gap_right - gap_left) if gap_right != gap_left else 0
    value = fpr[left] + weight * (fpr[index] - fpr[left])
    # Choose a real, finite operating threshold nearest the crossing.
    finite = np.flatnonzero(np.isfinite(thresholds))
    pick = finite[np.argmin(np.abs(fpr[finite] - fnr[finite]))]
    return {'eer_percent': round(float(value * 100), 4), 'threshold': float(thresholds[pick])}


def binary_metrics(labels, predictions):
    y, p = np.asarray(labels, bool), np.asarray(predictions, bool)
    tp = int(np.sum(y & p)); fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0,
            'tp': tp, 'fp': fp, 'fn': fn}
