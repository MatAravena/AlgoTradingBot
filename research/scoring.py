import numpy as np
from sklearn.metrics import log_loss

def pwa(y_true, proba_pos):
    """
    Probability-Weighted Accuracy:
    weight each hit by its probability; encourages calibrated models.
    """
    preds = (proba_pos >= 0.5).astype(int)
    hits = (preds == (y_true > 0)).astype(float)
    weights = np.maximum(np.where(preds==1, proba_pos, 1-proba_pos), 1e-6)
    return float((hits * weights).sum() / len(hits))

def neg_logloss(y_true, proba_pos):
    return -log_loss((y_true>0).astype(int), np.c_[1-proba_pos, proba_pos], labels=[0,1])
