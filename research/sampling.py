import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.05):
        self.kf = KFold(n_splits=n_splits, shuffle=False)
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, t1: pd.Series):
        n = len(X)
        mbar = int(n * self.embargo_pct)
        for train_idx, test_idx in self.kf.split(X):
            test_st = test_idx[0]; test_en = test_idx[-1]
            # purge: drop train samples whose t1 overlaps test period
            test_start_time = X.index[test_st]
            test_end_time   = X.index[test_en]
            train_mask = np.ones(n, dtype=bool)
            # purge overlapping events
            overlaps = (t1.loc[X.index[train_mask]] >= test_start_time) & (X.index[train_mask] <= test_end_time)
            purge_idx = np.where(overlaps.values)[0]
            train_mask[purge_idx] = False
            # embargo around test
            emb_lo = max(0, test_st - mbar)
            emb_hi = min(n, test_en + mbar + 1)
            train_mask[emb_lo:emb_hi] = False
            yield np.where(train_mask)[0], test_idx
        return
