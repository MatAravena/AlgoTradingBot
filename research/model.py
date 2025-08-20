import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from .scoring import pwa, neg_logloss

def fit_model(X, y, sample_weight=None, n_splits=5, embargo_pct=0.05,
              n_estimators=400, max_depth=6, min_samples_leaf=3, class_weight='balanced'):
    from .sampling import PurgedKFold
    X = X.astype(float).replace([np.inf,-np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    y_bin = (y > 0).astype(int)

    base = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        n_jobs=-1, class_weight=class_weight, random_state=42
    )
    # Calibrated for better proba
    clf = CalibratedClassifierCV(base, method='isotonic', cv=3)

    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    scores = []
    for tr, te in cv.split(X, t1=pd.Series(X.index, index=X.index)):
        clf.fit(X.iloc[tr], y_bin.iloc[tr], sample_weight=None if sample_weight is None else sample_weight.iloc[tr])
        proba = clf.predict_proba(X.iloc[te])[:,1]
        scores.append({'pwa': pwa(y.iloc[te], proba), 'neg_logloss': neg_logloss(y.iloc[te], proba)})
    return clf, pd.DataFrame(scores)
