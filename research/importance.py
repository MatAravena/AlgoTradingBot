import numpy as np
import pandas as pd
from .scoring import pwa, neg_logloss

def mda_importance(model, X: pd.DataFrame, y: pd.Series, scorer='pwa', n_rounds=5, random_state=42):
    rng = np.random.RandomState(random_state)
    base_proba = model.predict_proba(X)[:,1]
    base = pwa(y, base_proba) if scorer=='pwa' else neg_logloss(y, base_proba)
    imps = []
    for col in X.columns:
        scores = []
        for _ in range(n_rounds):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            proba = model.predict_proba(Xp)[:,1]
            s = pwa(y, proba) if scorer=='pwa' else neg_logloss(y, proba)
            scores.append(base - s)  # drop in score
        imps.append({'feature': col, 'importance': float(np.mean(scores)), 'std': float(np.std(scores))})
    imp_df = pd.DataFrame(imps).sort_values('importance', ascending=False).reset_index(drop=True)
    return base, imp_df
