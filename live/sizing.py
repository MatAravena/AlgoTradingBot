import numpy as np

def prob_vol_sizer(proba, vol, max_risk_per_trade=0.005):
    """
    Size increases with edge (proba-0.5) and decreases with volatility.
    Returns a fraction of equity to allocate (long only).
    """
    edge = np.clip(proba - 0.5, 0, 0.49)
    inv_vol = 1.0 / max(vol, 1e-6)
    raw = edge * inv_vol
    # normalize and cap
    return float(min(raw, max_risk_per_trade))
