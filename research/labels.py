import numpy as np
import pandas as pd

def ewm_volatility(prices: pd.Series, span: int = 48) -> pd.Series:
    rets = prices.pct_change()
    return rets.ewm(span=span, adjust=False).std().fillna(method='bfill')

def get_barrier_times(index: pd.DatetimeIndex, t1_bars: int) -> pd.Series:
    # timeout index for each observation
    locs = np.arange(len(index))
    t1_idx = locs + t1_bars
    t1_idx[t1_idx >= len(index)] = len(index) - 1
    return pd.Series(index[t1_idx], index=index)

def triple_barrier_labels(prices: pd.Series,
                          pt_mult=2.0, sl_mult=1.0,
                          t1_bars=24, vol_ewm_span=48):
    """
    Returns DataFrame with columns:
      t1: timeout timestamp
      label: {-1,0,1}
      hit: {'pt','sl','t1'} which barrier hit
      w: sample weight (|ret| / std) simple proxy
    """
    vol = ewm_volatility(prices, span=vol_ewm_span).replace(0, np.nan).ffill()
    t1 = get_barrier_times(prices.index, t1_bars)
    out = pd.DataFrame(index=prices.index, data={'t1': t1})

    pt = prices * (1 + pt_mult * vol)
    sl = prices * (1 - sl_mult * vol)

    labels, hit = [], []
    for t0 in prices.index:
        p0 = prices.loc[t0]
        t_expire = out.at[t0, 't1']
        if pd.isna(t_expire):  # end of sample
            labels.append(0); hit.append('t1'); continue
        path = prices.loc[t0:t_expire]
        up = (path >= pt.loc[t0]).idxmax() if (path >= pt.loc[t0]).any() else None
        dn = (path <= sl.loc[t0]).idxmax() if (path <= sl.loc[t0]).any() else None

        if up and dn:
            winner = up if up < dn else dn
            if winner == up: labels.append(1); hit.append('pt')
            else:            labels.append(-1); hit.append('sl')
        elif up:
            labels.append(1); hit.append('pt')
        elif dn:
            labels.append(-1); hit.append('sl')
        else:
            # timeout -> use sign of return at t1 (weak)
            r = prices.loc[t_expire] / p0 - 1
            labels.append(np.sign(r) if r != 0 else 0); hit.append('t1')

    out['label'] = pd.Series(labels, index=prices.index).astype(int)
    out['hit'] = hit
    # simple weights: absolute return to t1 normalized by vol (avoid zero)
    r_to_t1 = prices.reindex(out['t1']).values / prices.values - 1.0
    out['w'] = np.abs(r_to_t1) / (vol.values + 1e-12)
    out['w'] = np.clip(out['w'].replace([np.inf, -np.inf], 0).fillna(0), 0, out['w'].quantile(0.99))
    return out
