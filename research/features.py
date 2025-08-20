import numpy as np
import pandas as pd

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'macd_signal': signal_line, 'macd_hist': hist})

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3):
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    k_fast = 100 * (close - lowest) / (highest - lowest + 1e-12)
    d_slow = k_fast.rolling(d, min_periods=d).mean()
    return pd.DataFrame({'stoch_k': k_fast, 'stoch_d': d_slow})

def basic_features(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    df_ohlc: columns ['open','high','low','close','volume']
    """
    px = df_ohlc['close']
    rets = px.pct_change()
    vol = rets.ewm(span=48, adjust=False).std()

    feats = pd.DataFrame(index=df_ohlc.index)
    feats['ret_1']      = rets
    feats['ret_5']      = px.pct_change(5)
    feats['ewm_vol']    = vol
    feats['ema20']      = ema(px, 20)
    feats['ema50']      = ema(px, 50)
    feats['ema100']     = ema(px, 100)
    macd_df = macd(px)
    stoch_df = stochastic(df_ohlc['high'], df_ohlc['low'], px)

    feats = pd.concat([feats, macd_df, stoch_df], axis=1)
    # rate of change of vol helps regime awareness
    feats['vol_roc'] = feats['ewm_vol'].pct_change()
    return feats.replace([np.inf, -np.inf], np.nan).ffill().bfill()
