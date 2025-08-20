import os, time
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timezone
from research.features import basic_features
from research.labels import triple_barrier_labels
from research.model import fit_model
from research.importance import mda_importance
from live.sizing import prob_vol_sizer
from live.risk import DailyRiskState, update_pnl
from live.execution import make_client, place_entry_with_stop

# ---- Replace with your own market data fetchers ----
def fetch_ohlcv(symbol: str, timeframe: str, lookback_bars: int) -> pd.DataFrame:
    """
    Return DataFrame with ['open','high','low','close','volume'] indexed by UTC timestamp.
    Plug in your existing data client here.
    """
    raise NotImplementedError

def mid_price(row):
    return (row['high'] + row['low']) / 2.0

def train_signal_model(df: pd.DataFrame, cfg):
    px = df['close']
    feats = basic_features(df)
    lbls = triple_barrier_labels(px, cfg['labels']['pt_mult'], cfg['labels']['sl_mult'],
                                 cfg['labels']['t1_bars'], cfg['labels']['vol_ewm_span'])
    X = feats.align(lbls['label'], join='inner')[0]
    y = lbls['label'].reindex(X.index)
    # weights:
    w = lbls['w'].reindex(X.index)
    model, cv_scores = fit_model(
        X, y, sample_weight=w,
        n_splits=cfg['cv']['n_splits'],
        embargo_pct=cfg['cv']['embargo_pct'],
        n_estimators=cfg['model']['n_estimators'],
        max_depth=cfg['model']['max_depth'],
        min_samples_leaf=cfg['model']['min_samples_leaf'],
        class_weight=cfg['model']['class_weight']
    )
    _, imp = mda_importance(model, X, y, scorer='pwa', n_rounds=3)
    # Optional: drop weak features and refit
    weak = imp[imp['importance'] <= 0]['feature'].tolist()
    if weak:
        X2 = X.drop(columns=weak)
        model, _ = fit_model(X2, y, sample_weight=w,
                             n_splits=cfg['cv']['n_splits'],
                             embargo_pct=cfg['cv']['embargo_pct'],
                             n_estimators=cfg['model']['n_estimators'],
                             max_depth=cfg['model']['max_depth'],
                             min_samples_leaf=cfg['model']['min_samples_leaf'],
                             class_weight=cfg['model']['class_weight'])
        return model, X2.columns.tolist()
    return model, X.columns.tolist()

def run_symbol(symbol_cfg, cfg, broker):
    sym = symbol_cfg['symbol']
    tf  = cfg['data']['timeframe']
    n   = cfg['data']['lookback_bars']
    df  = fetch_ohlcv(sym, tf, n)
    model, feat_cols = train_signal_model(df, cfg)

    # Live decision on the most recent bar close
    feats = basic_features(df)[feat_cols].tail(1)
    proba_long = model.predict_proba(feats)[:,1][0]
    last = df.iloc[-1]
    vol = df['close'].pct_change().ewm(span=cfg['labels']['vol_ewm_span'], adjust=False).std().iloc[-1]
    px  = float(last['close'])

    if proba_long >= cfg['trade']['proba_entry_threshold']:
        # long entry example
        size_frac = prob_vol_sizer(proba_long, vol, cfg['trade']['max_risk_per_trade'])
        # Convert risk to quantity (very simplified; use your equity & risk model)
        equity = 100000  # TODO: pull from broker/account
        notional = equity * size_frac
        qty = max(notional / px, 0.0001)
        # stops using σ
        sl = px * (1 - cfg['labels']['sl_mult'] * vol)
        tp = px * (1 + cfg['labels']['pt_mult'] * vol)
        place_entry_with_stop(broker, sym, qty, side='buy',
                              limit_price=None, stop_price=sl, tif=cfg['trade']['tif'])
        print(f"[{sym}] BUY qty={qty:.6f} px={px:.2f} sl={sl:.2f} tp={tp:.2f} p={proba_long:.3f}")
    else:
        print(f"[{sym}] No trade. p_long={proba_long:.3f}")

def main():
    with open('config.yaml','r') as f:
        cfg = yaml.safe_load(f)
    broker = make_client(cfg['alpaca']['key_id'], cfg['alpaca']['secret'], cfg['alpaca']['paper'])
    # Daily risk state (stub)
    drs = DailyRiskState(day=datetime.utcnow().date())
    for sym_cfg in cfg['symbols']:
        run_symbol(sym_cfg, cfg, broker)

if __name__ == "__main__":
    main()
