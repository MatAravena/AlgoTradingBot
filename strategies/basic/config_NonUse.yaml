RSI_PERIOD = 14
MACD_FAST_EMA = 12
MACD_SLOW_EMA = 26
MACD_SIGNAL_EMA = 9
BIG_TREND_FAST_MA = 20
BIG_TREND_SLOW_MA = 50
MAX_RISK_PERCENT = 0.02
STOP_LOSS_PERCENT = 0.02
TAKE_PROFIT_PERCENT = 0.04
TIMEFRAME_MAIN = TimeFrame.Minute
TIMEFRAME_TREND = TimeFrame.Hour
WINDOW_SIZE = 20
UNDERLYING_SYMBOL = ["BTC", "ETH", "SOL", "ADA", "XRP"]

BB_LENGTH = 20
BB_STDDEV = 2.0
EMA_PERIOD = 20
ATR_PERIOD = 14



# Sensible defaults to get moving
# Asset/timeframe

# BTCUSDT @ 1h bars; US equities @ 30m–1h bars.

# Labeling

# pt_mult=2.0, sl_mult=1.0, t1_bars=24 (BTC 1d) or 13 (equities). EWM vol span = 48 bars. Then meta-label with threshold tuned via CV. machine-learning-for-as…

# Features (initial small set)

# r_t, EWM vol, EMA(20/50/100), MACD(12,26,9), Stoch(14,3,3), rolling RS vs benchmark; plus 1–3 microstructure/volume features if available. Keep it lean to avoid paralysis. trading for dummies for…

# Validation

# Purged K=5 folds, embargo=1 horizon; for dense events use CPCV with small k/N. machine-learning-for-as…

# Model

# Start with balanced RandomForest/GradientBoosting for calibrated class probas (for meta-label + PWA scoring). Use MDA to prune features. machine-learning-for-as…

# Sizing & exits

# Size = f(proba, cleaned vol). Initial stop = 1·σ (or 1·ATR), take-profit = 2·σ, trailing stop at 1–1.5·σ once in profit. trading for dummies for…

# Overfit checks

# Report DSR and FWER-adjusted significance for best fold; run MC resamples for stability. machine-learning-for-as…

