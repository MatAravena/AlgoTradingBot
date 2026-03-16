import os
import math
import logging
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
# from alpaca.stream import Stream
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from config import (
    RSI_PERIOD, MACD_FAST_EMA, MACD_SLOW_EMA, MACD_SIGNAL_EMA,
    BIG_TREND_FAST_MA, BIG_TREND_SLOW_MA, MAX_RISK_PERCENT,
    STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT, TIMEFRAME_MAIN, TIMEFRAME_TREND,
    WINDOW_SIZE, UNDERLYING_SYMBOL, ATR_PERIOD, BB_LENGTH, BB_STDDEV, EMA_PERIOD)

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# === Configuration ===
api_key = os.getenv("API_KEY_VAR")
api_secret = os.getenv("API_SECRET_VAR")
paper_trading = os.getenv("PAPER_FLAG_VAR") == "True"
base_url = os.getenv("TRADE_API_URL_VAR")

symbol = UNDERLYING_SYMBOL
rsi_period = RSI_PERIOD
macd_fast = MACD_FAST_EMA
macd_slow = MACD_SLOW_EMA
macd_signal = MACD_SIGNAL_EMA
htf_fast_ma = BIG_TREND_FAST_MA
htf_slow_ma = BIG_TREND_SLOW_MA
max_risk_pct = MAX_RISK_PERCENT
stop_loss_pct = STOP_LOSS_PERCENT
take_profit_pct = TAKE_PROFIT_PERCENT
timeframe_main = TIMEFRAME_MAIN
timeframe_trend = TIMEFRAME_TREND
window_size = WINDOW_SIZE

bb_length = BB_LENGTH
bb_stddev = BB_STDDEV
ema_period = EMA_PERIOD
atr_period = ATR_PERIOD

# === Clients ===
trading_client = TradingClient(api_key, api_secret, paper=paper_trading, base_url=base_url)
data_client = StockHistoricalDataClient(api_key, api_secret)

def fetch_bars(symbol, timeframe, limit):
    logger.debug(f"Fetching {limit} bars for {symbol} at timeframe {timeframe}")
    req = StockBarsRequest(symbols=[symbol], timeframe=timeframe, limit=limit)
    df = data_client.get_stock_bars(req).df
    bars = df.xs(symbol, level=0)
    logger.debug(f"Fetched {len(bars)} bars")
    return bars

def get_trend():
    bars = fetch_bars(symbol, timeframe_trend, limit=htf_slow_ma + window_size)
    close = bars['close']
    ma_fast = close.rolling(htf_fast_ma).mean().iloc[-1]
    ma_slow = close.rolling(htf_slow_ma).mean().iloc[-1]
    if ma_fast > ma_slow:
        logger.info("Higher timeframe trend: BULL")
        return "BULL"
    if ma_fast < ma_slow:
        logger.info("Higher timeframe trend: BEAR")
        return "BEAR"
    logger.info("Higher timeframe trend: NEUTRAL")
    return "NEUTRAL"

def compute_indicators(bars):
    close = bars['close']; high = bars['high']; low = bars['low']
    rsi = RSIIndicator(close, rsi_period).rsi()
    macd = MACD(close, macd_fast, macd_slow, macd_signal)
    macd_line = macd.macd(); macd_sig = macd.macd_signal()
    bb = BollingerBands(close, BB_LENGTH, BB_STDDEV)
    bb_upper = bb.bollinger_hband(); bb_lower = bb.bollinger_lband()
    ema = EMAIndicator(close, EMA_PERIOD).ema_indicator()
    atr = AverageTrueRange(high, low, close, ATR_PERIOD).average_true_range()
    logger.debug(f"Indicators - RSI: {rsi.iloc[-1]}, MACD: {macd_line.iloc[-1]}/{macd_sig.iloc[-1]}, "
                 f"BB: {bb_upper.iloc[-1]}/{bb_lower.iloc[-1]}, EMA: {ema.iloc[-1]}, ATR: {atr.iloc[-1]}")
    return rsi, macd_line, macd_sig, bb_upper, bb_lower, ema, atr

def crossed_up(s1, s2):
    diff = s1 - s2
    cross = (diff.shift(1) < 0) & (diff > 0)
    result = cross.iloc[-window_size:].any()
    if result:
        logger.debug("Crossed up detected")
    return result

def crossed_down(s1, s2):
    diff = s1 - s2
    cross = (diff.shift(1) > 0) & (diff < 0)
    result = cross.iloc[-window_size:].any()
    if result:
        logger.debug("Crossed down detected")
    return result

account = trading_client.get_account()
max_risk_amount = float(account.equity) * max_risk_pct
logger.info(f"Account equity: {account.equity}, max risk per trade: {max_risk_amount}")

# CryptoDataStream
stream = CryptoDataStream(api_key, api_secret, base_url=base_url, data_stream="sip")
stream = StockDataStream(api_key, api_secret, base_url=base_url, data_stream="sip")
stream = OptionDataStream(api_key, api_secret, base_url=base_url, data_stream="sip")

@stream.on_bar(symbol)
async def on_bar(bar):
    logger.info(f"New bar: {bar.symbol} {bar.timestamp} close={bar.close}")
    bars = fetch_bars(
        symbol,
        timeframe_main,
        limit=max(rsi_period, macd_slow, BB_LENGTH, EMA_PERIOD, ATR_PERIOD) + window_size
    )
    rsi_s, macd_line_s, macd_sig_s, bb_up_s, bb_lo_s, ema_s, atr_s = compute_indicators(bars)
    trend = get_trend()
    price = bar.close

    if trend == "BULL":
        cond1 = crossed_up(macd_line_s, macd_sig_s) and rsi_s.iloc[-1] < 30
        cond2 = price > bb_up_s.iloc[-1] and price > ema_s.iloc[-1] and atr_s.iloc[-1] > atr_s.iloc[-window_size:].mean()
        if cond1 and cond2:
            entry = price
            stop = entry * (1 - stop_loss_pct)
            target = entry * (1 + take_profit_pct)
            qty = math.floor(max_risk_amount / (entry - stop))
            logger.info(f"Placing LONG order: qty={qty}, entry={entry}, stop={stop}, target={target}")
            trading_client.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                type="limit",
                qty=qty,
                time_in_force=TimeInForce.GTC,
                limit_price=str(entry),
                order_class=OrderClass.BRACKET,
                stop_loss={'stop_price': str(stop)},
                take_profit={'limit_price': str(target)},
            )

    elif trend == "BEAR":
        cond1 = crossed_down(macd_line_s, macd_sig_s) and rsi_s.iloc[-1] > 70
        cond2 = price < bb_lo_s.iloc[-1] and price < ema_s.iloc[-1] and atr_s.iloc[-1] > atr_s.iloc[-window_size:].mean()
        if cond1 and cond2:
            entry = price
            stop = entry * (1 + stop_loss_pct)
            target = entry * (1 - take_profit_pct)
            qty = math.floor(max_risk_amount / (stop - entry))
            logger.info(f"Placing SHORT order: qty={qty}, entry={entry}, stop={stop}, target={target}")
            trading_client.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                type="limit",
                qty=qty,
                time_in_force=TimeInForce.GTC,
                limit_price=str(entry),
                order_class=OrderClass.BRACKET,
                stop_loss={'stop_price': str(stop)},
                take_profit={'limit_price': str(target)},
            )

    await manage_positions()

async def manage_positions():
    positions = trading_client.get_all_positions()
    for pos in positions:
        entry_price = float(pos.avg_entry_price)
        qty = float(pos.qty)
        current_price = float(pos.market_value) / qty
        profit_pct = ((current_price - entry_price) / entry_price) if pos.side == 'long' else ((entry_price - current_price) / entry_price)
        logger.info(f"Managing position {pos.symbol} side={pos.side} qty={pos.qty} unrealized_profit_pct={profit_pct:.2%}")
        if profit_pct > 0.02:
            logger.info(f"Would adjust stop to breakeven for {pos.symbol}")
            # Implement stop adjustment via trading_client.replace_order(...) as needed

if __name__ == "__main__":
    logger.info("Starting trading bot stream...")
    stream.run()
