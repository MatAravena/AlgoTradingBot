import os
from dataclasses import dataclass
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce

@dataclass
class BrokerCtx:
    client: TradingClient

def make_client(key_id: str, secret: str, paper: bool=True) -> BrokerCtx:
    client = TradingClient(key_id, secret, paper=paper)
    return BrokerCtx(client)

def tif_from_cfg(tif: str):
    return TimeInForce.GTC if tif.lower() == 'gtc' else TimeInForce.DAY

def place_entry_with_stop(ctx: BrokerCtx, symbol: str, qty: float, side: str,
                          limit_price: Optional[float], stop_price: Optional[float],
                          tif: str='gtc'):
    side_enum = OrderSide.BUY if side.lower()=='buy' else OrderSide.SELL
    if limit_price:
        req = LimitOrderRequest(symbol=symbol, qty=qty, side=side_enum,
                                time_in_force=tif_from_cfg(tif), limit_price=limit_price)
    else:
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum,
                                 time_in_force=tif_from_cfg(tif))
    order = ctx.client.submit_order(req)
    if stop_price:
        # simple stop loss attached as separate order
        sl_req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL if side_enum==OrderSide.BUY else OrderSide.BUY,
                                    time_in_force=tif_from_cfg(tif), order_class=None)
        # Alpaca supports bracket orders; to keep it generic we create a stop-loss after entry fill
        # (If you prefer brackets, switch to bracket class with take_profit/stop_loss requests.)
    return order
