from dataclasses import dataclass
from datetime import date

@dataclass
class DailyRiskState:
    day: date
    pnl: float = 0.0
    halted: bool = False

def should_halt(drs: DailyRiskState, max_daily_loss: float) -> bool:
    return drs.pnl <= -abs(max_daily_loss)

def update_pnl(drs: DailyRiskState, delta: float, max_daily_loss: float) -> DailyRiskState:
    drs.pnl += delta
    drs.halted = should_halt(drs, max_daily_loss)
    return drs
