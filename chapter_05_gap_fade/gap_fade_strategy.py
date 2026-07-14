"""
Chapter 5: Gap Fade Strategy
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

The Gap Fade strategy is a mean-reversion approach that bets on gaps being filled.
The logic is opposite to ORB: instead of following the gap direction, we trade 
against it, expecting the price to return to the previous close.

Logic:
- Detect a significant gap at market open
- If gap is UP (price opens much higher than yesterday's close) -> SELL
- If gap is DOWN (price opens much lower than yesterday's close) -> BUY
- Target: previous close (gap fill)
- Stop loss: beyond the gap extreme (if gap keeps expanding, we're wrong)

This strategy works best in ranging markets and fails in strong trending markets.
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, List, Tuple


class GapFadeStrategy:
    """
    Gap Fade (Mean Reversion) Strategy implementation.
    """
    
    def __init__(self, df: pd.DataFrame,
                 min_gap_pct: float = 0.3,
                 max_gap_pct: float = 2.0,
                 profit_target: str = 'previous_close',
                 stop_loss_multiplier: float = 1.5,
                 open_time: time = time(9, 30)):
        """
        Initialize the Gap Fade strategy.
        
        Args:
            df: DataFrame with intraday OHLCV data
            min_gap_pct: Minimum gap percentage to trade (default: 0.3%)
            max_gap_pct: Maximum gap percentage to trade (default: 2.0%)
            profit_target: 'previous_close' or fixed percentage
            stop_loss_multiplier: Multiple of gap size for stop loss
            open_time: Market open time
        """
        self.df = df.copy()
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.profit_target = profit_target
        self.stop_loss_multiplier = stop_loss_multiplier
        self.open_time = open_time
        
        self.trades = []
    
    def detect_gap(self) -> Tuple[float, float, str]:
        """
        Detect if there's a tradeable gap at market open.
        
        Returns:
            Tuple with (gap_size, gap_pct, gap_type)
            gap_type: 'up', 'down', or 'none'
        """
        # Get opening price and previous close
        open_price = self.df.iloc[0]['open']
        
        # Previous close is the last close before the current session
        # For simplicity, we'll use a value passed or calculated
        # In real implementation, this would come from daily data
        prev_close = self.df.iloc[0]['close'] * 0.998  # Simulated previous close
        
        gap_size = open_price - prev_close
        gap_pct = (gap_size / prev_close) * 100
        
        # Determine gap type
        if gap_pct >= self.min_gap_pct and gap_pct <= self.max_gap_pct:
            gap_type = 'up'
        elif gap_pct <= -self.min_gap_pct and gap_pct >= -self.max_gap_pct:
            gap_type = 'down'
        else:
            gap_type = 'none'
        
        print(f"Gap Detection:")
        print(f"  Open Price:    {open_price:.5f}")
        print(f"  Previous Close: {prev_close:.5f}")
        print(f"  Gap Size:      {gap_size:.5f} ({gap_pct:+.2f}%)")
        print(f"  Gap Type:      {gap_type.upper()}")
        
        return gap_size, gap_pct, gap_type
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on gap fade logic.
        
        Returns:
            DataFrame with added 'signal' column
        """
        gap_size, gap_pct, gap_type = self.detect_gap()
        
        df = self.df.copy()
        df['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        if gap_type == 'up':
            # Gap up -> Sell (fade the gap)
            df.iloc[0, df.columns.get_loc('signal')] = -1
            print(f"\nSignal: SELL (fading the up gap)")
        
        elif gap_type == 'down':
            # Gap down -> Buy (fade the gap)
            df.iloc[0, df.columns.get_loc('signal')] = 1
            print(f"\nSignal: BUY (fading the down gap)")
        
        else:
            print(f"\nNo tradeable gap detected. No signal.")
        
        return df
    
    def calculate_trade_levels(self, signal_type: int,
                               entry_price: float,
                               prev_close: float,
                               gap_size: float) -> Dict[str, float]:
        """
        Calculate entry, stop loss, and take profit levels.
        
        Args:
            signal_type: 1 for buy, -1 for sell
            entry_price: Price at which the trade is entered
            prev_close: Previous day's close (target for gap fill)
            gap_size: Size of the gap
        
        Returns:
            Dictionary with 'entry', 'stop_loss', 'take_profit'
        """
        if signal_type == 1:  # Buy (fading down gap)
            take_profit = prev_close  # Target: gap fill
            stop_loss = entry_price - (abs(gap_size) * self.stop_loss_multiplier)
        else:  # Sell (fading up gap)
            take_profit = prev_close  # Target: gap fill
            stop_loss = entry_price + (abs(gap_size) * self.stop_loss_multiplier)
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk': risk,
            'reward': reward,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def backtest_simple(self) -> Dict:
        """
        Simple backtest: simulate one trade based on gap fade signal.
        
        Returns:
            Dictionary with backtest results
        """
        df = self.generate_signals()
        
        if df.iloc[0]['signal'] == 0:
            return {'total_trades': 0, 'reason': 'no_tradeable_gap'}
        
        # Get trade parameters
        signal_type = df.iloc[0]['signal']
        entry_price = df.iloc[0]['close']
        prev_close = df.iloc[0]['close'] * 0.998  # Simulated
        gap_size = entry_price - prev_close
        
        levels = self.calculate_trade_levels(signal_type, entry_price, prev_close, gap_size)
        
        print(f"\nTrade Setup:")
        print(f"  Entry:        {levels['entry']:.5f}")
        print(f"  Stop Loss:    {levels['stop_loss']:.5f}")
        print(f"  Take Profit:  {levels['take_profit']:.5f}")
        print(f"  Risk/Reward:  {levels['risk_reward_ratio']:.2f}")
        
        # Simulate trade outcome (simplified)
        # In real backtest, we'd iterate through the data
        trade_result = {
            'total_trades': 1,
            'direction': 'BUY' if signal_type == 1 else 'SELL',
            'entry': levels['entry'],
            'stop_loss': levels['stop_loss'],
            'take_profit': levels['take_profit'],
            'risk_reward_ratio': levels['risk_reward_ratio']
        }
        
        return trade_result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic intraday data with a gap up
    np.random.seed(42)
    n = 390  # 6.5 hours of 1-minute data
    
    dates = pd.date_range('2025-07-14 09:30', periods=n, freq='1min')
    
    # Simulate a gap up scenario
    prev_close = 1.10000
    open_price = 1.10500  # Gap up of 0.5%
    
    # Price fades the gap (returns to previous close)
    prices = open_price - np.cumsum(np.random.randn(n) * 0.0001 - 0.00005)
    
    volume = np.random.randint(1000, 5000, n)
    volume[:15] = np.random.randint(10000, 20000, 15)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.0003),
        'low': prices - np.abs(np.random.randn(n) * 0.0003),
        'close': prices + np.random.randn(n) * 0.0001,
        'volume': volume
    }, index=dates)
    
    print("=" * 60)
    print("CHAPTER 5: GAP FADE STRATEGY DEMO")
    print("=" * 60)
    
    # Initialize strategy
    gap_fade = GapFadeStrategy(
        df,
        min_gap_pct=0.3,
        max_gap_pct=2.0,
        stop_loss_multiplier=1.5
    )
    
    # Run backtest
    results = gap_fade.backtest_simple()
