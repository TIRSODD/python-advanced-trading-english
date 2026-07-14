"""
Chapter 4: Opening Range Breakout (ORB) Strategy
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

The Opening Range Breakout (ORB) is one of the most popular intraday strategies.
The idea is simple: the first N minutes of the session define a "range" (high/low).
When the price breaks above or below this range, it signals a potential trend for the day.

Logic:
- Define the Opening Range (first N minutes, typically 15 or 30)
- Buy if price breaks above the range high
- Sell if price breaks below the range low
- Stop loss at the opposite side of the range
- Take profit at a multiple of the range size (e.g., 2x or 3x)
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, List, Tuple


class ORBStrategy:
    """
    Opening Range Breakout (ORB) Strategy implementation.
    """
    
    def __init__(self, df: pd.DataFrame, 
                 opening_minutes: int = 15,
                 profit_target_multiplier: float = 2.0,
                 open_time: time = time(9, 30)):
        """
        Initialize the ORB strategy.
        
        Args:
            df: DataFrame with intraday OHLCV data
            opening_minutes: Minutes to define the opening range (default: 15)
            profit_target_multiplier: Multiple of range size for take profit (default: 2.0)
            open_time: Market open time (default: 9:30 AM)
        """
        self.df = df.copy()
        self.opening_minutes = opening_minutes
        self.profit_target_multiplier = profit_target_multiplier
        self.open_time = open_time
        self.signals = []
    
    def calculate_opening_range(self) -> Tuple[float, float, float]:
        """
        Calculate the Opening Range (high, low, and size).
        
        Returns:
            Tuple with (range_high, range_low, range_size)
        """
        opening_data = self.df.head(self.opening_minutes)
        
        range_high = opening_data['high'].max()
        range_low = opening_data['low'].min()
        range_size = range_high - range_low
        
        print(f"Opening Range ({self.opening_minutes} min):")
        print(f"  High: {range_high:.5f}")
        print(f"  Low:  {range_low:.5f}")
        print(f"  Size: {range_size:.5f}")
        
        return range_high, range_low, range_size
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on ORB breakouts.
        
        Returns:
            DataFrame with added 'signal' column
        """
        range_high, range_low, range_size = self.calculate_opening_range()
        
        df = self.df.copy()
        df['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Only consider data after the opening range
        df.loc[df.index[self.opening_minutes]:, 'signal'] = 0
        
        # Buy signal: price breaks above range high
        buy_condition = df['close'] > range_high
        df.loc[buy_condition & (df.index >= df.index[self.opening_minutes]), 'signal'] = 1
        
        # Sell signal: price breaks below range low
        sell_condition = df['close'] < range_low
        df.loc[sell_condition & (df.index >= df.index[self.opening_minutes]), 'signal'] = -1
        
        # Count signals
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        
        print(f"\nSignals generated:")
        print(f"  Buy signals:  {buy_signals}")
        print(f"  Sell signals: {sell_signals}")
        
        self.signals = df[df['signal'] != 0]
        
        return df
    
    def calculate_trade_levels(self, signal_type: int, 
                               entry_price: float,
                               range_high: float,
                               range_low: float,
                               range_size: float) -> Dict[str, float]:
        """
        Calculate entry, stop loss, and take profit levels for a trade.
        
        Args:
            signal_type: 1 for buy, -1 for sell
            entry_price: Price at which the trade is entered
            range_high: Opening range high
            range_low: Opening range low
            range_size: Size of the opening range
        
        Returns:
            Dictionary with 'entry', 'stop_loss', 'take_profit'
        """
        if signal_type == 1:  # Buy
            stop_loss = range_low
            take_profit = entry_price + (range_size * self.profit_target_multiplier)
        else:  # Sell
            stop_loss = range_high
            take_profit = entry_price - (range_size * self.profit_target_multiplier)
        
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
        Simple backtest: simulate trades based on generated signals.
        
        Returns:
            Dictionary with backtest results
        """
        df = self.generate_signals()
        range_high, range_low, range_size = self.calculate_opening_range()
        
        trades = []
        
        for idx, row in df[df['signal'] != 0].iterrows():
            levels = self.calculate_trade_levels(
                row['signal'], row['close'], range_high, range_low, range_size
            )
            trades.append({
                'timestamp': idx,
                'direction': 'BUY' if row['signal'] == 1 else 'SELL',
                **levels
            })
        
        # Calculate simple statistics
        if not trades:
            return {'total_trades': 0}
        
        results = {
            'total_trades': len(trades),
            'buy_trades': sum(1 for t in trades if t['direction'] == 'BUY'),
            'sell_trades': sum(1 for t in trades if t['direction'] == 'SELL'),
            'avg_risk_reward': np.mean([t['risk_reward_ratio'] for t in trades])
        }
        
        print(f"\nBacktest Results:")
        print(f"  Total trades:    {results['total_trades']}")
        print(f"  Buy trades:      {results['buy_trades']}")
        print(f"  Sell trades:     {results['sell_trades']}")
        print(f"  Avg Risk/Reward: {results['avg_risk_reward']:.2f}")
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic intraday data
    np.random.seed(42)
    n = 390  # 6.5 hours of 1-minute data (typical US session)
    
    dates = pd.date_range('2025-07-14 09:30', periods=n, freq='1min')
    
    # Simulate price movement with a breakout after 30 minutes
    base_price = 1.10000
    prices = base_price + np.cumsum(np.random.randn(n) * 0.0002)
    
    # Add a breakout move after minute 30
    prices[30:] += 0.005  # Upward breakout
    
    volume = np.random.randint(1000, 5000, n)
    volume[:15] = np.random.randint(10000, 20000, 15)  # High opening volume
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.0003),
        'low': prices - np.abs(np.random.randn(n) * 0.0003),
        'close': prices + np.random.randn(n) * 0.0001,
        'volume': volume
    }, index=dates)
    
    print("=" * 60)
    print("CHAPTER 4: ORB STRATEGY DEMO")
    print("=" * 60)
    
    # Initialize strategy
    orb = ORBStrategy(df, opening_minutes=15, profit_target_multiplier=2.0)
    
    # Run backtest
    results = orb.backtest_simple()
