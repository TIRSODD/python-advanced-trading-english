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
       
