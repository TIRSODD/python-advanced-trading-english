"""
Chapter 3: Market Open Analyzer
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

The market open is the most volatile and liquid period of the trading day.
Understanding its anatomy is crucial for gap trading strategies.

This module analyzes the first N minutes of the session to identify:
- Opening Range (high/low of the first N minutes)
- Initial volatility (ATR of the open)
- Volume spikes compared to the daily average
- Directional bias (bullish/bearish open)
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, Tuple


class MarketOpenAnalyzer:
    """
    Analyze the anatomy of the market opening session.
    """
    
    def __init__(self, df: pd.DataFrame, open_time: time = time(9, 30)):
        """
        Initialize the analyzer.
        
        Args:
            df: DataFrame with OHLCV data (must have a DatetimeIndex)
            open_time: Market open time (default: 9:30 AM for US markets)
        """
        self.df = df.copy()
        self.open_time = open_time
        self.opening_stats = {}
    
    def analyze_opening_range(self, minutes: int = 15) -> Dict[str, float]:
        """
        Calculate the Opening Range (high and low of the first N minutes).
        
        The Opening Range is a key concept in ORB (Opening Range Breakout) strategies.
        
        Args:
            minutes: Number of minutes to consider for the opening range
        
        Returns:
            Dictionary with 'open_high', 'open_low', 'open_range'
        """
        # Filter data for the opening period
        # Assuming df is sorted by time and we take the first 'minutes' rows of each day
        # For simplicity, we'll analyze the global first N minutes
        
        opening_data = self.df.head(minutes)
        
        if len(opening_data) == 0:
            return {'open_high': np.nan, 'open_low': np.nan, 'open_range': np.nan}
        
        open_high = opening_data['high'].max()
        open_low = opening_data['low'].min()
        open_range = open_high - open_low
        
        self.opening_stats['opening_range'] = {
            'open_high': open_high,
            'open_low': open_low,
            'open_range': open_range,
            'minutes': minutes
        }
        
        return self.opening_stats['opening_range']
    
    def calculate_opening_volatility(self, minutes: int = 15) -> float:
        """
        Calculate the volatility during the opening period using ATR (Average True Range).
        
        Args:
            minutes: Number of minutes to consider
        
        Returns:
            Average True Range (ATR) for the opening period
        """
        opening_data = self.df.head(minutes)
        
        if len(opening_data) < 2:
            return 0.0
        
        high = opening_data['high']
        low = opening_data['low']
        close = opening_data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.mean()
        
        self.opening_stats['opening_volatility'] = atr
        
        return atr
    
    def detect_volume_spike(self, minutes: int = 15, threshold: float = 2.0) -> bool:
        """
        Detect if the opening volume is significantly higher than average.
        
        Args:
            minutes: Number of minutes to consider for opening volume
            threshold: Multiplier for average volume to consider a spike (default: 2x)
        
        Returns:
            True if volume spike detected, False otherwise
        """
        if 'volume' not in self.df.columns:
            return False
        
        opening_volume = self.df.head(minutes)['volume'].sum()
        avg_volume = self.df['volume'].mean() * minutes
        
        if avg_volume == 0:
            return False
        
        volume_ratio = opening_volume / avg_volume
        is_spike = volume_ratio > threshold
        
        self.opening_stats['volume_spike'] = {
            'opening_volume': opening_volume,
            'avg_volume': avg_volume,
            'ratio': volume_ratio,
            'is_spike': is_spike
        }
        
        return is_spike
    
    def determine_opening_bias(self, minutes: int = 15) -> str:
        """
        Determine if the opening is bullish or bearish.
        
        Args:
            minutes: Number of minutes to consider
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        opening_data = self.df.head(minutes)
        
        if len(opening_data) == 0:
            return 'neutral'
        
        open_price = opening_data.iloc[0]['open']
        close_price = opening_data.iloc[-1]['close']
        
        change_pct = (close_price - open_price) / open_price * 100
        
        if change_pct > 0.1:
            bias = 'bullish'
        elif change_pct < -0.1:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        self.opening_stats['opening_bias'] = {
            'bias': bias,
            'change_pct': change_pct
        }
        
        return bias
    
    def get_full_analysis(self, minutes: int = 15) -> Dict:
        """
        Run all opening analyses and return a comprehensive report.
        
        Args:
            minutes: Number of minutes to consider
        
        Returns:
            Dictionary with all opening statistics
        """
        print(f"Analyzing first {minutes} minutes of market open...")
        print("-" * 60)
        
        opening_range = self.analyze_opening_range(minutes)
        print(f"Opening Range: {opening_range['open_low']:.5f} - {opening_range['open_high']:.5f}")
        print(f"Range Size: {opening_range['open_range']:.5f}")
        
        volatility = self.calculate_opening_volatility(minutes)
        print(f"Opening Volatility (ATR): {volatility:.5f}")
        
        has_spike = self.detect_volume_spike(minutes)
        if 'volume_spike' in self.opening_stats:
            vol_data = self.opening_stats['volume_spike']
            print(f"Volume Ratio: {vol_data['ratio']:.2f}x {'(SPIKE!)' if vol_data['is_spike'] else ''}")
        
        bias = self.determine_opening_bias(minutes)
        bias_data = self.opening_stats['opening_bias']
        print(f"Opening Bias: {bias.upper()} ({bias_data['change_pct']:+.2f}%)")
        
        return self.opening_stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic data simulating a market open
    np.random.seed(42)
    n = 100  # 100 minutes of data
    
    dates = pd.date_range('2025-07-14 09:30', periods=n, freq='1min')
    
    # Simulate a bullish opening with high volume
    base_price = 1.10000
    prices = base_price + np.cumsum(np.random.randn(n) * 0.0003 + 0.0001)  # slight upward drift
    
    volume = np.random.randint(1000, 5000, n)
    volume[:15] = np.random.randint(10000, 20000, 15)  # High volume in first 15 mins
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.0002),
        'low': prices - np.abs(np.random.randn(n) * 0.0002),
        'close': prices + np.random.randn(n) * 0.0001,
        'volume': volume
    }, index=dates)
    
    print("=" * 60)
    print("CHAPTER 3: MARKET OPEN ANALYZER DEMO")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MarketOpenAnalyzer(df)
    
    # Run full analysis
    stats = analyzer.get_full_analysis(minutes=15)
