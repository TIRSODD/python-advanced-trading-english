"""
Chapter 3: Gap Detector and Classifier
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

A gap occurs when the opening price of an asset is significantly different 
from the previous day's closing price, leaving a "gap" in the price chart 
where no trading occurred.

This module detects gaps and classifies them based on their size relative 
to the asset's recent volatility (ATR).
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class GapDetector:
    """
    Detect and classify price gaps in OHLCV data.
    """
    
    def __init__(self, df: pd.DataFrame, atr_period: int = 14):
        """
        Initialize the detector.
        
        Args:
            df: DataFrame with daily or intraday OHLCV data
            atr_period: Period for Average True Range calculation (for gap sizing)
        """
        self.df = df.copy()
        self.atr_period = atr_period
        self.gaps = []
        
        # Pre-calculate ATR for gap classification
        self._calculate_atr()
    
    def _calculate_atr(self):
        """Calculate Average True Range (ATR) for volatility-based gap sizing."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=self.atr_period).mean()
    
    def detect_gaps(self, min_gap_pct: float = 0.1) -> pd.DataFrame:
        """
        Detect all gaps in the dataset.
        
        Args:
            min_gap_pct: Minimum gap percentage to be considered (default: 0.1%)
        
        Returns:
            DataFrame containing only the rows with detected gaps
        """
        # Calculate previous close
        self.df['prev_close'] = self.df['close'].shift(1)
        
        # Calculate gap size and percentage
        self.df['gap_size'] = self.df['open'] - self.df['prev_close']
        self.df['gap_pct'] = (self.df['gap_size'] / self.df['prev_close']) * 100
        
        # Identify gap direction
        self.df['gap_type'] = 'none'
        self.df.loc[self.df['gap_pct'] > min_gap_pct, 'gap_type'] = 'up'
        self.df.loc[self.df['gap_pct'] < -min_gap_pct, 'gap_type'] = 'down'
        
        # Filter rows with gaps
        gaps_df = self.df[self.df['gap_type'] != 'none'].copy()
        
        print(f"Detected {len(gaps_df)} gaps (>{min_gap_pct}%) in {len(self.df)} rows.")
        
        return gaps_df
    
    def classify_gaps_by_size(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify gaps based on their size relative to ATR.
        
        - Small Gap: < 0.5x ATR
        - Medium Gap: 0.5x to 1.5x ATR
        - Large Gap: > 1.5x ATR
        
        Args:
            gaps_df: DataFrame of detected gaps
        
        Returns:
            DataFrame with added 'gap_size_class' column
        """
        if 'atr' not in gaps_df.columns or gaps_df.empty:
            return gaps_df
        
        gaps_df = gaps_df.copy()
        gaps_df['gap_abs_size'] = gaps_df['gap_size'].abs()
        
        # Classification logic
        conditions = [
            gaps_df['gap_abs_size'] < (0.5 * gaps_df['atr']),
            (gaps_df['gap_abs_size'] >= 0.5 * gaps_df['atr']) & 
            (gaps_df['gap_abs_size'] <= 1.5 * gaps_df['atr']),
            gaps_df['gap_abs_size'] > (1.5 * gaps_df['atr'])
        ]
        choices = ['small', 'medium', 'large']
        
        gaps_df['gap_size_class'] = np.select(conditions, choices, default='unknown')
        
        # Summary
        print("\nGap Size Classification Summary:")
        print(gaps_df['gap_size_class'].value_counts())
        
        return gaps_df
    
    def get_gap_statistics(self, gaps_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the detected gaps.
        
        Args:
            gaps_df: DataFrame of detected and classified gaps
        
        Returns:
            Dictionary with gap statistics
        """
        if gaps_df.empty:
            return {}
        
        stats = {
            'total_gaps': len(gaps_df),
            'up_gaps': len(gaps_df[gaps_df['gap_type'] == 'up']),
            'down_gaps': len(gaps_df[gaps_df['gap_type'] == 'down']),
            'avg_gap_pct': gaps_df['gap_pct'].mean(),
            'max_gap_pct': gaps_df['gap_pct'].max(),
            'min_gap_pct': gaps_df['gap_pct'].min()
        }
        
        print("\nGap Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic daily data with some gaps
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2025-01-01', periods=n, freq='B')  # Business days
    
    # Generate base prices
    close_prices = 100 + np.cumsum(np.random.randn(n) * 1.5)
    
    # Create open prices (usually close to previous close, but with gaps)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 100
    
    # Inject some artificial gaps
    open_prices[10] += 3.0   # Large up gap
    open_prices[25] -= 2.5   # Large down gap
    open_prices[50] += 0.5   # Small up gap
    
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 0.5)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 0.5)
    volume = np.random.randint(100000, 500000, n)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    print("=" * 60)
    print("CHAPTER 3: GAP DETECTOR DEMO")
    print("=" * 60)
    
    # Initialize detector
    detector = GapDetector(df, atr_period=14)
    
    # Detect gaps (minimum 0.5% to filter noise)
    gaps_df = detector.detect_gaps(min_gap_pct=0.5)
    
    if not gaps_df.empty:
        # Classify them
        gaps_df = detector.classify_gaps_by_size(gaps_df)
        
        # Get stats
        detector.get_gap_statistics(gaps_df)
