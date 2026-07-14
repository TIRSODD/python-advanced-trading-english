"""
Chapter 6: Gap Classifier
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

Not all gaps are created equal. This module classifies gaps into four categories:

1. Common Gap: Small gap in a ranging market, usually filled quickly.
2. Breakaway Gap: Large gap that breaks out of a consolidation pattern, 
   signaling the start of a new trend. Rarely filled.
3. Continuation Gap: Appears in the middle of a strong trend, confirming momentum.
4. Exhaustion Gap: Appears at the end of a trend, often followed by a reversal.

Classification criteria:
- Gap size relative to ATR
- Volume compared to average
- Position relative to recent support/resistance
- Price action in the days following the gap
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class GapClassifier:
    """
    Classify gaps into Common, Breakaway, Continuation, or Exhaustion.
    """
    
    def __init__(self, df: pd.DataFrame, atr_period: int = 14):
        """
        Initialize the classifier.
        
        Args:
            df: DataFrame with daily OHLCV data (must have 'prev_close' and 'gap_pct')
            atr_period: Period for ATR calculation
        """
        self.df = df.copy()
        self.atr_period = atr_period
        
        # Pre-calculate indicators
        self._calculate_atr()
        self._calculate_volume_ma()
        self._calculate_trend_position()
    
    def _calculate_atr(self):
        """Calculate Average True Range."""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=self.atr_period).mean()
    
    def _calculate_volume_ma(self):
        """Calculate moving average of volume."""
        self.df['volume_ma'] = self.df['volume'].rolling(window=20).mean()
    
    def _calculate_trend_position(self):
        """
        Determine if price is in an uptrend, downtrend, or ranging.
        Uses 20-day and 50-day moving averages.
        """
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma50'] = self.df['close'].rolling(window=50).mean()
        
        # Trend classification
        self.df['trend'] = 'ranging'
        self.df.loc[self.df['ma20'] > self.df['ma50'], 'trend'] = 'uptrend'
        self.df.loc[self.df['ma20'] < self.df['ma50'], 'trend'] = 'downtrend'
    
    def classify_gap(self, row: pd.Series) -> str:
        """
        Classify a single gap based on multiple criteria.
        
        Args:
            row: DataFrame row with gap information
        
        Returns:
            Gap classification: 'common', 'breakaway', 'continuation', 'exhaustion'
        """
        gap_pct = abs(row['gap_pct'])
        atr = row['atr'] if pd.notna(row['atr']) else 0
        volume = row['volume']
        volume_ma = row['volume_ma'] if pd.notna(row['volume_ma']) else volume
        trend = row['trend']
        
        # Calculate gap size relative to ATR
        gap_atr_ratio = gap_pct / (atr / row['close'] * 100) if atr > 0 else 0
        
        # Volume ratio
        volume_ratio = volume / volume_ma if volume_ma > 0 else 1
        
        # Classification logic
        if gap_atr_ratio < 0.5 and volume_ratio < 1.5:
            # Small gap, normal volume -> Common
            return 'common'
        
        elif gap_atr_ratio > 1.5 and volume_ratio > 2.0:
            # Large gap, high volume
            if trend == 'ranging':
                # Breaking out of range -> Breakaway
                return 'breakaway'
            elif trend in ['uptrend', 'downtrend']:
                # In strong trend -> Check if it's continuation or exhaustion
                # Simplified: if gap is very large, likely exhaustion
                if gap_atr_ratio > 3.0:
                    return 'exhaustion'
                else:
                    return 'continuation'
        
        elif gap_atr_ratio > 2.0 and volume_ratio > 1.5:
            # Medium-large gap with above-average volume
            if trend == 'ranging':
                return 'breakaway'
            else:
                return 'continuation'
        
        else:
            # Default to common
            return 'common'
    
    def classify_all_gaps(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all gaps in the dataset.
        
        Args:
            gaps_df: DataFrame with detected gaps
        
        Returns:
            DataFrame with added 'gap_class' column
        """
        if gaps_df.empty:
            return gaps_df
        
        gaps_df = gaps_df.copy()
        gaps_df['gap_class'] = gaps_df.apply(self.classify_gap, axis=1)
        
        # Print summary
        print("\nGap Classification Summary:")
        print(gaps_df['gap_class'].value_counts())
        
        return gaps_df
    
    def get_classification_stats(self, gaps_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for each gap classification.
        
        Args:
            gaps_df: DataFrame with classified gaps
        
        Returns:
            Dictionary with statistics per gap class
        """
        if gaps_df.empty:
            return {}
        
        stats = {}
        for gap_class in ['common', 'breakaway', 'continuation', 'exhaustion']:
            class_gaps = gaps_df[gaps_df['gap_class'] == gap_class]
            
            if len(class_gaps) > 0:
                stats[gap_class] = {
                    'count': len(class_gaps),
                    'avg_gap_pct': class_gaps['gap_pct'].mean(),
                    'avg_volume_ratio': (class_gaps['volume'] / class_gaps['volume_ma']).mean()
                }
        
        print("\nClassification Statistics:")
        for gap_class, stat in stats.items():
            print(f"  {gap_class.upper()}: {stat['count']} gaps, "
                  f"avg {stat['avg_gap_pct']:+.2f}%, "
                  f"avg volume ratio: {stat['avg_volume_ratio']:.2f}x")
        
        return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic daily data with various gap types
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    
    # Generate base prices with trend changes
    close_prices = 100 + np.cumsum(np.random.randn(n) * 1.5)
    
    # Add some trend phases
    close_prices[20:40] += 5  # Uptrend
    close_prices[60:80] -= 5  # Downtrend
    
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 100
    
    # Inject different gap types
    # Common gap (small)
    open_prices[10] += 0.3
    
    # Breakaway gap (large, breaking range)
    open_prices[20] += 4.0
    
    # Continuation gap (medium, in trend)
    open_prices[30] += 2.0
    
    # Exhaustion gap (very large, end of trend)
    open_prices[40] += 5.0
    
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 0.5)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 0.5)
    
    volume = np.random.randint(100000, 500000, n)
    # High volume on gap days
    volume[10] *= 1.2
    volume[20] *= 3.0
    volume[30] *= 2.0
    volume[40] *= 4.0
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    # Calculate gap info
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    
    # Filter gaps (minimum 0.3%)
    gaps_df = df[df['gap_pct'].abs() >= 0.3].copy()
    
    print("=" * 60)
    print("CHAPTER 6: GAP CLASSIFIER DEMO")
    print("=" * 60)
    print(f"\nTotal gaps detected: {len(gaps_df)}")
    
    # Initialize classifier
    classifier = GapClassifier(df, atr_period=14)
    
    # Classify gaps
    gaps_df = classifier.classify_all_gaps(gaps_df)
    
    # Get stats
    classifier.get_classification_stats(gaps_df)
