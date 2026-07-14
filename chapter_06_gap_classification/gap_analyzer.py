"""
Chapter 6: Gap Behavior Analyzer
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

After detecting and classifying gaps, we need to know how they behave.
The most important question for a Gap Fade trader is: "Does the gap fill?"

This module analyzes post-gap price action to calculate:
- Fill Rate: Percentage of gaps that completely close.
- Time to Fill: How many days/bars it takes to fill the gap.
- Maximum Excursion: How far the price moves against the fade before filling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class GapAnalyzer:
    """
    Analyze the behavior of gaps after they occur.
    """
    
    def __init__(self, df: pd.DataFrame, look_ahead_days: int = 10):
        """
        Initialize the analyzer.
        
        Args:
            df: DataFrame with daily OHLCV data (must include 'prev_close' and 'gap_pct')
            look_ahead_days: Number of days to look ahead to see if gap fills
        """
        self.df = df.copy()
        self.look_ahead_days = look_ahead_days
        self.fill_stats = []
    
    def analyze_gap_fills(self) -> pd.DataFrame:
        """
        Check if each gap gets filled within the look-ahead period.
        
        A gap is considered "filled" if:
        - For an UP gap: Price drops down to touch or break the previous close.
        - For a DOWN gap: Price rises up to touch or break the previous close.
        
        Returns:
            DataFrame with added 'filled', 'days_to_fill', and 'max_excursion' columns
        """
        df = self.df.copy()
        df['filled'] = False
        df['days_to_fill'] = np.nan
        df['max_excursion_pct'] = np.nan
        
        # Get indices of rows with gaps
        gap_indices = df[df['gap_pct'].abs() >= 0.1].index  # Analyze all gaps > 0.1%
        
        for idx in gap_indices:
            pos = df.index.get_loc(idx)
            if pos + self.look_ahead_days > len(df):
                continue
            
            row = df.iloc[pos]
            prev_close = row['prev_close']
            gap_type = 'up' if row['gap_pct'] > 0 else 'down'
            
            # Look ahead
            future_data = df.iloc[pos + 1 : pos + 1 + self.look_ahead_days]
            
            filled = False
            days_to_fill = np.nan
            max_excursion = 0.0
            
            for i, (_, future_row) in enumerate(future_data.iterrows()):
                if gap_type == 'up':
                    # Check if low touches previous close
                    if future_row['low'] <= prev_close:
                        filled = True
                        days_to_fill = i + 1
                        break
                    # Calculate max excursion (how much higher it went)
                    excursion = (future_row['high'] - row['open']) / row['open'] * 100
                    max_excursion = max(max_excursion, excursion)
                    
                else:  # down gap
                    # Check if high touches previous close
                    if future_row['high'] >= prev_close:
                        filled = True
                        days_to_fill = i + 1
                        break
                    # Calculate max excursion (how much lower it went)
                    excursion = (row['open'] - future_row['low']) / row['open'] * 100
                    max_excursion = max(max_excursion, excursion)
            
            # Update dataframe
            df.loc[idx, 'filled'] = filled
            if filled:
                df.loc[idx, 'days_to_fill'] = days_to_fill
            df.loc[idx, 'max_excursion_pct'] = max_excursion
        
        self.df = df
        return df[df['gap_pct'].abs() >= 0.1]
    
    def calculate_fill_statistics(self, gaps_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for gap fills.
        
        Args:
            gaps_df: DataFrame with analyzed gaps
        
        Returns:
            Dictionary with fill statistics
        """
        if gaps_df.empty:
            return {}
        
        total_gaps = len(gaps_df)
        filled_gaps = gaps_df[gaps_df['filled'] == True]
        
        fill_rate = len(filled_gaps) / total_gaps * 100 if total_gaps > 0 else 0
        avg_days_to_fill = filled_gaps['days_to_fill'].mean() if len(filled_gaps) > 0 else 0
        avg_max_excursion = gaps_df['max_excursion_pct'].mean()
        
        # Separate by direction
        up_gaps = gaps_df[gaps_df['gap_pct'] > 0]
        down_gaps = gaps_df[gaps_df['gap_pct'] < 0]
        
        up_fill_rate = len(up_gaps[up_gaps['filled']]) / len(up_gaps) * 100 if len(up_gaps) > 0 else 0
        down_fill_rate = len(down_gaps[down_gaps['filled']]) / len(down_gaps) * 100 if len(down_gaps) > 0 else 0
        
        stats = {
            'total_gaps': total_gaps,
            'filled_gaps': len(filled_gaps),
            'fill_rate_pct': fill_rate,
            'avg_days_to_fill': avg_days_to_fill,
            'avg_max_excursion_pct': avg_max_excursion,
            'up_gap_fill_rate': up_fill_rate,
            'down_gap_fill_rate': down_fill_rate
        }
        
        print("\nGap Fill Statistics:")
        print(f"  Total Gaps Analyzed: {total_gaps}")
        print(f"  Overall Fill Rate:   {fill_rate:.1f}%")
        print(f"  Avg Days to Fill:    {avg_days_to_fill:.1f}")
        print(f"  Avg Max Excursion:   {avg_max_excursion:.2f}%")
        print(f"  Up Gap Fill Rate:    {up_fill_rate:.1f}%")
        print(f"  Down Gap Fill Rate:  {down_fill_rate:.1f}%")
        
        return stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic daily data with gaps
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    
    close_prices = 100 + np.cumsum(np.random.randn(n) * 1.0)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 100
    
    # Inject gaps
    open_prices[10] += 2.0   # Up gap
    open_prices[25] -= 1.5   # Down gap
    open_prices[40] += 3.0   # Large up gap
    open_prices[60] -= 2.5   # Large down gap
    open_prices[80] += 1.0   # Small up gap
    
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 0.5)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(100000, 500000, n)
    }, index=dates)
    
    # Calculate gap info
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    
    print("=" * 60)
    print("CHAPTER 6: GAP ANALYZER DEMO")
    print("=" * 60)
    
    # Initialize analyzer (look ahead 10 days)
    analyzer = GapAnalyzer(df, look_ahead_days=10)
    
    # Analyze fills
    gaps_df = analyzer.analyze_gap_fills()
    
    # Print stats
    analyzer.calculate_fill_statistics(gaps_df)
    
    # Show specific gap results
    print("\nDetailed Gap Results:")
    print(gaps_df[['open', 'prev_close', 'gap_pct', 'filled', 'days_to_fill', 'max_excursion_pct']].head(10))
