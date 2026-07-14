"""
Chapter 7: Inter-Session Gap Analyzer
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

While most traders focus on the daily opening gap, significant price movements
also occur between trading sessions. The Forex market operates 24 hours, but
liquidity and volatility vary dramatically across sessions:

- Asian Session (Tokyo): 00:00 - 09:00 UTC (low volatility, range-bound)
- European Session (London): 08:00 - 17:00 UTC (high volatility, trend-setting)
- US Session (New York): 13:00 - 22:00 UTC (highest volatility, news-driven)

This module analyzes gaps that form between these sessions, which can provide
trading opportunities based on how different market participants react to
overnight news and positioning.
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, List, Tuple


class InterSessionGapAnalyzer:
    """
    Analyze gaps between different trading sessions.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer.
        
        Args:
            df: DataFrame with intraday OHLCV data (DatetimeIndex required)
        """
        self.df = df.copy()
        self.session_gaps = []
        
        # Define session times (UTC)
        self.sessions = {
            'asian': {'start': time(0, 0), 'end': time(9, 0)},
            'european': {'start': time(8, 0), 'end': time(17, 0)},
            'us': {'start': time(13, 0), 'end': time(22, 0)}
        }
    
    def identify_session_boundaries(self) -> pd.DataFrame:
        """
        Identify the first and last bar of each session for each day.
        
        Returns:
            DataFrame with session boundary information
        """
        df = self.df.copy()
        df['date'] = df.index.date
        df['time'] = df.index.time
        
        session_boundaries = []
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            
            for session_name, session_times in self.sessions.items():
                session_data = day_data[
                    (day_data['time'] >= session_times['start']) & 
                    (day_data['time'] <= session_times['end'])
                ]
                
                if len(session_data) > 0:
                    session_boundaries.append({
                        'date': date,
                        'session': session_name,
                        'first_bar_time': session_data.index[0],
                        'last_bar_time': session_data.index[-1],
                        'first_open': session_data.iloc[0]['open'],
                        'last_close': session_data.iloc[-1]['close'],
                        'session_high': session_data['high'].max(),
                        'session_low': session_data['low'].min(),
                        'session_volume': session_data['volume'].sum()
                    })
        
        boundaries_df = pd.DataFrame(session_boundaries)
        return boundaries_df
    
    def calculate_inter_session_gaps(self) -> pd.DataFrame:
        """
        Calculate gaps between consecutive sessions.
        
        Returns:
            DataFrame with inter-session gap information
        """
        boundaries = self.identify_session_boundaries()
        
        if boundaries.empty:
            return pd.DataFrame()
        
        # Sort by date and session
        session_order = ['asian', 'european', 'us']
        boundaries['session_order'] = boundaries['session'].apply(
            lambda x: session_order.index(x)
        )
        boundaries = boundaries.sort_values(['date', 'session_order'])
        
        # Calculate gaps between sessions
        gaps = []
        
        for i in range(1, len(boundaries)):
            prev_session = boundaries.iloc[i - 1]
            curr_session = boundaries.iloc[i]
            
            # Only calculate gaps between consecutive sessions on same day
            # or between US close and next day's Asian open
            if prev_session['date'] == curr_session['date']:
                gap_type = f"{prev_session['session']}_to_{curr_session['session']}"
            elif curr_session['session'] == 'asian':
                gap_type = f"us_to_asian_overnight"
            else:
                continue
            
            # Calculate gap
            gap_size = curr_session['first_open'] - prev_session['last_close']
            gap_pct = (gap_size / prev_session['last_close']) * 100
            
            gaps.append({
                'date': curr_session['date'],
                'gap_type': gap_type,
                'prev_session': prev_session['session'],
                'curr_session': curr_session['session'],
                'prev_close': prev_session['last_close'],
                'curr_open': curr_session['first_open'],
                'gap_size': gap_size,
                'gap_pct': gap_pct,
                'direction': 'up' if gap_pct > 0 else 'down'
            })
        
        gaps_df = pd.DataFrame(gaps)
        self.session_gaps = gaps_df
        
        print(f"Detected {len(gaps_df)} inter-session gaps")
        
        return gaps_df
    
    def analyze_gap_characteristics(self, gaps_df: pd.DataFrame) -> Dict:
        """
        Analyze characteristics of inter-session gaps.
        
        Args:
            gaps_df: DataFrame with inter-session gaps
        
        Returns:
            Dictionary with gap characteristics
        """
        if gaps_df.empty:
            return {}
        
        # Group by gap type
        gap_types = gaps_df['gap_type'].unique()
        
        stats = {}
        for gap_type in gap_types:
            type_gaps = gaps_df[gaps_df['gap_type'] == gap_type]
            
            stats[gap_type] = {
                'count': len(type_gaps),
                'avg_gap_pct': type_gaps['gap_pct'].mean(),
                'max_gap_pct': type_gaps['gap_pct'].abs().max(),
                'up_gaps': len(type_gaps[type_gaps['direction'] == 'up']),
                'down_gaps': len(type_gaps[type_gaps['direction'] == 'down'])
            }
        
        print("\nInter-Session Gap Characteristics:")
        for gap_type, stat in stats.items():
            print(f"\n{gap_type.upper()}:")
            print(f"  Count: {stat['count']}")
            print(f"  Avg Gap: {stat['avg_gap_pct']:+.2f}%")
            print(f"  Max Gap: {stat['max_gap_pct']:.2f}%")
            print(f"  Up Gaps: {stat['up_gaps']}")
            print(f"  Down Gaps: {stat['down_gaps']}")
        
        return stats
    
    def get_session_volatility_profile(self) -> pd.DataFrame:
        """
        Calculate volatility profile for each session.
        
        Returns:
            DataFrame with volatility metrics per session
        """
        boundaries = self.identify_session_boundaries()
        
        if boundaries.empty:
            return pd.DataFrame()
        
        # Calculate session range (high - low) as % of open
        boundaries['range_pct'] = (
            (boundaries['session_high'] - boundaries['session_low']) / 
            boundaries['first_open'] * 100
        )
        
        # Group by session
        session_stats = boundaries.groupby('session').agg({
            'range_pct': ['mean', 'std', 'max'],
            'session_volume': ['mean', 'std']
        }).round(2)
        
        print("\nSession Volatility Profile:")
        print(session_stats)
        
        return session_stats


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic 24-hour intraday data for 5 days
    np.random.seed(42)
    
    # Generate 5 days of 1-minute data (24 hours per day)
    start_date = pd.Timestamp('2025-07-07 00:00')
    n_minutes = 5 * 24 * 60  # 5 days * 24 hours * 60 minutes
    
    dates = pd.date_range(start_date, periods=n_minutes, freq='1min')
    
    # Simulate price with different volatility per session
    base_price = 1.10000
    prices = [base_price]
    
    for i in range(1, n_minutes):
        hour = dates[i].hour
        
        # Different volatility per session
        if 0 <= hour < 9:  # Asian session (low volatility)
            volatility = 0.0001
        elif 8 <= hour < 17:  # European session (medium volatility)
            volatility = 0.0003
        elif 13 <= hour < 22:  # US session (high volatility)
            volatility = 0.0005
        else:  # Off-hours
            volatility = 0.0001
        
        # Add some trend bias during US session
        if 13 <= hour < 17:
            drift = 0.00005  # Slight upward bias
        else:
            drift = 0
        
        new_price = prices[-1] + np.random.randn() * volatility + drift
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n_minutes) * 0.0002),
        'low': prices - np.abs(np.random.randn(n_minutes) * 0.0002),
        'close': prices + np.random.randn(n_minutes) * 0.0001,
        'volume': np.random.randint(100, 1000, n_minutes)
    }, index=dates)
    
    print("=" * 60)
    print("CHAPTER 7: INTER-SESSION GAP ANALYZER DEMO")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = InterSessionGapAnalyzer(df)
    
    # Calculate inter-session gaps
    gaps_df = analyzer.calculate_inter_session_gaps()
    
    if not gaps_df.empty:
        # Analyze characteristics
        analyzer.analyze_gap_characteristics(gaps_df)
    
    # Get session volatility profile
    analyzer.get_session_volatility_profile()
