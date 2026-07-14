"""
Chapter 7: Overnight Gap Fade Strategy
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

The overnight gap (between NY close and Asian open) is often caused by:
- Low liquidity during off-hours
- News releases outside major session times
- Position adjustments by institutional traders

This gap tends to fade (fill) during the European or US sessions when liquidity
returns. This strategy exploits that tendency.

Logic:
- Calculate overnight gap (Asian open vs NY previous close)
- If gap > threshold, enter fade trade at Asian open
- Target: NY previous close (gap fill)
- Stop loss: Beyond the gap extreme
- Hold until European session or end of day
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, List, Tuple


class OvernightGapFadeStrategy:
    """
    Trade overnight gaps by fading them during high-liquidity sessions.
    """
    
    def __init__(self, df: pd.DataFrame,
                 min_gap_pct: float = 0.2,
                 max_gap_pct: float = 1.5,
                 risk_per_trade: float = 0.01,
                 initial_capital: float = 10000.0,
                 ny_close_time: time = time(22, 0),
                 asian_open_time: time = time(0, 0)):
        """
        Initialize the strategy.
        
        Args:
            df: DataFrame with 24-hour intraday OHLCV data
            min_gap_pct: Minimum overnight gap to trade (default: 0.2%)
            max_gap_pct: Maximum overnight gap to trade (default: 1.5%)
            risk_per_trade: Risk per trade as fraction of capital
            initial_capital: Starting capital
            ny_close_time: NY session close time (UTC)
            asian_open_time: Asian session open time (UTC)
        """
        self.df = df.copy()
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.risk_per_trade = risk_per_trade
        self.initial_capital = initial_capital
        self.ny_close_time = ny_close_time
        self.asian_open_time = asian_open_time
        
        self.trades = []
    
    def detect_overnight_gaps(self) -> pd.DataFrame:
        """
        Detect overnight gaps in the dataset.
        
        Returns:
            DataFrame with overnight gap information
        """
        df = self.df.copy()
        df['date'] = df.index.date
        df['time'] = df.index.time
        
        # Find NY close and Asian open for each day
        overnight_gaps = []
        
        unique_dates = sorted(df['date'].unique())
        
        for i in range(1, len(unique_dates)):
            prev_date = unique_dates[i - 1]
            curr_date = unique_dates[i]
            
            # Get previous day's NY session last bar
            prev_day_data = df[df['date'] == prev_date]
            ny_close_data = prev_day_data[prev_day_data['time'] >= self.ny_close_time]
            
            if len(ny_close_data) == 0:
                continue
            
            ny_close_price = ny_close_data.iloc[-1]['close']
            
            # Get current day's Asian session first bar
            curr_day_data = df[df['date'] == curr_date]
            asian_open_data = curr_day_data[
                (curr_day_data['time'] >= self.asian_open_time) & 
                (curr_day_data['time'] < time(9, 0))
            ]
            
            if len(asian_open_data) == 0:
                continue
            
            asian_open_price = asian_open_data.iloc[0]['open']
            
            # Calculate gap
            gap_size = asian_open_price - ny_close_price
            gap_pct = (gap_size / ny_close_price) * 100
            
            # Check if gap is tradeable
            if abs(gap_pct) >= self.min_gap_pct and abs(gap_pct) <= self.max_gap_pct:
                overnight_gaps.append({
                    'date': curr_date,
                    'ny_close': ny_close_price,
                    'asian_open': asian_open_price,
                    'gap_size': gap_size,
                    'gap_pct': gap_pct,
                    'direction': 'up' if gap_pct > 0 else 'down'
                })
        
        gaps_df = pd.DataFrame(overnight_gaps)
        
        print(f"Detected {len(gaps_df)} tradeable overnight gaps")
        
        return gaps_df
    
    def simulate_trades(self, gaps_df: pd.DataFrame) -> List[Dict]:
        """
        Simulate fade trades on overnight gaps.
        
        Args:
            gaps_df: DataFrame with detected overnight gaps
        
        Returns:
            List of trade dictionaries
        """
        if gaps_df.empty:
            return []
        
        capital = self.initial_capital
        trades = []
        
        for _, gap_row in gaps_df.iterrows():
            date = gap_row['date']
            gap_pct = gap_row['gap_pct']
            ny_close = gap_row['ny_close']
            asian_open = gap_row['asian_open']
            
            # Determine trade direction (fade the gap)
            if gap_pct > 0:
                # Gap up -> Sell
                direction = -1
                entry_price = asian_open
                target_price = ny_close
                stop_loss = asian_open + (abs(asian_open - ny_close) * 1.5)
            else:
                # Gap down -> Buy
                direction = 1
                entry_price = asian_open
                target_price = ny_close
                stop_loss = asian_open - (abs(asian_open - ny_close) * 1.5)
            
            # Calculate position size
            risk_amount = capital * self.risk_per_trade
            risk_per_unit = abs(entry_price - stop_loss)
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            # Simulate intraday price action
            day_data = self.df[self.df.index.date == date]
            
            exit_price = None
            exit_reason = 'end_of_day'
            
            for _, intra_row in day_data.iterrows():
                if direction == -1:  # Short
                    if intra_row['low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'take_profit'
                        break
                    elif intra_row['high'] >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
                else:  # Long
                    if intra_row['high'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'take_profit'
                        break
                    elif intra_row['low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
            
            if exit_price is None:
                exit_price = day_data.iloc[-1]['close']
            
            # Calculate PnL
            if direction == 1:
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            capital += pnl
            
            trades.append({
                'date': date,
                'direction': 'BUY' if direction == 1 else 'SELL',
                'gap_pct': gap_pct,
                'entry': entry_price,
                'exit': exit_price,
                'size': position_size,
                'pnl': pnl,
                'exit_reason': exit_reason
            })
            
            print(f"[{date}] {trades[-1]['direction']} | Gap: {gap_pct:+.2f}% | "
                  f"Exit: {exit_reason} | PnL: {pnl:+.2f}")
        
        self.trades = trades
        self.final_capital = capital
        
        return trades
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {'total_trades': 0}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital * 100
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor
        }
    
    def print_report(self):
        """Print backtest report."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("BACKTEST REPORT - OVERNIGHT GAP FADE STRATEGY")
        print("=" * 60)
        print(f"Initial Capital:    ${metrics['initial_capital']:,.2f}")
        print(f"Final Capital:      ${metrics['final_capital']:,.2f}")
        print(f"Total Return:       {metrics['total_return_pct']:+.2f}%")
        print("-" * 60)
        print(f"Total Trades:       {metrics['total_trades']}")
        print(f"Winning Trades:     {metrics['winning_trades']}")
        print(f"Losing Trades:      {metrics['losing_trades']}")
        print(f"Win Rate:           {metrics['win_rate_pct']:.1f}%")
        print("-" * 60)
        print(f"Gross Profit:       ${metrics['gross_profit']:,.2f}")
        print(f"Gross Loss:         ${metrics['gross_loss']:,.2f}")
        print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
        print("=" * 60)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic 24-hour data for 30 days
    np.random.seed(42)
    
    start_date = pd.Timestamp('2025-06-01 00:00')
    n_days = 30
    n_minutes = n_days * 24 * 60
    
    dates = pd.date_range(start_date, periods=n_minutes, freq='1min')
    
    # Simulate price with overnight gaps
    base_price = 1.10000
    prices = [base_price]
    
    for i in range(1, n_minutes):
        hour = dates[i].hour
        day = dates[i].day
        
        # Different volatility per session
        if 0 <= hour < 9:  # Asian
            volatility = 0.0001
        elif 8 <= hour < 17:  # European
            volatility = 0.0003
        elif 13 <= hour < 22:  # US
            volatility = 0.0005
        else:
            volatility = 0.0001
        
        # Add overnight gap on some days
        if hour == 0 and day % 5 == 0:  # Every 5 days
            gap = np.random.choice([-1, 1]) * np.random.uniform(0.003, 0.01)
            prices.append(prices[-1] + gap)
        else:
            prices.append(prices[-1] + np.random.randn() * volatility)
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n_minutes) * 0.0002),
        'low': prices - np.abs(np.random.randn(n_minutes) * 0.0002),
        'close': prices + np.random.randn(n_minutes) * 0.0001,
        'volume': np.random.randint(100, 1000, n_minutes)
    }, index=dates)
    
    print("=" * 60)
    print("CHAPTER 7: OVERNIGHT GAP FADE STRATEGY DEMO")
    print("=" * 60)
    
    # Initialize strategy
    strategy = OvernightGapFadeStrategy(
        df,
        min_gap_pct=0.2,
        max_gap_pct=1.5,
        risk_per_trade=0.01,
        initial_capital=10000.0
    )
    
    # Detect gaps
    gaps_df = strategy.detect_overnight_gaps()
    
    if not gaps_df.empty:
        # Simulate trades
        strategy.simulate_trades(gaps_df)
        
        # Print report
        strategy.print_report()
