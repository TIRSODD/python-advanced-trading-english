"""
Chapter 5: Gap Fade Strategy Backtester
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

A complete backtester for the Gap Fade strategy.
This module simulates trades across multiple days where gaps occur,
applying proper risk management and calculating professional metrics.

Metrics calculated:
- Total Return (%)
- Win Rate (%)
- Profit Factor (gross profit / gross loss)
- Maximum Drawdown (%)
- Average Win/Loss Ratio
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, List


class GapFadeBacktester:
    """
    Complete backtester for the Gap Fade strategy.
    """
    
    def __init__(self, daily_data: pd.DataFrame,
                 intraday_data_dict: Dict[str, pd.DataFrame],
                 min_gap_pct: float = 0.3,
                 max_gap_pct: float = 2.0,
                 risk_per_trade: float = 0.01,
                 initial_capital: float = 10000.0):
        """
        Initialize the backtester.
        
        Args:
            daily_data: DataFrame with daily OHLCV data (to detect gaps)
            intraday_data_dict: Dictionary {date: intraday_df} for trade simulation
            min_gap_pct: Minimum gap percentage to trade
            max_gap_pct: Maximum gap percentage to trade
            risk_per_trade: Risk per trade as fraction of capital
            initial_capital: Starting capital
        """
        self.daily_data = daily_data.copy()
        self.intraday_data_dict = intraday_data_dict
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.risk_per_trade = risk_per_trade
        self.initial_capital = initial_capital
        
        self.trades = []
        self.equity_curve = []
    
    def detect_tradeable_gaps(self) -> List[pd.Timestamp]:
        """
        Identify days with tradeable gaps based on daily data.
        
        Returns:
            List of dates with tradeable gaps
        """
        # Calculate gap percentage
        self.daily_data['prev_close'] = self.daily_data['close'].shift(1)
        self.daily_data['gap_pct'] = (
            (self.daily_data['open'] - self.daily_data['prev_close']) / 
            self.daily_data['prev_close'] * 100
        )
        
        # Filter tradeable gaps
        tradeable_mask = (
            (self.daily_data['gap_pct'].abs() >= self.min_gap_pct) & 
            (self.daily_data['gap_pct'].abs() <= self.max_gap_pct)
        )
        
        tradeable_dates = self.daily_data[tradeable_mask].index.tolist()
        
        print(f"Detected {len(tradeable_dates)} tradeable gaps out of {len(self.daily_data)} days.")
        
        return tradeable_dates
    
    def simulate_trades(self) -> List[Dict]:
        """
        Simulate fade trades on days with tradeable gaps.
        
        Returns:
            List of trade dictionaries
        """
        tradeable_dates = self.detect_tradeable_gaps()
        
        capital = self.initial_capital
        trades = []
        
        for date in tradeable_dates:
            if date not in self.intraday_data_dict:
                continue
            
            intraday_df = self.intraday_data_dict[date]
            if len(intraday_df) == 0:
                continue
            
            # Get gap info
            row = self.daily_data.loc[date]
            gap_pct = row['gap_pct']
            open_price = row['open']
            prev_close = row['prev_close']
            
            # Determine direction (Fade logic)
            if gap_pct > 0:
                # Gap up -> Sell
                direction = -1
                entry_price = open_price
                target_price = prev_close
                # Stop loss above the gap high (simplified)
                stop_loss = open_price + (abs(open_price - prev_close) * 1.5)
            else:
                # Gap down -> Buy
                direction = 1
                entry_price = open_price
                target_price = prev_close
                # Stop loss below the gap low (simplified)
                stop_loss = open_price - (abs(open_price - prev_close) * 1.5)
            
            # Calculate position size
            risk_amount = capital * self.risk_per_trade
            risk_per_unit = abs(entry_price - stop_loss)
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            # Simulate intraday price action to find exit
            exit_price = None
            exit_reason = 'end_of_day'
            
            for _, intra_row in intraday_df.iterrows():
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
                exit_price = intraday_df.iloc[-1]['close']
            
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
            
            print(f"[{date.date()}] {trades[-1]['direction']} | Gap: {gap_pct:+.2f}% | "
                  f"Exit: {exit_reason} | PnL: {pnl:+.2f}")
        
        self.trades = trades
        self.final_capital = capital
        
        return trades
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics from the simulated trades."""
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
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital * 100
        
        # Maximum drawdown
        equity = [self.initial_capital]
        for trade in self.trades:
            equity.append(equity[-1] + trade['pnl'])
        
        equity_series = pd.Series(equity)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
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
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'max_drawdown_pct': max_drawdown
        }
    
    def print_report(self):
        """Print a complete backtest report."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("BACKTEST REPORT - GAP FADE STRATEGY")
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
        print("-" * 60)
        print(f"Avg Win:            ${metrics['avg_win']:,.2f}")
        print(f"Avg Loss:           ${metrics['avg_loss']:,.2f}")
        print(f"Win/Loss Ratio:     {metrics['win_loss_ratio']:.2f}")
        print("-" * 60)
        print(f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
        print("=" * 60)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic daily data
    np.random.seed(42)
    n_days = 50
    
    dates = pd.date_range('2025-01-01', periods=n_days, freq='B')
    
    close_prices = 100 + np.cumsum(np.random.randn(n_days) * 1.0)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = 100
    
    # Inject some gaps
    for i in range(5, n_days, 5):
        open_prices[i] += np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
    
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n_days) * 0.5)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n_days) * 0.5)
    
    daily_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(100000, 500000, n_days)
    }, index=dates)
    
    # Create synthetic intraday data for gap days
    intraday_dict = {}
    for date in dates:
        # Simulate 100 minutes of intraday data
        intra_dates = pd.date_range(date, periods=100, freq='1min')
        base = daily_data.loc[date, 'open']
        prices = base + np.cumsum(np.random.randn(100) * 0.1)
        
        intraday_dict[date] = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(100) * 0.1),
            'low': prices - np.abs(np.random.randn(100) * 0.1),
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        }, index=intra_dates)
    
    print("=" * 60)
    print("CHAPTER 5: GAP FADE BACKTESTER DEMO")
    print("=" * 60)
    
    # Initialize backtester
    backtester = GapFadeBacktester(
        daily_data=daily_data,
        intraday_data_dict=intraday_dict,
        min_gap_pct=0.3,
        max_gap_pct=2.0,
        risk_per_trade=0.01,
        initial_capital=10000.0
    )
    
    # Run simulation
    backtester.simulate_trades()
    
    # Print report
    backtester.print_report()
