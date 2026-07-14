"""
Chapter 4: ORB Strategy Backtester
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

A complete backtester for the Opening Range Breakout (ORB) strategy.
This module simulates trades with proper risk management and calculates
professional performance metrics.

Metrics calculated:
- Total Return (%)
- Win Rate (%)
- Profit Factor (gross profit / gross loss)
- Maximum Drawdown (%)
- Sharpe Ratio (annualized)
- Average Win/Loss Ratio
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Dict, List, Tuple


class ORBBacktester:
    """
    Complete backtester for the ORB strategy.
    """
    
    def __init__(self, df: pd.DataFrame,
                 opening_minutes: int = 15,
                 profit_target_multiplier: float = 2.0,
                 risk_per_trade: float = 0.01,
                 initial_capital: float = 10000.0,
                 open_time: time = time(9, 30)):
        """
        Initialize the backtester.
        
        Args:
            df: DataFrame with intraday OHLCV data
            opening_minutes: Minutes to define the opening range
            profit_target_multiplier: Multiple of range size for take profit
            risk_per_trade: Risk per trade as fraction of capital (default: 1%)
            initial_capital: Starting capital (default: $10,000)
            open_time: Market open time
        """
        self.df = df.copy()
        self.opening_minutes = opening_minutes
        self.profit_target_multiplier = profit_target_multiplier
        self.risk_per_trade = risk_per_trade
        self.initial_capital = initial_capital
        self.open_time = open_time
        
        self.trades = []
        self.equity_curve = []
    
    def calculate_opening_range(self) -> Tuple[float, float, float]:
        """Calculate the Opening Range."""
        opening_data = self.df.head(self.opening_minutes)
        
        range_high = opening_data['high'].max()
        range_low = opening_data['low'].min()
        range_size = range_high - range_low
        
        return range_high, range_low, range_size
    
    def simulate_trades(self) -> List[Dict]:
        """
        Simulate trades based on ORB signals.
        
        Returns:
            List of trade dictionaries
        """
        range_high, range_low, range_size = self.calculate_opening_range()
        
        if range_size == 0:
            print("Opening range size is zero. No trades possible.")
            return []
        
        df = self.df.copy()
        capital = self.initial_capital
        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        trades = []
        
        # Iterate through data after opening range
        for i in range(self.opening_minutes, len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Check for entry signals (only if flat)
            if position == 0:
                if row['close'] > range_high:
                    # Buy signal
                    position = 1
                    entry_price = row['close']
                    stop_loss = range_low
                    take_profit = entry_price + (range_size * self.profit_target_multiplier)
                    
                    # Calculate position size based on risk
                    risk_amount = capital * self.risk_per_trade
                    risk_per_unit = abs(entry_price - stop_loss)
                    position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                    
                    print(f"[{timestamp}] BUY at {entry_price:.5f} | "
                          f"SL: {stop_loss:.5f} | TP: {take_profit:.5f} | "
                          f"Size: {position_size:.2f}")
                
                elif row['close'] < range_low:
                    # Sell signal
                    position = -1
                    entry_price = row['close']
                    stop_loss = range_high
                    take_profit = entry_price - (range_size * self.profit_target_multiplier)
                    
                    risk_amount = capital * self.risk_per_trade
                    risk_per_unit = abs(entry_price - stop_loss)
                    position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                    
                    print(f"[{timestamp}] SELL at {entry_price:.5f} | "
                          f"SL: {stop_loss:.5f} | TP: {take_profit:.5f} | "
                          f"Size: {position_size:.2f}")
            
            # Check for exit conditions (if in position)
            if position != 0:
                # Check stop loss
                if position == 1 and row['low'] <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * position_size
                    capital += pnl
                    
                    trades.append({
                        'timestamp': timestamp,
                        'direction': 'BUY',
                        'entry': entry_price,
                        'exit': exit_price,
                        'size': position_size,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss'
                    })
                    
                    print(f"  -> STOP LOSS hit at {exit_price:.5f} | PnL: {pnl:.2f}")
                    position = 0
                
                elif position == -1 and row['high'] >= stop_loss:
                    exit_price = stop_loss
                    pnl = (entry_price - exit_price) * position_size
                    capital += pnl
                    
                    trades.append({
                        'timestamp': timestamp,
                        'direction': 'SELL',
                        'entry': entry_price,
                        'exit': exit_price,
                        'size': position_size,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss'
                    })
                    
                    print(f"  -> STOP LOSS hit at {exit_price:.5f} | PnL: {pnl:.2f}")
                    position = 0
                
                # Check take profit
                elif position == 1 and row['high'] >= take_profit:
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) * position_size
                    capital += pnl
                    
                    trades.append({
                        'timestamp': timestamp,
                        'direction': 'BUY',
                        'entry': entry_price,
                        'exit': exit_price,
                        'size': position_size,
                        'pnl': pnl,
                        'exit_reason': 'take_profit'
                    })
                    
                    print(f"  -> TAKE PROFIT hit at {exit_price:.5f} | PnL: {pnl:.2f}")
                    position = 0
                
                elif position == -1 and row['low'] <= take_profit:
                    exit_price = take_profit
                    pnl = (entry_price - exit_price) * position_size
                    capital += pnl
                    
                    trades.append({
                        'timestamp': timestamp,
                        'direction': 'SELL',
                        'entry': entry_price,
                        'exit': exit_price,
                        'size': position_size,
                        'pnl': pnl,
                        'exit_reason': 'take_profit'
                    })
                    
                    print(f"  -> TAKE PROFIT hit at {exit_price:.5f} | PnL: {pnl:.2f}")
                    position = 0
        
        # Close any open position at the end
        if position != 0:
            exit_price = df.iloc[-1]['close']
            if position == 1:
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            capital += pnl
            trades.append({
                'timestamp': df.index[-1],
                'direction': 'BUY' if position == 1 else 'SELL',
                'entry': entry_price,
                'exit': exit_price,
                'size': position_size,
                'pnl': pnl,
                'exit_reason': 'end_of_day'
            })
        
        self.trades = trades
        self.final_capital = capital
        
        return trades
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from the simulated trades.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {'total_trades': 0}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
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
        
        # Maximum drawdown (simplified)
        equity = [self.initial_capital]
        for trade in self.trades:
            equity.append(equity[-1] + trade['pnl'])
        
        equity_series = pd.Series(equity)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        metrics = {
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
        
        return metrics
    
    def print_report(self):
        """Print a complete backtest report."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("BACKTEST REPORT - ORB STRATEGY")
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
    # Create synthetic intraday data
    np.random.seed(42)
    n = 390  # 6.5 hours of 1-minute data
    
    dates = pd.date_range('2025-07-14 09:30', periods=n, freq='1min')
    
    # Simulate price movement with a breakout
    base_price = 1.10000
    prices = base_price + np.cumsum(np.random.randn(n) * 0.0002)
    prices[30:] += 0.005  # Upward breakout after 30 minutes
    
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
    print("CHAPTER 4: ORB BACKTESTER DEMO")
    print("=" * 60)
    
    # Initialize backtester
    backtester = ORBBacktester(
        df,
        opening_minutes=15,
        profit_target_multiplier=2.0,
        risk_per_trade=0.01,
        initial_capital=10000.0
    )
    
    # Simulate trades
    trades = backtester.simulate_trades()
    
    # Print report
    backtester.print_report()
