"""
Shared Utility Functions
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

This module contains helper functions used across multiple chapters.
By centralizing these functions, we avoid code duplication and ensure
consistent calculations throughout the project.
"""

import pandas as pd
import numpy as np
from datetime import time
from typing import Tuple


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sharpe Ratio.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of trading periods per year (252 for daily)
    
    Returns:
        Annualized Sharpe Ratio
    """
    if returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    return round(sharpe, 2)


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, float]:
    """
    Calculate Maximum Drawdown and its duration.
    
    Args:
        equity_curve: Series representing the portfolio value over time
    
    Returns:
        Tuple with (max_drawdown_pct, max_drawdown_duration_days)
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    
    max_dd = drawdown.min()
    
    # Calculate duration of max drawdown
    end_idx = drawdown.idxmin()
    start_idx = equity_curve[:end_idx].idxmax()
    duration = (end_idx - start_idx).days if hasattr(end_idx, 'days') else 0
    
    return round(max_dd * 100, 2), duration


def format_currency(value: float) -> str:
    """Format a number as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:+.2f}%"


def get_session_name(hour: int) -> str:
    """
    Determine the trading session based on the hour (UTC).
    
    Args:
        hour: Hour of the day (0-23)
    
    Returns:
        Session name: 'Asian', 'European', 'US', or 'Off-hours'
    """
    if 0 <= hour < 8:
        return 'Asian'
    elif 8 <= hour < 13:
        return 'European'
    elif 13 <= hour < 22:
        return 'US'
    else:
        return 'Off-hours'


def calculate_win_rate(wins: int, total_trades: int) -> float:
    """Calculate win rate percentage."""
    if total_trades == 0:
        return 0.0
    return round((wins / total_trades) * 100, 1)


def calculate_profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Calculate Profit Factor."""
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return round(gross_profit / abs(gross_loss), 2)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("UTILS/HELPERS.PY DEMO")
    print("=" * 60)
    
    # Test Sharpe Ratio
    np.random.seed(42)
    daily_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
    sharpe = calculate_sharpe_ratio(daily_returns)
    print(f"\nSharpe Ratio: {sharpe}")
    
    # Test Max Drawdown
    equity = pd.Series(10000 + np.cumsum(np.random.randn(252) * 50))
    max_dd, duration = calculate_max_drawdown(equity)
    print(f"Max Drawdown: {max_dd}% (Duration: {duration} days)")
    
    # Test formatting
    print(f"\nFormatted Currency: {format_currency(12345.678)}")
    print(f"Formatted Percentage: {format_percentage(-5.432)}")
    
    # Test session name
    print(f"\n10:00 UTC is the {get_session_name(10)} session.")
    print(f"15:00 UTC is the {get_session_name(15)} session.")
