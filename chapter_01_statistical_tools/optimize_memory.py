"""
Chapter 1: Memory Optimization for Intraday Data
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

Intraday analysis has a technical problem: data volume.
One minute of data from a year's worth of events adds up to more than 350,000 rows.
If you analyze 50 assets, your DataFrame will grow to tens of millions of rows.

By default, Pandas uses float64 and int64. For prices (which rarely need more
than 4-5 decimal places), this is wasteful. Switching to float32 reduces
memory usage by half.
"""

import pandas as pd
import numpy as np


def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    - Float columns (open, high, low, close) -> float32
    - Integer columns (volume) -> appropriate int type
    - Datetime index is preserved
    
    Args:
        df: DataFrame with OHLCV data
        verbose: If True, print memory reduction info
    
    Returns:
        Optimized DataFrame (copy)
    """
    df_optimized = df.copy()
    
    # Memory before optimization
    if verbose:
        mem_before = df_optimized.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"Memory before optimization: {mem_before:.2f} MB")
    
    # Optimize float columns (prices)
    float_cols = ['open', 'high', 'low', 'close']
    for col in float_cols:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].astype('float32')
    
    # Optimize volume column
    if 'volume' in df_optimized.columns:
        max_volume = df_optimized['volume'].max()
        if max_volume < np.iinfo(np.int32).max:
            df_optimized['volume'] = df_optimized['volume'].astype('int32')
        elif max_volume < np.iinfo(np.int16).max:
            df_optimized['volume'] = df_optimized['volume'].astype('int16')
    
    # Memory after optimization
    if verbose:
        mem_after = df_optimized.memory_usage(deep=True).sum() / (1024 ** 2)
        reduction = (1 - mem_after / mem_before) * 100
        print(f"Memory after optimization: {mem_after:.2f} MB")
        print(f"Memory reduction: {reduction:.1f}%")
    
    return df_optimized


def optimize_large_dataset(filepath: str, chunksize: int = 100000) -> pd.DataFrame:
    """
    Load and optimize a large CSV file in chunks to avoid memory overflow.
    
    Args:
        filepath: Path to the CSV file
        chunksize: Number of rows per chunk
    
    Returns:
        Optimized DataFrame
    """
    chunks = []
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunk = optimize_memory(chunk, verbose=False)
        chunks.append(chunk)
    
    df_complete = pd.concat(chunks, ignore_index=True)
    
    print(f"Loaded and optimized {len(df_complete)} rows from {filepath}")
    return df_complete


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    n = 500000  # Simulates one year of 1-minute data
    
    dates = pd.date_range('2025-01-01', periods=n, freq='1min')
    prices = 1.1 + np.cumsum(np.random.randn(n) * 0.0005)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.0003),
        'low': prices - np.abs(np.random.randn(n) * 0.0003),
        'close': prices + np.random.randn(n) * 0.0002,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)
    
    print("=" * 60)
    print("MEMORY OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    df_optimized = optimize_memory(df)
