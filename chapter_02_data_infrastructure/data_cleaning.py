"""
Chapter 2: Data Cleaning Module
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

Real market data is messy. This module provides functions to clean and validate
OHLCV data before analysis or backtesting.

Common issues addressed:
- Duplicate timestamps
- Negative or zero prices
- High < Low (data errors)
- Open/Close outside [Low, High] range
- Missing values
- Outliers (spikes)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class DataCleaner:
    """
    Clean and validate OHLCV data for algorithmic trading.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with raw DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        self.cleaning_log = []
    
    def clean_all(self) -> pd.DataFrame:
        """
        Execute all cleaning steps in order.
        
        Returns:
            Cleaned DataFrame
        """
        self._log("Starting data cleaning...")
        
        initial_rows = len(self.df)
        
        # Step 1: Remove duplicates
        self._remove_duplicates()
        
        # Step 2: Fix price errors
        self._fix_price_errors()
        
        # Step 3: Remove outliers
        self._remove_outliers()
        
        # Step 4: Handle missing values
        self._handle_missing_values()
        
        # Step 5: Filter zero volume
        self._filter_zero_volume()
        
        final_rows = len(self.df)
        removed_rows = initial_rows - final_rows
        
        self._log(f"Cleaning complete. Removed {removed_rows} rows ({removed_rows/initial_rows*100:.1f}%)")
        
        return self.df
    
    def _remove_duplicates(self):
        """Remove duplicate timestamps, keeping the first occurrence."""
        initial_len = len(self.df)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        duplicates_removed = initial_len - len(self.df)
        
        if duplicates_removed > 0:
            self._log(f"  Removed {duplicates_removed} duplicate timestamps")
    
    def _fix_price_errors(self):
        """
        Fix common price errors:
        - High < Low
        - Open/Close outside [Low, High] range
        - Negative or zero prices
        """
        # Fix High < Low (swap them)
        mask_hl = self.df['high'] < self.df['low']
        if mask_hl.any():
            count = mask_hl.sum()
            self.df.loc[mask_hl, ['high', 'low']] = self.df.loc[mask_hl, ['low', 'high']].values
            self._log(f"  Fixed {count} rows where High < Low (swapped)")
        
        # Fix Open outside [Low, High]
        mask_open_low = self.df['open'] < self.df['low']
        mask_open_high = self.df['open'] > self.df['high']
        
        if mask_open_low.any():
            self.df.loc[mask_open_low, 'open'] = self.df.loc[mask_open_low, 'low']
            self._log(f"  Fixed {mask_open_low.sum()} rows where Open < Low")
        
        if mask_open_high.any():
            self.df.loc[mask_open_high, 'open'] = self.df.loc[mask_open_high, 'high']
            self._log(f"  Fixed {mask_open_high.sum()} rows where Open > High")
        
        # Fix Close outside [Low, High]
        mask_close_low = self.df['close'] < self.df['low']
        mask_close_high = self.df['close'] > self.df['high']
        
        if mask_close_low.any():
            self.df.loc[mask_close_low, 'close'] = self.df.loc[mask_close_low, 'low']
            self._log(f"  Fixed {mask_close_low.sum()} rows where Close < Low")
        
        if mask_close_high.any():
            self.df.loc[mask_close_high, 'close'] = self.df.loc[mask_close_high, 'high']
            self._log(f"  Fixed {mask_close_high.sum()} rows where Close > High")
        
        # Remove rows with non-positive prices
        for col in ['open', 'high', 'low', 'close']:
            mask_invalid = self.df[col] <= 0
            if mask_invalid.any():
                count = mask_invalid.sum()
                self.df = self.df[~mask_invalid]
                self._log(f"  Removed {count} rows with non-positive {col}")
    
    def _remove_outliers(self, z_threshold: float = 5.0):
        """
        Remove outliers using Z-score method.
        
        Args:
            z_threshold: Number of standard deviations to consider as outlier
        """
        if len(self.df) < 10:
            return
        
        mean = self.df['close'].mean()
        std = self.df['close'].std()
        
        if std == 0:
            return
        
        z_scores = (self.df['close'] - mean) / std
        outliers = z_scores.abs() > z_threshold
        
        if outliers.any():
            count = outliers.sum()
            self.df = self.df[~outliers]
            self._log(f"  Removed {count} outliers (Z-score > {z_threshold})")
    
    def _handle_missing_values(self):
        """Handle missing values in price and volume columns."""
        # Forward fill prices
        for col in ['open', 'high', 'low', 'close']:
            if col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    self.df[col] = self.df[col].ffill()
                    self._log(f"  Forward-filled {null_count} missing values in {col}")
        
        # Fill volume with 0
        if 'volume' in self.df.columns:
            null_count = self.df['volume'].isnull().sum()
            if null_count > 0:
                self.df['volume'] = self.df['volume'].fillna(0)
                self._log(f"  Filled {null_count} missing volume values with 0")
    
    def _filter_zero_volume(self):
        """Remove rows with zero volume (no trading activity)."""
        if 'volume' not in self.df.columns:
            return
        
        zero_volume = self.df['volume'] == 0
        if zero_volume.any():
            count = zero_volume.sum()
            self.df = self.df[~zero_volume]
            self._log(f"  Removed {count} rows with zero volume")
    
    def _log(self, message: str):
        """Log a cleaning operation."""
        self.cleaning_log.append(message)
        print(message)
    
    def get_log(self) -> List[str]:
        """Return the cleaning log."""
        return self.cleaning_log


def clean_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to clean a DataFrame.
    
    Args:
        df: Raw DataFrame with OHLCV data
        verbose: If True, print cleaning log
    
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.clean_all()
    
    if verbose:
        print("\nCleaning Summary:")
        print("-" * 60)
        for entry in cleaner.get_log():
            print(entry)
    
    return cleaned_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic data with errors
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2025-01-01', periods=n, freq='1min')
    
    prices = 1.1 + np.cumsum(np.random.randn(n) * 0.0005)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.0003),
        'low': prices - np.abs(np.random.randn(n) * 0.0003),
        'close': prices + np.random.randn(n) * 0.0002,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)
    
    # Introduce some errors
    df.loc[10, 'high'] = 0.5  # High < Low
    df.loc[20, 'open'] = 0.3  # Open < Low
    df.loc[30, 'close'] = 2.0  # Close > High
    df.loc[40, 'close'] = -0.1  # Negative price
    df.loc[50, 'volume'] = 0  # Zero volume
    
    # Duplicate a row
    df = pd.concat([df, df.iloc[[100]]])
    
    print("=" * 60)
    print("CHAPTER 2: DATA CLEANING DEMO")
    print("=" * 60)
    print(f"\nOriginal data: {len(df)} rows")
    
    # Clean data
    df_clean = clean_dataframe(df)
    
    print(f"\nCleaned data: {len(df_clean)} rows")
