"""
Chapter 1: Complete Data Pipeline
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

A data pipeline is a series of linked steps that transform raw data into
data ready for analysis. The key is to make it modular, repeatable, and
fault-tolerant.

Pipeline Architecture:
    data/
    ├── 01_raw/           # Data as downloaded
    ├── 02_clean/         # Validated and clean data
    ├── 03_processed/     # Data with calculated indicators
    └── pipeline_log.txt  # Log of all operations

The pipeline follows these steps:
1. Download  -> Save raw data with timestamp
2. Validate  -> Check integrity (columns, ranges, nulls)
3. Clean     -> Remove outliers, duplicates, fill gaps
4. Transform -> Unify time zones, calculate returns
5. Store     -> Save in efficient format (Parquet)
6. Log       -> Log of all operations with checksums
"""

import os
import hashlib
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np


class DataPipeline:
    """
    Modular, repeatable, and fault-tolerant data pipeline for intraday data.
    """
    
    def __init__(self, base_path: str = './data_trading'):
        """
        Initialize the pipeline with directory structure.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, '01_raw')
        self.clean_path = os.path.join(base_path, '02_clean')
        self.processed_path = os.path.join(base_path, '03_processed')
        self.log_path = os.path.join(base_path, 'pipeline_log.txt')
        
        # Create directories
        for path in [self.raw_path, self.clean_path, self.processed_path]:
            os.makedirs(path, exist_ok=True)
        
        self.log_entries = []
    
    def process_complete(self, df: pd.DataFrame, symbol: str,
                         time_zone: str = 'UTC') -> pd.DataFrame:
        """
        Execute the complete pipeline for a single asset.
        
        Steps:
        1. Save raw data
        2. Validate integrity
        3. Clean data
        4. Transform (time zones, returns)
        5. Store processed data
        6. Log all operations
        
        Args:
            df: Raw DataFrame with OHLCV data
            symbol: Asset symbol (e.g., 'EURGBP')
            time_zone: Target time zone (default: UTC)
        
        Returns:
            Cleaned and processed DataFrame
        """
        self._log(f"Processing {symbol}...")
        
        # Step 1: Save raw data
        self._save_raw(df, symbol)
        
        # Step 2: Validate
        df = self._validate(df, symbol)
        
        # Step 3: Clean
        df = self._clean(df, symbol)
        
        # Step 4: Transform
        df = self._transform(df, symbol, time_zone)
        
        # Step 5: Store
        self._store(df, symbol)
        
        self._log(f"✓ {symbol} processed successfully")
        
        return df
    
    def _save_raw(self, df: pd.DataFrame, symbol: str):
        """Save raw data with timestamp and checksum."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.raw_path, f"{symbol}_{timestamp}.parquet")
        df.to_parquet(filepath)
        
        checksum = self._calculate_checksum(filepath)
        self._log(f"  Raw saved: {filepath} (SHA256: {checksum[:16]}...)")
    
    def _validate(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate data integrity.
        
        Checks:
        - Required columns exist
        - high >= low in all rows
        - open and close within [low, high] range
        - No null values
        - No negative prices
        """
        self._log(f"  Validating {symbol}...")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Verify high >= low
        invalid_hl = df[df['high'] < df['low']]
        if len(invalid_hl) > 0:
            self._log(f"  ⚠ Warning: {len(invalid_hl)} rows where high < low")
            df = df[df['high'] >= df['low']]
        
        # Verify open and close within range
        invalid_open = df[(df['open'] < df['low']) | (df['open'] > df['high'])]
        invalid_close = df[(df['close'] < df['low']) | (df['close'] > df['high'])]
        
        if len(invalid_open) > 0:
            self._log(f"  ⚠ Warning: {len(invalid_open)} rows with invalid open price")
        if len(invalid_close) > 0:
            self._log(f"  ⚠ Warning: {len(invalid_close)} rows with invalid close price")
        
        # Check for null values
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            self._log(f"  ⚠ Found {null_count} null values")
        
        # Check for negative prices
        for col in ['open', 'high', 'low', 'close']:
            negative = df[df[col] <= 0]
            if len(negative) > 0:
                self._log(f"   Warning: {len(negative)} rows with non-positive {col}")
                df = df[df[col] > 0]
        
        return df
    
    def _clean(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean data: remove outliers, duplicates, fill gaps.
        """
        self._log(f"  Cleaning {symbol}...")
        
        # Remove duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        duplicates_removed = initial_len - len(df)
        
        if duplicates_removed > 0:
            self._log(f"    Removed {duplicates_removed} duplicates")
        
        # Detect outliers using Z-score (5 standard deviations)
        mean = df['close'].mean()
        std = df['close'].std()
        df['z_score'] = (df['close'] - mean) / std
        outliers = df[df['z_score'].abs() > 5]
        
        if len(outliers) > 0:
            self._log(f"    Detected {len(outliers)} outliers (Z-score > 5)")
            df = df[df['z_score'].abs() <= 5]
            df = df.drop(columns=['z_score'])
        
        # Handle missing data
        # Forward fill prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Volume: leave as 0 (no trading)
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
            # CRITICAL: Filter rows with no volume
            df = df[df['volume'] > 0]
        
        # Save clean data
        filepath = os.path.join(self.clean_path, f"{symbol}_clean.parquet")
        df.to_parquet(filepath)
        
        return df
    
    def _transform(self, df: pd.DataFrame, symbol: str,
                   time_zone: str) -> pd.DataFrame:
        """
        Transform data: time zones, returns, indicators.
        """
        self._log(f"  Transforming {symbol}...")
        
        # Unify time zone
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(time_zone)
        
        # Calculate log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate basic indicators
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        
        return df
    
    def _store(self, df: pd.DataFrame, symbol: str):
        """Store processed data in Parquet format."""
        filepath = os.path.join(self.processed_path, f"{symbol}_processed.parquet")
        df.to_parquet(filepath)
        
        checksum = self._calculate_checksum(filepath)
        self._log(f"  Processed stored: {filepath} (SHA256: {checksum[:16]}...)")
    
    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA256 checksum of a file to verify integrity."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"[{timestamp}] {message}"
        self.log_entries.append(entry)
        print(entry)
    
    def save_log(self):
        """Save pipeline log to file."""
        with open(self.log_path, 'w') as f:
            f.write('\n'.join(self.log_entries))
        print(f"\n✓ Pipeline log saved to {self.log_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Create synthetic data
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
    
    print("=" * 60)
    print("CHAPTER 1: DATA PIPELINE DEMO")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DataPipeline('./data_trading_demo')
    
    # Process asset
    df_clean = pipeline.process_complete(df, symbol='EURGBP', time_zone='UTC')
    
    # Save log
    pipeline.save_log()
    
    print(f"\n✓ Pipeline completed. Processed {len(df_clean)} rows")
