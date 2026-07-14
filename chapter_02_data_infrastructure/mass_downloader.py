"""
Chapter 2: Mass Downloader for Multiple Assets
Book: Python for Advanced Algorithmic Trading - Part 1
Author: Tirso Díaz Díaz
Repository: https://github.com/TIRSODD/python-advanced-trading-english

When downloading multiple assets or for extended periods, you need to respect
rate limits and handle connection errors. This module provides automated retries
and configurable pauses.
"""

import time
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


class MassDownloader:
    """
    Download historical data for multiple assets with error handling and rate limiting.
    """
    
    def __init__(self, broker_api=None):
        """
        Initialize the downloader.
        
        Args:
            broker_api: Optional broker API connection (e.g., Interactive Brokers)
        """
        self.broker_api = broker_api
        self.download_log = []
    
    def download_portfolio(self, portfolio: List[str], 
                          start_date: str = '2024-01-01',
                          end_date: str = None,
                          interval: str = '1m',
                          max_retries: int = 3,
                          delay_seconds: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple assets.
        
        Args:
            portfolio: List of asset symbols (e.g., ['EURGBP=X', 'EURUSD=X'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date (default: today)
            interval: Time interval ('1m', '5m', '15m', '1h', '1d')
            max_retries: Maximum number of retry attempts
            delay_seconds: Delay between downloads to respect rate limits
        
        Returns:
            Dictionary with {symbol: DataFrame}
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data_dict = {}
        
        print(f"Downloading data for {len(portfolio)} assets...")
        print(f"Period: {start_date} to {end_date}")
        print(f"Interval: {interval}")
        print("-" * 60)
        
        for i, symbol in enumerate(portfolio, 1):
            print(f"\n[{i}/{len(portfolio)}] Downloading {symbol}...")
            
            success = False
            for attempt in range(max_retries):
                try:
                    df = self._download_single(symbol, start_date, end_date, interval)
                    
                    if df is not None and len(df) > 0:
                        data_dict[symbol] = df
                        print(f"  ✓ Downloaded {len(df)} rows")
                        self._log_download(symbol, True, len(df), attempt + 1)
                        success = True
                        break
                    else:
                        print(f"  ⚠ No data received")
                        self._log_download(symbol, False, 0, attempt + 1)
                        break
                        
                except Exception as e:
                    print(f"  ⚠ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    time.sleep(delay_seconds * (attempt + 1))  # Exponential backoff
            
            if not success:
                print(f"  ✗ Failed to download {symbol} after {max_retries} attempts")
                self._log_download(symbol, False, 0, max_retries)
            
            # Delay between downloads
            if i < len(portfolio):
                time.sleep(delay_seconds)
        
        print("\n" + "=" * 60)
        print(f"Download complete. Successfully downloaded {len(data_dict)}/{len(portfolio)} assets")
        
        return data_dict
    
    def _download_single(self, symbol: str, start_date: str, 
                        end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Download data for a single asset using yfinance.
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
            interval: Time interval
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if len(df) == 0:
                return None
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only standard columns
            standard_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in standard_cols if col in df.columns]]
            
            return df
            
        except Exception as e:
            print(f"  Error downloading {symbol}: {str(e)}")
            return None
    
    def _log_download(self, symbol: str, success: bool, 
                     rows: int, attempts: int):
        """Log download attempt."""
        self.download_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'success': success,
            'rows': rows,
            'attempts': attempts
        })
    
    def save_log(self, filepath: str = 'download_log.txt'):
        """Save download log to file."""
        with open(filepath, 'w') as f:
            f.write("DOWNLOAD LOG\n")
            f.write("=" * 60 + "\n\n")
            
            for entry in self.download_log:
                status = "✓" if entry['success'] else "✗"
                f.write(f"{status} {entry['symbol']}: "
                       f"{entry['rows']} rows, "
                       f"{entry['attempts']} attempts\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Example portfolio
    portfolio = [
        'EURGBP=X',
        'EURUSD=X',
        'GBPUSD=X',
        'USDJPY=X',
        'XAUUSD=X',  # Gold
        'XAGUSD=X'   # Silver
    ]
    
    print("=" * 60)
    print("CHAPTER 2: MASS DOWNLOADER DEMO")
    print("=" * 60)
    
    # Initialize downloader
    downloader = MassDownloader()
    
    # Download data (last 30 days, 5-minute intervals)
    data = downloader.download_portfolio(
        portfolio=portfolio,
        start_date='2025-06-01',
        end_date='2025-07-01',
        interval='5m',
        max_retries=3,
        delay_seconds=2.0
    )
    
    # Save log
    downloader.save_log('download_log.txt')
    
    # Show summary
    print("\nDownloaded assets:")
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} rows")
