"""
Ticker Data Caching Utility
Prevents rate limiting by caching yfinance API calls
"""
import time
import pandas as pd
import yfinance as yf

# Global cache
_TICKER_DATA_CACHE = {}
_CACHE_TIMESTAMP = {}
CACHE_DURATION = 3600  # 1 hour

class CachedTickerData:
    """
    Wrapper class that caches yfinance ticker data properties
    This prevents rate limits by caching .info, .financials, etc.
    """
    def __init__(self, ticker_symbol, force_refresh=False):
        self.symbol = ticker_symbol
        self._ticker_obj = None
        self._info = None
        self._financials = None
        self._balance_sheet = None
        self._cashflow = None
        self._history = None
        self._loaded = False
        
        # Load from cache or fetch new
        if not force_refresh and ticker_symbol in _TICKER_DATA_CACHE:
            cache_age = time.time() - _CACHE_TIMESTAMP.get(ticker_symbol, 0)
            if cache_age < CACHE_DURATION:
                cached_data = _TICKER_DATA_CACHE[ticker_symbol]
                self._info = cached_data.get('info')
                self._financials = cached_data.get('financials')
                self._balance_sheet = cached_data.get('balance_sheet')
                self._cashflow = cached_data.get('cashflow')
                self._history = cached_data.get('history')
                self._loaded = True
    
    def _ensure_loaded(self):
        """Lazy load ticker data on first access"""
        if not self._loaded:
            try:
                self._ticker_obj = yf.Ticker(self.symbol)
                # Fetch all data at once to minimize API calls
                self._info = self._ticker_obj.info
                self._financials = self._ticker_obj.financials
                self._balance_sheet = self._ticker_obj.balance_sheet
                self._cashflow = self._ticker_obj.cashflow
                self._loaded = True
                
                # Cache the data
                _TICKER_DATA_CACHE[self.symbol] = {
                    'info': self._info,
                    'financials': self._financials,
                    'balance_sheet': self._balance_sheet,
                    'cashflow': self._cashflow,
                    'history': self._history
                }
                _CACHE_TIMESTAMP[self.symbol] = time.time()
            except Exception as e:
                # Return empty data on error
                self._info = {}
                self._financials = pd.DataFrame()
                self._balance_sheet = pd.DataFrame()
                self._cashflow = pd.DataFrame()
                raise e
    
    @property
    def info(self):
        """Get cached info or fetch if needed"""
        if self._info is None:
            self._ensure_loaded()
        return self._info if self._info is not None else {}
    
    @property
    def financials(self):
        """Get cached financials or fetch if needed"""
        if self._financials is None:
            self._ensure_loaded()
        return self._financials if self._financials is not None else pd.DataFrame()
    
    @property
    def balance_sheet(self):
        """Get cached balance sheet or fetch if needed"""
        if self._balance_sheet is None:
            self._ensure_loaded()
        return self._balance_sheet if self._balance_sheet is not None else pd.DataFrame()
    
    @property
    def cashflow(self):
        """Get cached cashflow or fetch if needed"""
        if self._cashflow is None:
            self._ensure_loaded()
        return self._cashflow if self._cashflow is not None else pd.DataFrame()
    
    def history(self, *args, **kwargs):
        """Get historical data - cache by parameters"""
        cache_key = f"{self.symbol}_history_{str(args)}_{str(kwargs)}"
        
        if cache_key in _TICKER_DATA_CACHE:
            cache_age = time.time() - _CACHE_TIMESTAMP.get(cache_key, 0)
            if cache_age < CACHE_DURATION:
                return _TICKER_DATA_CACHE[cache_key]
        
        # Fetch new history
        if self._ticker_obj is None:
            self._ticker_obj = yf.Ticker(self.symbol)
        
        hist_data = self._ticker_obj.history(*args, **kwargs)
        
        # Cache it
        _TICKER_DATA_CACHE[cache_key] = hist_data
        _CACHE_TIMESTAMP[cache_key] = time.time()
        
        return hist_data

def get_cached_ticker(ticker_symbol, force_refresh=False):
    """
    Get cached ticker data wrapper
    This returns a CachedTickerData object that caches .info, .financials, etc.
    to prevent rate limits from repeated API calls
    """
    return CachedTickerData(ticker_symbol, force_refresh)

def clear_ticker_cache():
    """Clear the ticker data cache"""
    global _TICKER_DATA_CACHE, _CACHE_TIMESTAMP
    _TICKER_DATA_CACHE = {}
    _CACHE_TIMESTAMP = {}
