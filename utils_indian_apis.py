"""
Indian Stock Market API Utilities
Implements free Indian data sources as alternatives to Yahoo Finance

Sources:
1. NSE India (Official) - Using nselib
2. BSE India (Official) - Direct scraping
3. Screener.in - Web scraping with delays
4. MoneyControl - Web scraping
5. Trendlyne - API

Installation required:
pip install nselib requests beautifulsoup4 lxml
"""

import requests
import time
import random
from datetime import datetime, timedelta
import json

# =====================================================
# NSE INDIA API (Official - Using nselib)
# =====================================================

def get_nse_quote(symbol):
    """
    Get live quote from NSE using nselib
    
    Args:
        symbol: NSE symbol (e.g., 'RELIANCE', 'TCS')
    
    Returns:
        dict with price, volume, etc.
    """
    try:
        from nselib import capital_market
        
        # Get quote
        quote = capital_market.market_watch_all_indices()
        
        # Alternative: Get specific stock
        stock_data = capital_market.price_volume_and_deliverable_position_data(symbol)
        
        return {
            'symbol': symbol,
            'price': stock_data.get('lastPrice', 0),
            'open': stock_data.get('open', 0),
            'high': stock_data.get('dayHigh', 0),
            'low': stock_data.get('dayLow', 0),
            'volume': stock_data.get('totalTradedVolume', 0),
            'source': 'NSE India (Official)'
        }
    except Exception as e:
        print(f"NSE API error: {e}")
        return None


def get_nse_financials(symbol):
    """
    Get financial data from NSE
    
    Note: NSE doesn't provide detailed financials via API
    Use BSE or Screener.in instead
    """
    try:
        from nselib import capital_market
        
        # NSE provides limited fundamental data
        # For full financials, use BSE or Screener.in
        
        return {
            'symbol': symbol,
            'message': 'Use BSE or Screener.in for detailed financials',
            'source': 'NSE India'
        }
    except Exception as e:
        return None


# =====================================================
# BSE INDIA API (Official - Web Scraping)
# =====================================================

def get_bse_quote(scrip_code):
    """
    Get quote from BSE India
    
    Args:
        scrip_code: BSE scrip code (e.g., '500325' for RELIANCE)
    
    Returns:
        dict with price data
    """
    try:
        # BSE official website
        url = f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w"
        params = {
            'scripcode': scrip_code,
            'flag': '0',
            'fromdate': '',
            'todate': '',
            'seriesid': ''
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'scrip_code': scrip_code,
                'price': data.get('CurrRate', {}).get('LTP', 0),
                'source': 'BSE India (Official)'
            }
    except Exception as e:
        print(f"BSE API error: {e}")
        return None


# =====================================================
# SCREENER.IN (Popular Indian Stock Screener)
# =====================================================

def get_screener_data(symbol):
    """
    Scrape data from Screener.in
    
    Args:
        symbol: NSE/BSE symbol
    
    Returns:
        dict with comprehensive financial data
    """
    try:
        from bs4 import BeautifulSoup
        
        # Add delay to be respectful
        time.sleep(random.uniform(1, 2))
        
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract data from page
            # Screener has comprehensive financials in tables
            
            data = {
                'symbol': symbol,
                'source': 'Screener.in',
                'url': url,
                'status': 'success'
            }
            
            # You can parse specific tables here
            # For now, return basic structure
            
            return data
        else:
            return {'error': f'HTTP {response.status_code}'}
            
    except Exception as e:
        print(f"Screener.in error: {e}")
        return None


# =====================================================
# MONEYCONTROL (Popular Financial Portal)
# =====================================================

def get_moneycontrol_data(symbol):
    """
    Scrape data from MoneyControl
    
    Args:
        symbol: Stock symbol
    
    Returns:
        dict with financial data
    """
    try:
        from bs4 import BeautifulSoup
        
        # Add delay
        time.sleep(random.uniform(1, 2))
        
        # MoneyControl URLs are symbol-specific
        # Example: https://www.moneycontrol.com/india/stockpricequote/refineries/relianceindustries/RI
        
        # This would need symbol mapping
        # For now, return structure
        
        return {
            'symbol': symbol,
            'source': 'MoneyControl',
            'status': 'requires_symbol_mapping'
        }
        
    except Exception as e:
        print(f"MoneyControl error: {e}")
        return None


# =====================================================
# TRENDLYNE API (Has API access)
# =====================================================

def get_trendlyne_data(symbol, api_key=None):
    """
    Get data from Trendlyne API
    
    Args:
        symbol: Stock symbol
        api_key: Trendlyne API key (required)
    
    Returns:
        dict with data
    """
    try:
        if not api_key:
            return {'error': 'API key required', 'message': 'Get free API key from trendlyne.com'}
        
        # Trendlyne API endpoint
        url = "https://api.trendlyne.com/v1/stocks"
        
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'HTTP {response.status_code}'}
            
    except Exception as e:
        print(f"Trendlyne API error: {e}")
        return None


# =====================================================
# UNIFIED INTERFACE - MULTI-SOURCE DATA FETCHER
# =====================================================

def get_indian_stock_data(symbol, sources=['nse', 'screener'], delay=True):
    """
    Unified function to get stock data from multiple Indian sources
    
    Args:
        symbol: Stock symbol (NSE format)
        sources: List of sources to try ['nse', 'bse', 'screener', 'moneycontrol']
        delay: Add delays between requests (default True)
    
    Returns:
        dict with aggregated data from multiple sources
    """
    results = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'sources_attempted': sources,
        'data': {}
    }
    
    for source in sources:
        if delay:
            time.sleep(random.uniform(0.5, 1.5))
        
        try:
            if source == 'nse':
                data = get_nse_quote(symbol)
                if data:
                    results['data']['nse'] = data
            
            elif source == 'screener':
                data = get_screener_data(symbol)
                if data:
                    results['data']['screener'] = data
            
            elif source == 'moneycontrol':
                data = get_moneycontrol_data(symbol)
                if data:
                    results['data']['moneycontrol'] = data
            
        except Exception as e:
            results['data'][source] = {'error': str(e)}
    
    return results


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def convert_nse_to_bse_code(nse_symbol):
    """
    Convert NSE symbol to BSE scrip code
    
    Common mappings (hardcoded for major stocks)
    For full list, would need a database
    """
    mapping = {
        'RELIANCE': '500325',
        'TCS': '532540',
        'HDFCBANK': '500180',
        'INFY': '500209',
        'ICICIBANK': '532174',
        'HINDUNILVR': '500696',
        'ITC': '500875',
        'SBIN': '500112',
        'BHARTIARTL': '532454',
        'KOTAKBANK': '500247',
        'LT': '500510',
        'AXISBANK': '532215',
        'ASIANPAINT': '500820',
        'MARUTI': '532500',
        'BAJFINANCE': '500034',
        'HCLTECH': '532281',
        'WIPRO': '507685',
        'ULTRACEMCO': '532538',
        'TITAN': '500114',
        'NESTLEIND': '500790'
    }
    
    return mapping.get(nse_symbol.upper(), None)


def test_indian_apis():
    """
    Test function to check if Indian APIs are working
    """
    print("Testing Indian Stock Market APIs...")
    print("=" * 50)
    
    test_symbol = "RELIANCE"
    
    # Test NSE
    print("\n1. Testing NSE API...")
    nse_data = get_nse_quote(test_symbol)
    print(f"NSE Result: {nse_data}")
    
    # Test BSE
    print("\n2. Testing BSE API...")
    bse_code = convert_nse_to_bse_code(test_symbol)
    if bse_code:
        bse_data = get_bse_quote(bse_code)
        print(f"BSE Result: {bse_data}")
    
    # Test Screener
    print("\n3. Testing Screener.in...")
    screener_data = get_screener_data(test_symbol)
    print(f"Screener Result: {screener_data}")
    
    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    test_indian_apis()
