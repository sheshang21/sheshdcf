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


def fetch_screener_financials(symbol, num_years=5):
    """
    Scrape full financials from screener.in and return in DCF-compatible format
    
    All values returned in ₹ Lacs (Screener publishes in Crores, so multiply by 10)
    
    Args:
        symbol: NSE/BSE ticker symbol (e.g. 'TATACAP', 'VEDL')
        num_years: Number of historical years to fetch (default: 5)
    
    Returns:
        {
            'financials': dict with keys matching extract_financials_listed output,
            'shares': int - shares outstanding (derived from Net Profit / EPS),
            'company_name': str - company name from page
        }
        or None on failure
    """
    import time as _time
    import random as _random
    
    try:
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Try consolidated first, then standalone
        urls_to_try = [
            f"https://www.screener.in/company/{symbol}/consolidated/",
            f"https://www.screener.in/company/{symbol}/"
        ]
        
        soup = None
        for url in urls_to_try:
            _time.sleep(_random.uniform(1.5, 3.0))  # Respectful delay
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'lxml')
                break
        
        if soup is None:
            print(f"[Screener] Could not fetch page for {symbol}")
            return None
        
        # Get company name
        title_tag = soup.find('h1')
        company_name = title_tag.get_text(strip=True) if title_tag else symbol
        
        # Helper functions
        def find_table_by_heading(soup, heading_keywords):
            """Find table following h2/h3 matching keywords"""
            for heading in soup.find_all(['h2', 'h3']):
                text_lower = heading.get_text(strip=True).lower()
                if all(kw.lower() in text_lower for kw in heading_keywords):
                    sibling = heading.find_next_sibling()
                    while sibling:
                        if sibling.name == 'table':
                            return sibling
                        tbl = sibling.find('table') if sibling.name else None
                        if tbl:
                            return tbl
                        sibling = sibling.find_next_sibling()
            return None
        
        def parse_row(table, keywords):
            """Find row matching keywords and extract values"""
            if table is None:
                return []
            for tr in table.find_all('tr'):
                cells = tr.find_all(['td', 'th'])
                if not cells:
                    continue
                label = cells[0].get_text(strip=True).lower()
                label = label.replace('\xa0', ' ').replace('–', '').replace('-', '').strip()
                for kw in keywords:
                    if kw.lower() in label:
                        values = []
                        for cell in cells[1:]:
                            raw = cell.get_text(strip=True).replace(',', '').replace('\xa0', '')
                            try:
                                values.append(float(raw))
                            except (ValueError, TypeError):
                                values.append(0.0)
                        return values
            return []
        
        # Locate P&L and Balance Sheet tables
        all_tables = soup.find_all('table')
        
        print(f"[DEBUG] Found {len(all_tables)} tables total on Screener.in page")
        
        pl_table = find_table_by_heading(soup, ['profit', 'loss'])
        if pl_table is None and len(all_tables) >= 1:
            pl_table = all_tables[0]
            print(f"[DEBUG] Using first table as P&L (no heading found)")
        
        bs_table = find_table_by_heading(soup, ['balance', 'sheet'])
        if bs_table is None and len(all_tables) >= 2:
            bs_table = all_tables[1]
            print(f"[DEBUG] Using second table as Balance Sheet (no heading found)")
        
        if pl_table is None or bs_table is None:
            print(f"[Screener] Could not locate P&L/BS tables for {symbol}")
            print(f"[DEBUG] Found tables: {len(all_tables)}")
            return None
        
        # Parse rows from P&L
        raw_revenue = parse_row(pl_table, ['revenue', 'sales'])
        raw_interest = parse_row(pl_table, ['interest'])
        raw_expenses = parse_row(pl_table, ['expenses'])
        raw_depreciation = parse_row(pl_table, ['depreciation'])
        raw_pbt = parse_row(pl_table, ['profit before tax'])
        raw_tax_pct = parse_row(pl_table, ['tax %'])
        raw_net_profit = parse_row(pl_table, ['net profit'])
        raw_eps = parse_row(pl_table, ['eps in rs', 'eps'])
        
        # Parse rows from Balance Sheet - FIXED to actually find the data
        # The issue: Screener uses different exact labels, need to be more flexible
        
        def safe_parse(table, keywords_list):
            """Try multiple keyword variations"""
            for keywords in keywords_list:
                result = parse_row(table, keywords)
                if result:  # If we got non-empty result
                    return result
            return []
        
        # Try multiple variations for each field
        raw_equity_capital = safe_parse(bs_table, [
            ['equity capital'], 
            ['equity share capital'],
            ['share capital']
        ])
        
        raw_reserves = safe_parse(bs_table, [
            ['reserves'], 
            ['reserves and surplus'],
            ['other equity']
        ])
        
        raw_borrowing = safe_parse(bs_table, [
            ['borrowing'], 
            ['borrowings'],
            ['total borrowings'],
            ['debt']
        ])
        
        raw_payables = safe_parse(bs_table, [
            ['trade payables'], 
            ['payables'],
            ['accounts payable'],
            ['creditors']
        ])
        
        raw_receivables = safe_parse(bs_table, [
            ['trade receivables'], 
            ['receivables'],
            ['accounts receivable'],
            ['debtors']
        ])
        
        raw_gross_block = safe_parse(bs_table, [
            ['gross block'],
            ['fixed assets'],
            ['property plant equipment'],
            ['ppe']
        ])
        
        raw_accum_dep = safe_parse(bs_table, [
            ['accumulated depreciation'],
            ['depreciation']
        ])
        
        raw_cash = safe_parse(bs_table, [
            ['cash equivalents'],
            ['cash and bank'],
            ['cash'],
            ['bank balances']
        ])
        
        raw_inventory = safe_parse(bs_table, [
            ['inventories'],
            ['inventory'],
            ['stock']
        ])
        
        # DEBUG: Log what we found
        print(f"[DEBUG] Balance sheet parsing results:")
        print(f"  Equity Capital: {len(raw_equity_capital)} values")
        print(f"  Reserves: {len(raw_reserves)} values")
        print(f"  Borrowing: {len(raw_borrowing)} values")
        print(f"  Payables: {len(raw_payables)} values")
        print(f"  Receivables: {len(raw_receivables)} values")
        print(f"  Gross Block: {len(raw_gross_block)} values")
        print(f"  Cash: {len(raw_cash)} values")
        print(f"  Inventory: {len(raw_inventory)} values")
        
        # Normalize length
        def pad(lst, n):
            """Keep last n values; pad front with 0.0 if shorter"""
            lst = [v for v in lst if v is not None]
            if len(lst) < n:
                lst = [0.0] * (n - len(lst)) + lst
            return lst[-n:]  # Screener: oldest → newest
        
        n = num_years
        revenue = pad(raw_revenue, n)
        interest = pad(raw_interest, n)
        expenses = pad(raw_expenses, n)
        depreciation = pad(raw_depreciation, n)
        pbt = pad(raw_pbt, n)
        tax_pct = pad(raw_tax_pct, n)
        net_profit = pad(raw_net_profit, n)
        eps = pad(raw_eps, n)
        
        equity_capital = pad(raw_equity_capital, n)
        reserves = pad(raw_reserves, n)
        borrowing = pad(raw_borrowing, n)
        payables = pad(raw_payables, n)
        receivables = pad(raw_receivables, n)
        gross_block = pad(raw_gross_block, n)
        accum_dep = pad(raw_accum_dep, n)
        cash_vals = pad(raw_cash, n)
        inventory_vals = pad(raw_inventory, n)
        
        # Derive shares from EPS
        shares = 0
        for i in range(n - 1, -1, -1):  # Newest first
            if eps[i] != 0 and net_profit[i] != 0:
                shares = int((net_profit[i] * 10_000_000) / eps[i])
                break
        
        # Build financials dict (values in Lacs = Crores × 10)
        CR_TO_LAC = 10.0
        
        # Year labels: newest → oldest (index 0 = most recent)
        from datetime import datetime as _dt
        current_year = _dt.now().year
        years_labels = [str(current_year - i) for i in range(n)]
        
        financials_out = {
            'years': years_labels,
            'revenue': [],
            'cogs': [],
            'opex': [],
            'ebitda': [],
            'depreciation': [],
            'ebit': [],
            'interest': [],
            'interest_income': [],
            'tax': [],
            'nopat': [],
            'fixed_assets': [],
            'inventory': [],
            'receivables': [],
            'payables': [],
            'cash': [],
            'equity': [],
            'st_debt': [],
            'lt_debt': [],
        }
        
        # Iterate newest → oldest (reverse scraped arrays)
        for i in range(n - 1, -1, -1):
            rev = revenue[i] * CR_TO_LAC
            dep_val = depreciation[i] * CR_TO_LAC
            int_val = interest[i] * CR_TO_LAC
            pbt_val = pbt[i] * CR_TO_LAC
            
            # EBITDA = PBT + Interest + Depreciation
            ebitda_val = pbt_val + int_val + dep_val
            
            # COGS approximation: 55% of revenue (default)
            cogs_val = rev * 0.55 if rev > 0 else 0.0
            # OpEx derived: Revenue − COGS − OpEx = EBITDA
            opex_val = rev - cogs_val - ebitda_val
            if opex_val < 0:
                opex_val = expenses[i] * CR_TO_LAC
                cogs_val = rev - opex_val - ebitda_val
                if cogs_val < 0:
                    cogs_val = 0.0
                    opex_val = rev - ebitda_val
            
            ebit_val = ebitda_val - dep_val
            
            # Tax rate
            t_rate = tax_pct[i]
            if t_rate > 1:
                t_rate = t_rate / 100.0
            t_rate = max(0.0, min(t_rate, 0.40))
            tax_val = pbt_val * t_rate
            nopat_val = ebit_val * (1 - t_rate)
            
            # Balance sheet
            eq_val = (equity_capital[i] + reserves[i]) * CR_TO_LAC
            fa_val = (gross_block[i] - accum_dep[i]) * CR_TO_LAC
            if fa_val < 0:
                fa_val = gross_block[i] * CR_TO_LAC
            pay_val = payables[i] * CR_TO_LAC
            rec_val = receivables[i] * CR_TO_LAC
            cash_val = cash_vals[i] * CR_TO_LAC
            inv_val = inventory_vals[i] * CR_TO_LAC
            borrow_val = borrowing[i] * CR_TO_LAC
            st_debt_val = borrow_val * 0.30
            lt_debt_val = borrow_val * 0.70
            
            financials_out['revenue'].append(rev)
            financials_out['cogs'].append(cogs_val)
            financials_out['opex'].append(opex_val)
            financials_out['ebitda'].append(ebitda_val)
            financials_out['depreciation'].append(dep_val)
            financials_out['ebit'].append(ebit_val)
            financials_out['interest'].append(int_val)
            financials_out['interest_income'].append(int_val)
            financials_out['tax'].append(tax_val)
            financials_out['nopat'].append(nopat_val)
            financials_out['fixed_assets'].append(fa_val)
            financials_out['inventory'].append(inv_val)
            financials_out['receivables'].append(rec_val)
            financials_out['payables'].append(pay_val)
            financials_out['cash'].append(cash_val)
            financials_out['equity'].append(eq_val)
            financials_out['st_debt'].append(st_debt_val)
            financials_out['lt_debt'].append(lt_debt_val)
        
        return {
            'financials': financials_out,
            'shares': shares,
            'company_name': company_name
        }
    
    except Exception as e:
        print(f"[Screener] Error fetching financials for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_indian_apis()
