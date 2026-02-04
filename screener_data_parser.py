"""
Screener Data Parser - Matches HTML Structure
==============================================
Parses Screener.in HTML pages to extract financial data matching the exact structure.

This module provides:
1. Direct HTML parsing matching Screener's exact structure
2. Handles Balance Sheet and P&L tables with correct field mappings
3. Returns data in Rs. Crores (Screener's standard format)

Author: Shesh Ultimate
Version: 2.0
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random


def fetch_screener_financials_v2(symbol, num_years=5):
    """
    Enhanced Screener.in scraper matching exact HTML structure from documents
    
    Args:
        symbol: Stock symbol (e.g., 'NYKAA', 'RELIANCE')
        num_years: Number of years to extract (default 5)
    
    Returns:
        dict: Financial data in Rs. Crores matching the structure:
        {
            'years': [...],
            'revenue': [...],  # Called 'Sales' in Screener
            'expenses': [...],
            'operating_profit': [...],
            'other_income': [...],
            'interest': [...],
            'depreciation': [...],
            'profit_before_tax': [...],
            'tax_percent': [...],
            'net_profit': [...],
            'eps': [...],
            'equity_capital': [...],
            'reserves': [...],
            'borrowings': [...],  # Called 'Borrowings' in Screener
            'other_liabilities': [...],
            'fixed_assets': [...],
            'cwip': [...],
            'investments': [...],
            'other_assets': [...],
            'total_assets': [...],
            'total_liabilities': [...]
        }
    """
    
    try:
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
            time.sleep(random.uniform(1.5, 3.0))
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'lxml')
                st.success(f"✅ Connected to Screener.in: {url}")
                break
        
        if soup is None:
            st.error(f"❌ Could not access Screener.in for {symbol}")
            return None
        
        # Extract company name
        company_name = soup.find('h1').get_text(strip=True) if soup.find('h1') else symbol
        st.write(f"**Company:** {company_name}")
        
        # Find P&L section by ID (matches document structure)
        pl_section = soup.find('section', {'id': 'profit-loss'})
        if not pl_section:
            st.error("❌ Could not find Profit & Loss section")
            return None
        
        # Find Balance Sheet section by ID
        bs_section = soup.find('section', {'id': 'balance-sheet'})
        if not bs_section:
            st.error("❌ Could not find Balance Sheet section")
            return None
        
        # Get the main data tables (not segment tables)
        pl_table_div = pl_section.find('div', {'data-result-table': ''})
        bs_table_div = bs_section.find('div', {'data-result-table': ''})
        
        if not pl_table_div or not bs_table_div:
            st.error("❌ Could not find data tables")
            return None
        
        pl_table = pl_table_div.find('table', class_='data-table')
        bs_table = bs_table_div.find('table', class_='data-table')
        
        if not pl_table or not bs_table:
            st.error("❌ Could not find financial tables")
            return None
        
        st.success("✅ Found P&L and Balance Sheet tables")
        
        # Extract years from table headers
        years = []
        pl_headers = pl_table.find('thead').find_all('th')
        for th in pl_headers[1:]:  # Skip first column (item names)
            year_text = th.get_text(strip=True)
            if year_text and year_text != 'TTM':
                # Extract year from "Mar 2024" format
                try:
                    if 'Mar' in year_text or 'Sep' in year_text or 'Dec' in year_text:
                        year = int(year_text.split()[-1])
                        years.append(year)
                    elif len(year_text) == 4 and year_text.isdigit():
                        years.append(int(year_text))
                except:
                    pass
        
        if not years:
            st.error("❌ Could not extract years from table")
            return None
        
        st.write(f"**Years found:** {years}")
        
        # Helper function to extract row data matching exact field names
        def extract_row_by_exact_name(table, field_name, debug=False):
            """Extract values from a row that exactly matches the field name"""
            for tr in table.find('tbody').find_all('tr'):
                cells = tr.find_all('td')
                if not cells:
                    continue
                
                # Get the text from first cell (field name)
                first_cell = cells[0]
                label = first_cell.get_text(strip=True)
                
                # Check for exact match (case-insensitive)
                if label.lower() == field_name.lower():
                    values = []
                    for cell in cells[1:]:
                        text = cell.get_text(strip=True).replace(',', '').replace('\xa0', '')
                        try:
                            # Handle negative values
                            if text.startswith('-'):
                                values.append(-float(text[1:]))
                            else:
                                values.append(float(text))
                        except:
                            values.append(0.0)
                    
                    if debug:
                        st.write(f"  ✓ {field_name}: {values}")
                    
                    # Return only the years we need (limit to num_years)
                    return values[:len(years)][:num_years]
            
            if debug:
                st.write(f"  ✗ {field_name}: Not found")
            return [0.0] * min(len(years), num_years)
        
        st.write("### 📊 Extracting P&L Data")
        
        # Extract P&L items matching exact Screener field names
        sales = extract_row_by_exact_name(pl_table, 'Sales', debug=True)
        expenses = extract_row_by_exact_name(pl_table, 'Expenses', debug=True)
        operating_profit = extract_row_by_exact_name(pl_table, 'Operating Profit', debug=True)
        opm = extract_row_by_exact_name(pl_table, 'OPM %', debug=False)
        other_income = extract_row_by_exact_name(pl_table, 'Other Income', debug=True)
        interest = extract_row_by_exact_name(pl_table, 'Interest', debug=True)
        depreciation = extract_row_by_exact_name(pl_table, 'Depreciation', debug=True)
        profit_before_tax = extract_row_by_exact_name(pl_table, 'Profit before tax', debug=True)
        tax_percent = extract_row_by_exact_name(pl_table, 'Tax %', debug=False)
        net_profit = extract_row_by_exact_name(pl_table, 'Net Profit', debug=True)
        eps = extract_row_by_exact_name(pl_table, 'EPS in Rs', debug=True)
        
        st.write("### 🏦 Extracting Balance Sheet Data")
        
        # Extract Balance Sheet items matching exact Screener field names
        equity_capital = extract_row_by_exact_name(bs_table, 'Equity Capital', debug=True)
        reserves = extract_row_by_exact_name(bs_table, 'Reserves', debug=True)
        borrowings = extract_row_by_exact_name(bs_table, 'Borrowings', debug=True)
        other_liabilities = extract_row_by_exact_name(bs_table, 'Other Liabilities', debug=True)
        total_liabilities = extract_row_by_exact_name(bs_table, 'Total Liabilities', debug=True)
        
        fixed_assets = extract_row_by_exact_name(bs_table, 'Fixed Assets', debug=True)
        cwip = extract_row_by_exact_name(bs_table, 'CWIP', debug=True)
        investments = extract_row_by_exact_name(bs_table, 'Investments', debug=True)
        other_assets = extract_row_by_exact_name(bs_table, 'Other Assets', debug=True)
        total_assets = extract_row_by_exact_name(bs_table, 'Total Assets', debug=True)
        
        # Limit years to num_years
        years_limited = years[:num_years]
        
        # Construct the financials dictionary
        financials = {
            'years': years_limited,
            'revenue': sales,
            'expenses': expenses,
            'operating_profit': operating_profit,
            'opm': opm,
            'other_income': other_income,
            'interest': interest,
            'depreciation': depreciation,
            'profit_before_tax': profit_before_tax,
            'tax_percent': tax_percent,
            'net_profit': net_profit,
            'eps': eps,
            'equity_capital': equity_capital,
            'reserves': reserves,
            'borrowings': borrowings,
            'other_liabilities': other_liabilities,
            'total_liabilities': total_liabilities,
            'fixed_assets': fixed_assets,
            'cwip': cwip,
            'investments': investments,
            'other_assets': other_assets,
            'total_assets': total_assets,
            'company_name': company_name
        }
        
        st.success(f"✅ Extracted {len(years_limited)} years of data")
        
        return financials
        
    except Exception as e:
        st.error(f"❌ Error fetching Screener data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def convert_screener_to_dcf_format(screener_data):
    """
    Convert Screener data format to DCF-compatible format
    
    Args:
        screener_data: Dict from fetch_screener_financials_v2
    
    Returns:
        Dict in DCF format with derived metrics
    """
    
    if not screener_data:
        return None
    
    st.write("### 🔄 Converting to DCF Format")
    
    # Calculate derived metrics for DCF
    dcf_data = {
        'years': screener_data['years'],
        'revenue': screener_data['revenue'],
        'cogs': [],
        'opex': [],
        'ebitda': [],
        'depreciation': screener_data['depreciation'],
        'ebit': [],
        'interest': screener_data['interest'],
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
        'lt_debt': []
    }
    
    for i in range(len(screener_data['years'])):
        revenue = screener_data['revenue'][i]
        expenses = screener_data['expenses'][i]
        operating_profit = screener_data['operating_profit'][i]
        other_income = screener_data['other_income'][i]
        interest = screener_data['interest'][i]
        depreciation = screener_data['depreciation'][i]
        pbt = screener_data['profit_before_tax'][i]
        net_profit = screener_data['net_profit'][i]
        
        # EBITDA = Operating Profit + Depreciation
        ebitda = operating_profit + depreciation
        
        # EBIT = EBITDA - Depreciation = Operating Profit
        ebit = operating_profit
        
        # COGS and OpEx estimation
        # Total Cost = Revenue - Operating Profit
        total_cost = revenue - operating_profit
        # Assume 60% COGS, 40% OpEx
        cogs = total_cost * 0.6
        opex = total_cost * 0.4
        
        # Tax calculation
        tax = pbt - net_profit if pbt > 0 else 0
        tax_rate = (tax / pbt) if pbt > 0 else 0.25
        
        # NOPAT = EBIT * (1 - Tax Rate)
        nopat = ebit * (1 - tax_rate)
        
        # Interest income (50% of other income as approximation)
        interest_income = other_income * 0.5
        
        # Balance Sheet items
        equity = screener_data['equity_capital'][i] + screener_data['reserves'][i]
        borrowings = screener_data['borrowings'][i]
        
        # Split borrowings (assume 30% ST, 70% LT)
        st_debt = borrowings * 0.3
        lt_debt = borrowings * 0.7
        
        # Fixed assets (including CWIP)
        fixed_assets = screener_data['fixed_assets'][i] + screener_data['cwip'][i]
        
        # Working capital items (estimated from Other Assets/Liabilities)
        # These are approximations since Screener aggregates them
        other_assets_val = screener_data['other_assets'][i]
        other_liab_val = screener_data['other_liabilities'][i]
        
        # Estimate working capital components (rough approximations)
        inventory = other_assets_val * 0.2
        receivables = other_assets_val * 0.3
        cash = other_assets_val * 0.2
        payables = other_liab_val * 0.4
        
        # Append to DCF data
        dcf_data['cogs'].append(cogs)
        dcf_data['opex'].append(opex)
        dcf_data['ebitda'].append(ebitda)
        dcf_data['ebit'].append(ebit)
        dcf_data['interest_income'].append(interest_income)
        dcf_data['tax'].append(tax)
        dcf_data['nopat'].append(nopat)
        dcf_data['fixed_assets'].append(fixed_assets)
        dcf_data['inventory'].append(inventory)
        dcf_data['receivables'].append(receivables)
        dcf_data['payables'].append(payables)
        dcf_data['cash'].append(cash)
        dcf_data['equity'].append(equity)
        dcf_data['st_debt'].append(st_debt)
        dcf_data['lt_debt'].append(lt_debt)
    
    st.success("✅ Converted to DCF format")
    
    return dcf_data
