import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
from io import StringIO
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ================================
# UTILITY FUNCTIONS
# ================================

def sanitize_value(val):
    """Convert string values to float, handling various formats"""
    if pd.isna(val) or val == '' or val == '-':
        return 0.0
    try:
        return float(str(val).replace(',', ''))
    except:
        return 0.0

def parse_xml_to_dataframes(xml_content):
    """Parse XML file and extract Balance Sheet and P&L as DataFrames"""
    try:
        root = ET.fromstring(xml_content)
        
        balance_sheet_data = []
        profit_loss_data = []
        
        # Parse Balance Sheet
        bs_elem = root.find('BalanceSheet')
        if bs_elem is not None:
            for record in bs_elem.findall('record'):
                row_dict = {}
                for child in record:
                    if child.tag == 'BALANCE_SHEET':
                        row_dict['Item'] = child.text if child.text else ''
                    else:
                        row_dict[child.tag] = sanitize_value(child.text)
                if row_dict.get('Item'):
                    balance_sheet_data.append(row_dict)
        
        # Parse Profit & Loss
        pl_elem = root.find('Profit_Loss')
        if pl_elem is not None:
            for record in pl_elem.findall('record'):
                row_dict = {}
                for child in record:
                    if child.tag == 'PROFIT___LOSS':
                        row_dict['Item'] = child.text if child.text else ''
                    else:
                        row_dict[child.tag] = sanitize_value(child.text)
                if row_dict.get('Item'):
                    profit_loss_data.append(row_dict)
        
        df_bs = pd.DataFrame(balance_sheet_data)
        df_pl = pd.DataFrame(profit_loss_data)
        
        return df_bs, df_pl
    except Exception as e:
        st.error(f"Error parsing XML: {str(e)}")
        return None, None

def get_value_from_df(df, item_name, year_col):
    """Extract value from DataFrame by item name (case-insensitive partial match)"""
    if df is None or df.empty:
        return 0.0
    
    item_name_lower = item_name.lower()
    mask = df['Item'].str.lower().str.contains(item_name_lower, na=False, regex=False)
    matching = df[mask]
    
    if not matching.empty and year_col in matching.columns:
        return matching.iloc[0][year_col]
    return 0.0

def detect_year_columns(df):
    """Detect year columns dynamically (columns starting with _)"""
    if df is None or df.empty:
        return []
    
    year_cols = [col for col in df.columns if col.startswith('_') and col != 'Item']
    # Sort by numeric value after underscore
    year_cols.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    return year_cols

# ================================
# BUSINESS MODEL CLASSIFICATION (RULEBOOK COMPLIANT)
# ================================

def classify_business_model(financials, income_stmt=None, balance_sheet=None):
    """
    Classify company as OPERATING or INTEREST-DOMINANT per Rulebook Section 2.
    
    Returns:
        dict: {
            'type': 'OPERATING' or 'INTEREST_DOMINANT',
            'criteria_met': list of criteria that triggered classification,
            'metrics': dict of calculated ratios
        }
    """
    criteria_met = []
    metrics = {}
    
    # Calculate 3-year averages for classification
    try:
        # Method 1: From extracted financials dict
        if financials and 'revenue' in financials and 'interest' in financials:
            revenues = financials['revenue']
            interest_expenses = financials['interest']
            
            # Get interest income if available (from additional fields)
            interest_income = financials.get('interest_income', [0] * len(revenues))
            
            # Calculate averages
            avg_revenue = np.mean(revenues) if revenues else 0
            avg_interest_expense = np.mean(interest_expenses) if interest_expenses else 0
            avg_interest_income = np.mean(interest_income) if interest_income else 0
            
            # Criterion 1: Interest Income / Total Revenue ≥ 50%
            if avg_revenue > 0:
                interest_income_ratio = (avg_interest_income / avg_revenue) * 100
                metrics['interest_income_ratio'] = interest_income_ratio
                if interest_income_ratio >= 50:
                    criteria_met.append(f"Interest Income / Revenue = {interest_income_ratio:.1f}% (≥50%)")
            
            # Criterion 2: Interest Expense / Total Expenses ≥ 40%
            # Total Expenses = COGS + Opex + Interest + Depreciation
            total_expenses = []
            for i in range(len(revenues)):
                exp = (financials.get('cogs', [0]*len(revenues))[i] + 
                       financials.get('opex', [0]*len(revenues))[i] + 
                       interest_expenses[i] +
                       financials.get('depreciation', [0]*len(revenues))[i])
                total_expenses.append(exp)
            
            avg_total_expenses = np.mean(total_expenses) if total_expenses else 0
            if avg_total_expenses > 0:
                interest_expense_ratio = (avg_interest_expense / avg_total_expenses) * 100
                metrics['interest_expense_ratio'] = interest_expense_ratio
                if interest_expense_ratio >= 40:
                    criteria_met.append(f"Interest Expense / Total Expenses = {interest_expense_ratio:.1f}% (≥40%)")
        
        # Method 2: Check raw statements for Net Interest Income presence
        if income_stmt is not None:
            nii_fields = ['Net Interest Income', 'Net Interest Margin', 'Interest Income Net']
            for field in nii_fields:
                if field in income_stmt.index:
                    criteria_met.append(f"Presence of '{field}' line item")
                    break
        
        # Method 3: Balance sheet structure check
        if balance_sheet is not None:
            # Check for lending business indicators
            lending_indicators = ['Loans', 'Advances', 'Loans And Advances', 'Net Loans']
            financial_assets = ['Financial Assets', 'Investment Securities']
            
            for indicator in lending_indicators + financial_assets:
                if indicator in balance_sheet.index:
                    # Check if it's a significant portion
                    if 'Total Assets' in balance_sheet.index:
                        try:
                            asset_val = abs(balance_sheet.loc[indicator, balance_sheet.columns[0]])
                            total_assets = abs(balance_sheet.loc['Total Assets', balance_sheet.columns[0]])
                            if total_assets > 0 and (asset_val / total_assets) > 0.5:
                                criteria_met.append(f"Balance Sheet dominated by {indicator} ({asset_val/total_assets*100:.1f}% of assets)")
                        except:
                            pass
    
    except Exception as e:
        st.warning(f"Classification warning: {str(e)}")
    
    # Decision: INTEREST-DOMINANT if 2+ criteria met
    is_interest_dominant = len(criteria_met) >= 2
    
    classification = {
        'type': 'INTEREST_DOMINANT' if is_interest_dominant else 'OPERATING',
        'criteria_met': criteria_met,
        'metrics': metrics
    }
    
    return classification

def validate_fcff_eligibility(classification):
    """
    Check if FCFF DCF is valid per Rulebook Section 3.
    Returns: (is_valid: bool, reason: str)
    """
    if classification['type'] == 'INTEREST_DOMINANT':
        return False, "🚫 FCFF DCF is NOT VALID for Interest-Dominant entities. Debt is operating raw material, not financing."
    
    return True, "✅ FCFF DCF is valid for Operating Companies"

def show_classification_warning(classification):
    """Display business model classification and restrictions"""
    if classification['type'] == 'INTEREST_DOMINANT':
        st.error("""
        🚫 **INTEREST-DOMINANT ENTITY DETECTED**
        
        This company derives significant income from interest operations (lending/banking).
        
        **Why FCFF DCF is Invalid:**
        - Interest expense = Operating Cost (like COGS), not financing cost
        - Interest income = Revenue
        - Debt = Operating raw material (inventory equivalent)
        - EBIT/NOPAT/WACC are economically meaningless
        
        **Criteria Met:**
        """)
        for criterion in classification['criteria_met']:
            st.write(f"  • {criterion}")
        
        st.info("""
        **Recommended Valuation Methods:**
        - ✅ Residual Income Model (preferred)
        - ✅ Dividend Discount Model
        - ✅ P/B with ROE analysis
        - ✅ Relative valuation (P/E, P/B)
        
        ❌ FCFF DCF is blocked to prevent economically invalid valuation.
        """)
        
        return True  # Should stop execution
    
    else:
        st.success(f"""
        ✅ **OPERATING COMPANY CLASSIFICATION**
        
        FCFF DCF valuation is appropriate for this company.
        """)
        
        if classification['criteria_met']:
            with st.expander("ℹ️ Classification Details"):
                st.write("The following interest-related metrics were detected but did not exceed thresholds:")
                for criterion in classification['criteria_met']:
                    st.write(f"  • {criterion}")
        
        return False  # Can continue

# ================================
# BANK VALUATION METHODS
# ================================

def calculate_residual_income_model(financials, shares, cost_of_equity):
    """
    Residual Income Model for banks/financial institutions
    RI = Net Income - (Cost of Equity × Book Value of Equity)
    Value = Book Value + PV(Future Residual Income)
    """
    try:
        # Get latest data
        latest_equity = financials['equity'][-1] * 100000  # Convert from Lacs to Rupees
        
        # Calculate average ROE
        net_incomes = []
        equities = []
        for i in range(len(financials['years'])):
            # Approximate net income from NOPAT (banks don't really have NOPAT, using as proxy)
            ni = financials['nopat'][i] * 100000
            eq = financials['equity'][i] * 100000
            net_incomes.append(ni)
            equities.append(eq)
        
        avg_net_income = np.mean(net_incomes)
        roe = (avg_net_income / latest_equity * 100) if latest_equity > 0 else 15
        
        # Calculate historical book value growth rate
        bv_growth_rates = []
        for i in range(1, len(equities)):
            if equities[i-1] > 0:
                growth = (equities[i] - equities[i-1]) / equities[i-1] * 100
                bv_growth_rates.append(growth)
        
        # Use historical avg or default
        if bv_growth_rates:
            bv_growth = np.mean(bv_growth_rates)
            bv_growth = max(5, min(bv_growth, 20))  # Clamp between 5% and 20%
        else:
            bv_growth = 10.0  # Default
        
        # Terminal growth (typically lower than growth phase)
        terminal_growth = max(3, min(bv_growth * 0.5, 5))  # Half of BV growth, capped at 5%
        
        # Project 5 years of residual income
        projections = []
        current_bv = latest_equity
        current_ni = avg_net_income
        
        for year in range(1, 6):
            # Growth in book value
            current_bv = current_bv * (1 + bv_growth / 100)
            current_ni = current_bv * (roe / 100)
            
            # Residual income = NI - (Ke × BV)
            ri = current_ni - (cost_of_equity / 100 * current_bv)
            
            # Present value
            pv_ri = ri / ((1 + cost_of_equity / 100) ** year)
            projections.append({
                'year': year,
                'book_value': current_bv,
                'net_income': current_ni,
                'residual_income': ri,
                'pv_ri': pv_ri
            })
        
        # Terminal value of residual income
        if cost_of_equity / 100 > terminal_growth / 100:
            terminal_ri = projections[-1]['residual_income'] * (1 + terminal_growth / 100) / (cost_of_equity / 100 - terminal_growth / 100)
            pv_terminal_ri = terminal_ri / ((1 + cost_of_equity / 100) ** 5)
        else:
            pv_terminal_ri = 0
        
        # Total value = Current BV + Sum of PV(RI) + PV(Terminal RI)
        sum_pv_ri = sum([p['pv_ri'] for p in projections])
        total_equity_value = latest_equity + sum_pv_ri + pv_terminal_ri
        
        value_per_share = total_equity_value / shares if shares > 0 else 0
        
        return {
            'method': 'Residual Income Model',
            'current_book_value': latest_equity,
            'roe': roe,
            'bv_growth': bv_growth,
            'terminal_growth': terminal_growth,
            'projections': projections,
            'sum_pv_ri': sum_pv_ri,
            'terminal_ri_pv': pv_terminal_ri,
            'total_equity_value': total_equity_value,
            'value_per_share': value_per_share,
            'historical_bv_growth': bv_growth_rates if bv_growth_rates else None
        }
    except Exception as e:
        st.error(f"RI Model error: {str(e)}")
        return None

def calculate_dividend_discount_model(financials, shares, cost_of_equity, ticker=None):
    """
    Dividend Discount Model (Gordon Growth Model)
    Value = D1 / (Ke - g)
    Attempts to fetch actual dividend data, falls back to estimates
    """
    try:
        # Try to fetch actual dividend data from yfinance
        actual_dividends = []
        div_growth_calculated = None
        payout_ratio_calculated = None
        
        if ticker:
            try:
                stock = yf.Ticker(f"{ticker}.NS")
                dividends_hist = stock.dividends
                
                if not dividends_hist.empty and len(dividends_hist) > 0:
                    # Get annual dividends for last 3 years
                    dividends_by_year = dividends_hist.resample('Y').sum()
                    if len(dividends_by_year) >= 2:
                        recent_divs = dividends_by_year[-3:].values
                        actual_dividends = recent_divs.tolist()
                        
                        # Calculate growth rate from actual dividends
                        if len(actual_dividends) >= 2:
                            growth_rates = []
                            for i in range(1, len(actual_dividends)):
                                if actual_dividends[i-1] > 0:
                                    gr = (actual_dividends[i] - actual_dividends[i-1]) / actual_dividends[i-1] * 100
                                    growth_rates.append(gr)
                            if growth_rates:
                                div_growth_calculated = np.mean(growth_rates)
                                # Clamp between -10% and 20%
                                div_growth_calculated = max(-10, min(div_growth_calculated, 20))
                        
                        # Calculate payout ratio from actual data
                        latest_div = actual_dividends[-1] if actual_dividends else 0
                        latest_ni = financials['nopat'][-1] * 100000
                        if latest_ni > 0 and latest_div > 0:
                            payout_ratio_calculated = (latest_div * shares) / latest_ni
                            payout_ratio_calculated = max(0.1, min(payout_ratio_calculated, 0.9))
            except Exception as e:
                pass
        
        # Calculate average earnings
        net_incomes = []
        for i in range(len(financials['years'])):
            ni = financials['nopat'][i] * 100000
            net_incomes.append(ni)
        
        avg_net_income = np.mean(net_incomes)
        
        # Use calculated or default values
        payout_ratio = payout_ratio_calculated if payout_ratio_calculated else 0.40
        div_growth = div_growth_calculated if div_growth_calculated else 8.0
        
        # Calculate dividends
        total_dividends = avg_net_income * payout_ratio
        dps = total_dividends / shares if shares > 0 else 0
        
        # If we have actual dividends, use the latest as current DPS
        if actual_dividends:
            dps = actual_dividends[-1]
        
        # Next year dividend
        d1 = dps * (1 + div_growth / 100)
        
        # DDM valuation
        if cost_of_equity <= div_growth:
            return None
        
        value_per_share = d1 / ((cost_of_equity - div_growth) / 100)
        
        # 5-year projection
        projections = []
        current_div = dps
        for year in range(1, 6):
            current_div = current_div * (1 + div_growth / 100)
            pv_div = current_div / ((1 + cost_of_equity / 100) ** year)
            projections.append({
                'year': year,
                'dividend': current_div,
                'pv_dividend': pv_div
            })
        
        return {
            'method': 'Dividend Discount Model',
            'current_dps': dps,
            'payout_ratio': payout_ratio * 100,
            'dividend_growth': div_growth,
            'next_year_dps': d1,
            'projections': projections,
            'value_per_share': value_per_share,
            'using_actual_data': bool(actual_dividends),
            'historical_dividends': actual_dividends if actual_dividends else None
        }
    except Exception as e:
        st.error(f"DDM error: {str(e)}")
        return None

def calculate_pb_roe_valuation(financials, shares, cost_of_equity):
    """
    P/B with ROE Analysis
    Fair P/B = ROE / Cost of Equity
    """
    try:
        # Latest book value
        latest_equity = financials['equity'][-1] * 100000
        book_value_per_share = latest_equity / shares if shares > 0 else 0
        
        # Calculate ROE
        net_incomes = []
        equities = []
        for i in range(len(financials['years'])):
            ni = financials['nopat'][i] * 100000
            eq = financials['equity'][i] * 100000
            net_incomes.append(ni)
            equities.append(eq)
        
        avg_net_income = np.mean(net_incomes)
        avg_equity = np.mean(equities)
        roe = (avg_net_income / avg_equity * 100) if avg_equity > 0 else 15
        
        # Fair P/B ratio
        fair_pb = roe / cost_of_equity
        
        # Fair value per share
        value_per_share = book_value_per_share * fair_pb
        
        return {
            'method': 'P/B with ROE Analysis',
            'book_value_per_share': book_value_per_share,
            'roe': roe,
            'cost_of_equity': cost_of_equity,
            'fair_pb_ratio': fair_pb,
            'value_per_share': value_per_share,
            'historical_roe': [(net_incomes[i] / equities[i] * 100) for i in range(len(net_incomes))]
        }
    except Exception as e:
        st.error(f"P/B ROE error: {str(e)}")
        return None

def calculate_relative_valuation(ticker, financials, shares, peer_tickers=None):
    """
    Relative Valuation using peer multiples
    P/E and P/B comparisons with actual peer data
    """
    try:
        if not ticker:
            return None
        
        # Get stock info
        stock = yf.Ticker(f"{ticker}.NS")
        info = stock.info
        current_price = info.get('currentPrice', 0)
        
        # Calculate company metrics
        latest_ni = financials['nopat'][-1] * 100000
        eps = latest_ni / shares if shares > 0 else 0
        
        latest_equity = financials['equity'][-1] * 100000
        bvps = latest_equity / shares if shares > 0 else 0
        
        current_pe = current_price / eps if eps > 0 else 0
        current_pb = current_price / bvps if bvps > 0 else 0
        
        # Fetch peer multiples
        peer_pe_list = []
        peer_pb_list = []
        peer_data = []
        
        # Default bank peers if none provided
        if not peer_tickers:
            # Determine sector based on classification
            peer_tickers = "HDFCBANK,ICICIBANK,SBIN,AXISBANK,KOTAKBANK"
        
        peers = [t.strip() for t in peer_tickers.split(',') if t.strip()]
        
        for peer in peers[:10]:  # Limit to 10 peers
            try:
                peer_stock = yf.Ticker(f"{peer}.NS")
                peer_info = peer_stock.info
                
                peer_pe = peer_info.get('trailingPE', 0)
                peer_pb = peer_info.get('priceToBook', 0)
                peer_price = peer_info.get('currentPrice', 0)
                
                if peer_pe and peer_pe > 0 and peer_pe < 100:  # Sanity check
                    peer_pe_list.append(peer_pe)
                
                if peer_pb and peer_pb > 0 and peer_pb < 20:  # Sanity check
                    peer_pb_list.append(peer_pb)
                
                if peer_price > 0:
                    peer_data.append({
                        'ticker': peer,
                        'price': peer_price,
                        'pe': peer_pe if peer_pe else 'N/A',
                        'pb': peer_pb if peer_pb else 'N/A'
                    })
            except Exception as e:
                continue
        
        # Calculate sector averages
        if peer_pe_list:
            sector_avg_pe = np.median(peer_pe_list)  # Use median to avoid outliers
            sector_low_pe = np.percentile(peer_pe_list, 25)
            sector_high_pe = np.percentile(peer_pe_list, 75)
        else:
            sector_avg_pe = 20  # Fallback
            sector_low_pe = 15
            sector_high_pe = 25
        
        if peer_pb_list:
            sector_avg_pb = np.median(peer_pb_list)
            sector_low_pb = np.percentile(peer_pb_list, 25)
            sector_high_pb = np.percentile(peer_pb_list, 75)
        else:
            sector_avg_pb = 3  # Fallback
            sector_low_pb = 2
            sector_high_pb = 4
        
        # Fair value based on sector multiples
        fair_value_pe = eps * sector_avg_pe
        fair_value_pb = bvps * sector_avg_pb
        
        # Conservative and aggressive estimates
        conservative_value = eps * sector_low_pe
        aggressive_value = eps * sector_high_pe
        
        return {
            'method': 'Relative Valuation',
            'current_price': current_price,
            'eps': eps,
            'bvps': bvps,
            'current_pe': current_pe,
            'current_pb': current_pb,
            'sector_avg_pe': sector_avg_pe,
            'sector_avg_pb': sector_avg_pb,
            'sector_low_pe': sector_low_pe,
            'sector_high_pe': sector_high_pe,
            'sector_low_pb': sector_low_pb,
            'sector_high_pb': sector_high_pb,
            'fair_value_pe_based': fair_value_pe,
            'fair_value_pb_based': fair_value_pb,
            'conservative_value': conservative_value,
            'aggressive_value': aggressive_value,
            'avg_fair_value': (fair_value_pe + fair_value_pb) / 2,
            'peer_count': len(peer_pe_list),
            'peer_data': peer_data
        }
    except Exception as e:
        st.error(f"Relative valuation error: {str(e)}")
        return None

# ================================
# YAHOO FINANCE SCRAPING
# ================================

def fetch_yahoo_financials(ticker):
    """Fetch financial statements from Yahoo Finance using yfinance"""
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get company info
        info = stock.info
        
        if income_stmt.empty or balance_sheet.empty:
            return None, "No financial data available"
        
        # Get shares outstanding
        shares = info.get('sharesOutstanding', 0)
        if shares == 0:
            shares = info.get('impliedSharesOutstanding', 0)
        
        return {
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'info': info,
            'shares': shares
        }, None
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def extract_financials_listed(yahoo_data):
    """Extract financial metrics from Yahoo Finance data"""
    try:
        income_stmt = yahoo_data['income_statement']
        balance_sheet = yahoo_data['balance_sheet']
        cash_flow = yahoo_data['cash_flow']
        
        # Show available fields for debugging
        st.write("**Available Income Statement Fields:**")
        st.write(list(income_stmt.index[:20]))  # Show first 20 fields
        
        # Get last 3 years (columns are sorted newest to oldest)
        years = income_stmt.columns[:3]
        
        opex_methods_used = []  # Track which method was used for each year
        
        financials = {
            'years': [str(y.year) for y in years],
            'revenue': [],
            'cogs': [],
            'opex': [],
            'ebitda': [],
            'depreciation': [],
            'ebit': [],
            'interest': [],
            'interest_income': [],  # Added for business classification
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
        
        for year in years:
            # Income Statement - Values are already in the correct currency
            revenue = abs(income_stmt.loc['Total Revenue', year]) if 'Total Revenue' in income_stmt.index else 0
            cogs = abs(income_stmt.loc['Cost Of Revenue', year]) if 'Cost Of Revenue' in income_stmt.index else 0
            
            # Try to get Operating Expenses directly from various fields
            opex = 0
            opex_method = "None"
            
            # Method 1: Try direct operating expense fields
            opex_fields = [
                'Operating Expense',
                'Total Operating Expenses',
                'Operating Expenses',
                'Selling General And Administration',
                'Selling General Administrative',
                'SG&A Expense'
            ]
            
            for field in opex_fields:
                if field in income_stmt.index:
                    opex = abs(income_stmt.loc[field, year])
                    opex_method = f"Method 1: Direct field '{field}'"
                    break
            
            # Method 2: If not found, try to calculate from Gross Profit - Operating Income
            if opex == 0:
                gross_profit = 0
                if 'Gross Profit' in income_stmt.index:
                    gross_profit = abs(income_stmt.loc['Gross Profit', year])
                elif revenue > 0 and cogs > 0:
                    gross_profit = revenue - cogs
                
                operating_income = abs(income_stmt.loc['Operating Income', year]) if 'Operating Income' in income_stmt.index else 0
                
                if gross_profit > 0 and operating_income > 0:
                    opex = gross_profit - operating_income
                    opex_method = "Method 2: Gross Profit - Operating Income"
            
            # Method 3: If still not found, try SG&A + R&D + Other
            if opex == 0:
                sga = abs(income_stmt.loc['Selling General And Administration', year]) if 'Selling General And Administration' in income_stmt.index else 0
                rd = abs(income_stmt.loc['Research And Development', year]) if 'Research And Development' in income_stmt.index else 0
                other_opex = abs(income_stmt.loc['Other Operating Expenses', year]) if 'Other Operating Expenses' in income_stmt.index else 0
                opex = sga + rd + other_opex
                if opex > 0:
                    opex_method = f"Method 3: SG&A({sga/100000:.2f}) + R&D({rd/100000:.2f}) + Other({other_opex/100000:.2f})"
            
            # Get EBITDA
            if 'EBITDA' in income_stmt.index:
                ebitda = abs(income_stmt.loc['EBITDA', year])
            elif 'Normalized EBITDA' in income_stmt.index:
                ebitda = abs(income_stmt.loc['Normalized EBITDA', year])
            else:
                # Calculate: EBITDA = Revenue - COGS - Opex
                ebitda = revenue - cogs - opex
            
            # Get depreciation separately for projections
            if 'Reconciled Depreciation' in income_stmt.index:
                depreciation = abs(income_stmt.loc['Reconciled Depreciation', year])
            elif 'Depreciation And Amortization' in cash_flow.index:
                depreciation = abs(cash_flow.loc['Depreciation And Amortization', year])
            elif 'Depreciation' in income_stmt.index:
                depreciation = abs(income_stmt.loc['Depreciation', year])
            else:
                # Calculate from Operating Income vs EBITDA
                operating_income = abs(income_stmt.loc['Operating Income', year]) if 'Operating Income' in income_stmt.index else 0
                if ebitda > operating_income:
                    depreciation = ebitda - operating_income
                else:
                    depreciation = revenue * 0.02  # Assume 2% of revenue if not available
            
            # Final safety check: If opex is still 0 or unreasonable, derive from EBITDA
            if opex == 0 or opex < 0:
                opex = revenue - cogs - ebitda
                opex_method = "Method 4 (Fallback): Revenue - COGS - EBITDA"
                if opex < 0:
                    opex = revenue * 0.15  # Assume 15% of revenue as default
                    opex_method = "Method 5 (Default): 15% of Revenue"
            
            # EBIT
            ebit = ebitda - depreciation
            
            # Interest Expense
            interest = 0
            if 'Interest Expense' in income_stmt.index:
                interest = abs(income_stmt.loc['Interest Expense', year])
            elif 'Interest Expense Non Operating' in income_stmt.index:
                interest = abs(income_stmt.loc['Interest Expense Non Operating', year])
            elif 'Net Interest Income' in income_stmt.index:
                # For banks, net interest income is revenue, not expense
                net_int = income_stmt.loc['Net Interest Income', year]
                if net_int < 0:
                    interest = abs(net_int)
            
            # Interest Income (for business classification)
            interest_income = 0
            if 'Interest Income' in income_stmt.index:
                interest_income = abs(income_stmt.loc['Interest Income', year])
            elif 'Interest And Dividend Income' in income_stmt.index:
                interest_income = abs(income_stmt.loc['Interest And Dividend Income', year])
            elif 'Net Interest Income' in income_stmt.index:
                # For banks, this is the primary revenue
                net_int = income_stmt.loc['Net Interest Income', year]
                if net_int > 0:
                    interest_income = abs(net_int)
            
            # Tax
            tax = abs(income_stmt.loc['Tax Provision', year]) if 'Tax Provision' in income_stmt.index else 0
            
            # NOPAT (using 25% tax as default)
            tax_rate_effective = (tax / (ebit - interest)) if (ebit - interest) > 0 else 0.25
            tax_rate_effective = min(max(tax_rate_effective, 0), 0.35)  # Clamp between 0 and 35%
            nopat = ebit * (1 - tax_rate_effective)
            
            # Balance Sheet - Values are already in the correct currency
            total_assets = abs(balance_sheet.loc['Total Assets', year]) if 'Total Assets' in balance_sheet.index else 0
            
            # Fixed Assets
            if 'Net PPE' in balance_sheet.index:
                fixed_assets = abs(balance_sheet.loc['Net PPE', year])
            elif 'Gross PPE' in balance_sheet.index:
                fixed_assets = abs(balance_sheet.loc['Gross PPE', year])
            elif 'Properties' in balance_sheet.index:
                fixed_assets = abs(balance_sheet.loc['Properties', year])
            else:
                fixed_assets = total_assets * 0.3  # Assume 30% of total assets
            
            # Current Assets
            inventory = abs(balance_sheet.loc['Inventory', year]) if 'Inventory' in balance_sheet.index else 0
            
            receivables = 0
            if 'Receivables' in balance_sheet.index:
                receivables = abs(balance_sheet.loc['Receivables', year])
            elif 'Accounts Receivable' in balance_sheet.index:
                receivables = abs(balance_sheet.loc['Accounts Receivable', year])
            elif 'Gross Accounts Receivable' in balance_sheet.index:
                receivables = abs(balance_sheet.loc['Gross Accounts Receivable', year])
            
            cash = 0
            if 'Cash And Cash Equivalents' in balance_sheet.index:
                cash = abs(balance_sheet.loc['Cash And Cash Equivalents', year])
            elif 'Cash Cash Equivalents And Short Term Investments' in balance_sheet.index:
                cash = abs(balance_sheet.loc['Cash Cash Equivalents And Short Term Investments', year])
            
            # Liabilities
            payables = 0
            if 'Payables' in balance_sheet.index:
                payables = abs(balance_sheet.loc['Payables', year])
            elif 'Accounts Payable' in balance_sheet.index:
                payables = abs(balance_sheet.loc['Accounts Payable', year])
            elif 'Payables And Accrued Expenses' in balance_sheet.index:
                payables = abs(balance_sheet.loc['Payables And Accrued Expenses', year])
            
            # Debt
            st_debt = 0
            if 'Current Debt' in balance_sheet.index:
                st_debt = abs(balance_sheet.loc['Current Debt', year])
            elif 'Current Debt And Capital Lease Obligation' in balance_sheet.index:
                st_debt = abs(balance_sheet.loc['Current Debt And Capital Lease Obligation', year])
            
            lt_debt = 0
            if 'Long Term Debt' in balance_sheet.index:
                lt_debt = abs(balance_sheet.loc['Long Term Debt', year])
            elif 'Long Term Debt And Capital Lease Obligation' in balance_sheet.index:
                lt_debt = abs(balance_sheet.loc['Long Term Debt And Capital Lease Obligation', year])
            
            # Equity
            equity = 0
            if 'Stockholders Equity' in balance_sheet.index:
                equity = abs(balance_sheet.loc['Stockholders Equity', year])
            elif 'Total Equity Gross Minority Interest' in balance_sheet.index:
                equity = abs(balance_sheet.loc['Total Equity Gross Minority Interest', year])
            elif 'Common Stock Equity' in balance_sheet.index:
                equity = abs(balance_sheet.loc['Common Stock Equity', year])
            
            # Convert to Lacs (divide by 100,000)
            # Yahoo Finance data is in actual currency (Rupees for Indian stocks)
            financials['revenue'].append(revenue / 100000)
            financials['cogs'].append(cogs / 100000)
            financials['opex'].append(opex / 100000)
            financials['ebitda'].append(ebitda / 100000)
            financials['depreciation'].append(depreciation / 100000)
            financials['ebit'].append(ebit / 100000)
            financials['interest'].append(interest / 100000)
            financials['interest_income'].append(interest_income / 100000)  # For business classification
            financials['tax'].append(tax / 100000)
            financials['nopat'].append(nopat / 100000)
            
            financials['fixed_assets'].append(fixed_assets / 100000)
            financials['inventory'].append(inventory / 100000)
            financials['receivables'].append(receivables / 100000)
            financials['payables'].append(payables / 100000)
            financials['cash'].append(cash / 100000)
            financials['equity'].append(equity / 100000)
            financials['st_debt'].append(st_debt / 100000)
            financials['lt_debt'].append(lt_debt / 100000)
            
            # Track which method was used for opex
            opex_methods_used.append(f"{year.year}: {opex_method}")
        
        # Debug output
        st.write("**Debug: Extracted Financials (in Lacs)**")
        st.write(f"Revenue: {financials['revenue']}")
        st.write(f"COGS: {financials['cogs']}")
        st.write(f"Opex: {financials['opex']}")
        st.write(f"**Opex Extraction Methods:**")
        for method in opex_methods_used:
            st.write(f"  - {method}")
        st.write(f"EBITDA: {financials['ebitda']}")
        st.write(f"EBIT: {financials['ebit']}")
        st.write(f"Equity: {financials['equity']}")
        st.write(f"Debt: {[financials['st_debt'][i] + financials['lt_debt'][i] for i in range(len(financials['st_debt']))]}")
        
        return financials
        
    except Exception as e:
        st.error(f"Error extracting financials: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def get_stock_beta(ticker, market_ticker='^BSESN', period_years=3):
    """Calculate beta using regression of stock returns vs market returns"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_years*365)
        
        # Download stock data
        stock = yf.download(f"{ticker}.NS", start=start_date, end=end_date, progress=False)
        market = yf.download(market_ticker, start=start_date, end=end_date, progress=False)
        
        if stock.empty or market.empty:
            return 1.0
        
        # Calculate returns
        stock_returns = stock['Close'].pct_change().dropna()
        market_returns = market['Close'].pct_change().dropna()
        
        # Align data
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner')
        aligned.columns = ['stock', 'market']
        aligned = aligned.dropna()
        
        if len(aligned) < 2:
            return 1.0
        
        # Calculate beta using covariance method
        covariance = aligned['stock'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return max(0.1, min(beta, 3.0))  # Clamp between 0.1 and 3.0
        
    except Exception as e:
        st.warning(f"Could not calculate beta for {ticker}: {str(e)}")
        return 1.0

def get_risk_free_rate():
    """Get risk-free rate from government bond yields"""
    try:
        # Fetch 10-year India G-Sec yield from yfinance
        gsec = yf.Ticker("^TNX")  # Using 10-year treasury as proxy
        info = gsec.info
        if 'previousClose' in info:
            return info['previousClose']
    except:
        pass
    
    # Fallback to 7% for Indian G-Sec
    return 7.0

def get_market_return():
    """Calculate market return from Sensex historical data"""
    try:
        end_date = datetime.now()
        start_date = datetime.now() - timedelta(days=20*365)  # 20 years for better data
        
        sensex = yf.download('^BSESN', start=start_date, end=end_date, progress=False)
        
        if not sensex.empty and len(sensex) > 252:  # At least 1 year of data
            # Calculate CAGR
            start_price = float(sensex['Close'].iloc[0])
            end_price = float(sensex['Close'].iloc[-1])
            num_years = len(sensex) / 252  # 252 trading days per year
            
            if start_price > 0 and num_years > 0:
                cagr = ((end_price / start_price) ** (1 / num_years) - 1) * 100
                st.info(f"📊 Sensex CAGR (last {num_years:.1f} years): {cagr:.2f}%")
                return max(8.0, min(cagr, 25.0))  # Clamp between 8% and 25%
    except Exception as e:
        st.warning(f"Could not fetch Sensex data: {str(e)}")
    
    # Fallback
    st.warning("⚠️ Using fallback market return of 12%")
    return 12.0

# ================================
# ADVANCED CHARTING FUNCTIONS
# ================================

def create_waterfall_chart(valuation):
    """Create waterfall chart showing DCF value buildup"""
    fig = go.Figure(go.Waterfall(
        name = "DCF Waterfall",
        orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["PV of Projected FCFF", "PV of Terminal Value", "Enterprise Value"],
        textposition = "outside",
        text = [f"₹{valuation['sum_pv_fcff']:.2f}L", 
                f"₹{valuation['pv_terminal_value']:.2f}L",
                f"₹{valuation['enterprise_value']:.2f}L"],
        y = [valuation['sum_pv_fcff'], valuation['pv_terminal_value'], 0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title = "DCF Valuation Waterfall",
        showlegend = False,
        height = 500,
        yaxis_title="Value (₹ Lacs)"
    )
    
    return fig

def create_fcff_projection_chart(projections):
    """Create detailed FCFF projection chart with components"""
    years = [f"Year {y}" for y in projections['year']]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('FCFF Projection', 'Revenue & EBITDA', 'Working Capital Changes', 'Capex & Depreciation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # FCFF Projection
    fig.add_trace(
        go.Bar(name='FCFF', x=years, y=projections['fcff'], marker_color='#2E86AB'),
        row=1, col=1
    )
    
    # Revenue & EBITDA
    fig.add_trace(
        go.Scatter(name='Revenue', x=years, y=projections['revenue'], mode='lines+markers', line=dict(color='#06A77D', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(name='EBITDA', x=years, y=projections['ebitda'], mode='lines+markers', line=dict(color='#F77F00', width=3)),
        row=1, col=2
    )
    
    # Working Capital
    fig.add_trace(
        go.Bar(name='Δ WC', x=years, y=projections['delta_wc'], marker_color='#D62828'),
        row=2, col=1
    )
    
    # Capex & Depreciation
    fig.add_trace(
        go.Bar(name='Capex', x=years, y=projections['capex'], marker_color='#F77F00'),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(name='Depreciation', x=years, y=projections['depreciation'], marker_color='#06A77D'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Financial Projections")
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="₹ Lacs", row=1, col=1)
    fig.update_yaxes(title_text="₹ Lacs", row=1, col=2)
    fig.update_yaxes(title_text="₹ Lacs", row=2, col=1)
    fig.update_yaxes(title_text="₹ Lacs", row=2, col=2)
    
    return fig

def create_sensitivity_heatmap(projections, wacc_range, g_range, num_shares):
    """Create sensitivity analysis heatmap"""
    last_fcff = projections['fcff'][-1]
    n = len(projections['fcff'])
    
    # Create matrix
    matrix = []
    for w in wacc_range:
        row = []
        for g in g_range:
            if g >= w - 0.1:
                row.append(None)
            else:
                try:
                    fcff_n1 = last_fcff * (1 + g/100)
                    tv = fcff_n1 / ((w/100) - (g/100))
                    pv_tv = tv / ((1 + w/100) ** n)
                    
                    # Calculate sum_pv_fcff (approximate from first calc)
                    sum_pv_fcff = sum([projections['fcff'][i] / ((1 + w/100) ** (i+1)) for i in range(len(projections['fcff']))])
                    
                    ev = sum_pv_fcff + pv_tv
                    eq_val = ev * 100000
                    fv = eq_val / num_shares if num_shares > 0 else 0
                    row.append(fv)
                except:
                    row.append(None)
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"{g:.1f}%" for g in g_range],
        y=[f"{w:.1f}%" for w in wacc_range],
        colorscale='RdYlGn',
        text=[[f"₹{val:.1f}" if val else "N/A" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size":10},
        colorbar=dict(title="Fair Value ₹")
    ))
    
    fig.update_layout(
        title='Sensitivity Analysis: Fair Value per Share',
        xaxis_title='Terminal Growth Rate (g)',
        yaxis_title='WACC',
        height=600
    )
    
    return fig

def create_historical_financials_chart(financials):
    """Create comprehensive historical financials overview"""
    years = financials['years']
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Revenue & EBITDA Trend', 'Profitability Margins', 
                       'Balance Sheet Health', 'Cash Flow Quality',
                       'Working Capital Efficiency', 'Leverage Ratios'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue & EBITDA
    fig.add_trace(
        go.Bar(name='Revenue', x=years, y=financials['revenue'], marker_color='#06A77D'),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(name='EBITDA Margin %', x=years, 
                  y=[(financials['ebitda'][i]/financials['revenue'][i]*100) if financials['revenue'][i] > 0 else 0 
                     for i in range(len(years))],
                  mode='lines+markers', line=dict(color='#F77F00', width=3), marker=dict(size=10)),
        row=1, col=1, secondary_y=True
    )
    
    # Profitability Margins
    ebitda_margins = [(financials['ebitda'][i]/financials['revenue'][i]*100) if financials['revenue'][i] > 0 else 0 
                      for i in range(len(years))]
    ebit_margins = [(financials['ebit'][i]/financials['revenue'][i]*100) if financials['revenue'][i] > 0 else 0 
                    for i in range(len(years))]
    
    fig.add_trace(
        go.Scatter(name='EBITDA Margin', x=years, y=ebitda_margins, mode='lines+markers', line=dict(width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(name='EBIT Margin', x=years, y=ebit_margins, mode='lines+markers', line=dict(width=3)),
        row=1, col=2
    )
    
    # Balance Sheet
    fig.add_trace(
        go.Bar(name='Equity', x=years, y=financials['equity'], marker_color='#06A77D'),
        row=2, col=1, secondary_y=False
    )
    total_debt = [financials['st_debt'][i] + financials['lt_debt'][i] for i in range(len(years))]
    fig.add_trace(
        go.Bar(name='Debt', x=years, y=total_debt, marker_color='#D62828'),
        row=2, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(name='Debt/Equity', x=years,
                  y=[(total_debt[i]/financials['equity'][i]) if financials['equity'][i] > 0 else 0 
                     for i in range(len(years))],
                  mode='lines+markers', line=dict(color='#2E86AB', width=3)),
        row=2, col=1, secondary_y=True
    )
    
    # Cash Flow Quality (NOPAT vs EBIT)
    fig.add_trace(
        go.Bar(name='EBIT', x=years, y=financials['ebit'], marker_color='#F77F00'),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(name='NOPAT', x=years, y=financials['nopat'], marker_color='#06A77D'),
        row=2, col=2
    )
    
    # Working Capital Components
    fig.add_trace(
        go.Bar(name='Inventory', x=years, y=financials['inventory'], marker_color='#2E86AB'),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(name='Receivables', x=years, y=financials['receivables'], marker_color='#06A77D'),
        row=3, col=1
    )
    fig.add_trace(
        go.Bar(name='Payables', x=years, y=financials['payables'], marker_color='#D62828'),
        row=3, col=1
    )
    
    # Leverage Ratios
    debt_to_ebitda = [(total_debt[i]/financials['ebitda'][i]) if financials['ebitda'][i] > 0 else 0 
                      for i in range(len(years))]
    interest_coverage = [(financials['ebit'][i]/financials['interest'][i]) if financials['interest'][i] > 0 else 0 
                         for i in range(len(years))]
    
    fig.add_trace(
        go.Scatter(name='Debt/EBITDA', x=years, y=debt_to_ebitda, mode='lines+markers', line=dict(width=3)),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(name='Interest Coverage', x=years, y=interest_coverage, mode='lines+markers', line=dict(width=3)),
        row=3, col=2
    )
    
    fig.update_layout(height=1200, showlegend=True, title_text="Historical Financial Analysis Dashboard")
    fig.update_yaxes(title_text="₹ Lacs", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Margin %", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Margin %", row=1, col=2)
    fig.update_yaxes(title_text="₹ Lacs", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Ratio", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="₹ Lacs", row=2, col=2)
    fig.update_yaxes(title_text="₹ Lacs", row=3, col=1)
    fig.update_yaxes(title_text="Ratio", row=3, col=2)
    
    return fig

def create_wacc_breakdown_chart(wacc_details):
    """Create visual breakdown of WACC components"""
    labels = ['Cost of Equity (Ke)', 'After-tax Cost of Debt (Kd)']
    values = [wacc_details['ke'], wacc_details['kd_after_tax']]
    weights = [wacc_details['we'], wacc_details['wd']]
    contributions = [wacc_details['ke'] * wacc_details['we'] / 100, 
                    wacc_details['kd_after_tax'] * wacc_details['wd'] / 100]
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=('Capital Structure Weights', 'WACC Components Contribution')
    )
    
    # Capital structure pie
    fig.add_trace(
        go.Pie(labels=['Equity', 'Debt'], values=[wacc_details['we'], wacc_details['wd']],
               marker_colors=['#06A77D', '#D62828']),
        row=1, col=1
    )
    
    # WACC contribution bar
    fig.add_trace(
        go.Bar(name='Contribution to WACC', x=labels, y=contributions,
               marker_color=['#06A77D', '#D62828'],
               text=[f"{c:.2f}%" for c in contributions],
               textposition='auto'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True, title_text=f"WACC Breakdown (Total: {wacc_details['wacc']:.2f}%)")
    
    return fig

def create_bank_valuation_comparison_chart(valuations_dict):
    """Create comparison chart for multiple bank valuation methods"""
    methods = []
    values = []
    colors = []
    
    color_map = {
        'Residual Income Model': '#2E86AB',
        'Dividend Discount Model': '#06A77D',
        'P/B with ROE Analysis': '#F77F00',
        'Relative Valuation (P/E)': '#D62828',
        'Relative Valuation (P/B)': '#9D4EDD'
    }
    
    for method, val_data in valuations_dict.items():
        if val_data and 'value_per_share' in val_data:
            methods.append(method)
            values.append(val_data['value_per_share'])
            colors.append(color_map.get(method, '#888888'))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        marker_color=colors,
        text=[f"₹{v:.2f}" for v in values],
        textposition='auto',
    ))
    
    if values:
        avg_value = np.mean(values)
        fig.add_hline(y=avg_value, line_dash="dash", line_color="red",
                     annotation_text=f"Average: ₹{avg_value:.2f}",
                     annotation_position="right")
    
    fig.update_layout(
        title="Bank Valuation Methods Comparison",
        xaxis_title="Valuation Method",
        yaxis_title="Fair Value per Share (₹)",
        height=500,
        showlegend=False
    )
    
    return fig

def create_price_vs_value_gauge(current_price, fair_value):
    """Create gauge chart showing current price vs fair value"""
    if fair_value == 0:
        return None
    
    ratio = (current_price / fair_value) * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_price,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Current Price vs Fair Value (₹{fair_value:.2f})", 'font': {'size': 20}},
        delta = {'reference': fair_value, 'valueformat': '.2f'},
        gauge = {
            'axis': {'range': [None, max(current_price, fair_value) * 1.5], 'tickformat': '₹.2f'},
            'bar': {'color': "#2E86AB"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, fair_value * 0.8], 'color': '#06A77D'},
                {'range': [fair_value * 0.8, fair_value * 1.2], 'color': '#F4D35E'},
                {'range': [fair_value * 1.2, fair_value * 2], 'color': '#D62828'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': fair_value}}))
    
    fig.update_layout(height=400)
    
    if ratio < 80:
        recommendation = "🟢 UNDERVALUED - Potential Buy"
    elif ratio > 120:
        recommendation = "🔴 OVERVALUED - Potential Sell"
    else:
        recommendation = "🟡 FAIRLY VALUED - Hold"
    
    fig.add_annotation(
        text=recommendation,
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=16, color="black", family="Arial Black")
    )
    
    return fig

# ================================
# DCF CALCULATION FUNCTIONS
# ================================

def extract_financials_unlisted(df_bs, df_pl, year_cols):
    """Extract financial metrics from XML DataFrames"""
    num_years = min(3, len(year_cols))
    last_years = year_cols[-num_years:]
    
    financials = {
        'years': last_years,
        'revenue': [],
        'cogs': [],
        'opex': [],
        'ebitda': [],
        'depreciation': [],
        'ebit': [],
        'interest': [],
        'interest_income': [],  # Added for business classification
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
    
    for year_col in last_years:
        # Income Statement
        revenue = get_value_from_df(df_pl, 'Net Revenue', year_col)
        cogs = get_value_from_df(df_pl, 'Cost of Materials', year_col)
        employee_exp = get_value_from_df(df_pl, 'Employee Benefit', year_col)
        other_exp = get_value_from_df(df_pl, 'Other Expenses', year_col)
        depreciation = get_value_from_df(df_pl, 'Depreciation', year_col)
        interest = get_value_from_df(df_pl, 'Finance Costs', year_col)
        interest_income = get_value_from_df(df_pl, 'Finance Income', year_col)  # For classification
        tax = get_value_from_df(df_pl, 'Income Tax', year_col)
        
        opex = employee_exp + other_exp
        ebitda = revenue - opex - cogs
        ebit = ebitda - depreciation
        pbt = ebit - interest
        pat = pbt - tax
        nopat = ebit * (1 - 0.25)  # Assuming 25% tax
        
        financials['revenue'].append(revenue)
        financials['cogs'].append(cogs)
        financials['opex'].append(opex)
        financials['ebitda'].append(ebitda)
        financials['depreciation'].append(depreciation)
        financials['ebit'].append(ebit)
        financials['interest'].append(interest)
        financials['interest_income'].append(interest_income)  # Store for classification
        financials['tax'].append(tax)
        financials['nopat'].append(nopat)
        
        # Balance Sheet
        fixed_assets = get_value_from_df(df_bs, 'Tangible Assets', year_col)
        inventory = get_value_from_df(df_bs, 'Inventories', year_col)
        receivables = get_value_from_df(df_bs, 'Trade Receivables', year_col)
        payables = get_value_from_df(df_bs, 'Trade Payables', year_col)
        cash = get_value_from_df(df_bs, 'Cash and Bank', year_col)
        equity = get_value_from_df(df_bs, 'Total Equity', year_col)
        st_debt = get_value_from_df(df_bs, 'Short Term Borrowings', year_col)
        lt_debt = get_value_from_df(df_bs, 'Long Term Borrowings', year_col)
        
        financials['fixed_assets'].append(fixed_assets)
        financials['inventory'].append(inventory)
        financials['receivables'].append(receivables)
        financials['payables'].append(payables)
        financials['cash'].append(cash)
        financials['equity'].append(equity)
        financials['st_debt'].append(st_debt)
        financials['lt_debt'].append(lt_debt)
    
    return financials

def perform_comparative_valuation(target_ticker, comp_tickers_str, target_financials=None, target_shares=None):
    """Perform comparative valuation using peer multiples"""
    try:
        comp_tickers = [t.strip() for t in comp_tickers_str.split(',') if t.strip()]
        
        if not comp_tickers:
            return None
        
        results = {
            'target': {},
            'comparables': [],
            'multiples_stats': {},
            'valuations': {}
        }
        
        # Get target company data
        if target_ticker:
            # Listed company
            target_stock = yf.Ticker(f"{target_ticker}.NS")
            target_info = target_stock.info
            target_financials_yf = target_stock.financials
            target_bs = target_stock.balance_sheet
            
            results['target'] = {
                'name': target_info.get('longName', target_ticker),
                'current_price': target_info.get('currentPrice', 0),
                'shares': target_info.get('sharesOutstanding', 0),
                'market_cap': target_info.get('marketCap', 0),
                'enterprise_value': target_info.get('enterpriseValue', 0),
                'revenue': abs(target_financials_yf.loc['Total Revenue', target_financials_yf.columns[0]]) if 'Total Revenue' in target_financials_yf.index else 0,
                'ebitda': target_info.get('ebitda', 0),
                'net_income': abs(target_financials_yf.loc['Net Income', target_financials_yf.columns[0]]) if 'Net Income' in target_financials_yf.index else 0,
                'book_value_per_share': target_info.get('bookValue', 0),
                'total_debt': target_bs.loc['Long Term Debt', target_bs.columns[0]] if 'Long Term Debt' in target_bs.index else 0,
                'cash': target_bs.loc['Cash And Cash Equivalents', target_bs.columns[0]] if 'Cash And Cash Equivalents' in target_bs.index else 0,
            }
            
            # Calculate EPS and Book Value
            if results['target']['shares'] > 0:
                results['target']['eps'] = results['target']['net_income'] / results['target']['shares']
                
        else:
            # Unlisted company
            results['target'] = {
                'name': 'Target Company (Unlisted)',
                'current_price': 0,
                'shares': target_shares,
                'market_cap': 0,
                'enterprise_value': 0,
                'revenue': target_financials['revenue'][-1] * 100000,  # Convert from Lacs
                'ebitda': target_financials['ebitda'][-1] * 100000,
                'net_income': target_financials['nopat'][-1] * 100000,  # Using NOPAT as proxy
                'book_value_per_share': 0,
                'total_debt': (target_financials['st_debt'][-1] + target_financials['lt_debt'][-1]) * 100000,
                'cash': target_financials['cash'][-1] * 100000,
                'eps': (target_financials['nopat'][-1] * 100000) / target_shares if target_shares > 0 else 0,
            }
        
        # Get comparable companies data
        comp_data = []
        for ticker in comp_tickers:
            try:
                comp_stock = yf.Ticker(f"{ticker}.NS" if not ticker.endswith('.NS') else ticker)
                comp_info = comp_stock.info
                comp_financials_yf = comp_stock.financials
                comp_bs = comp_stock.balance_sheet
                
                # Extract financial data
                shares = comp_info.get('sharesOutstanding', 0)
                price = comp_info.get('currentPrice', 0)
                market_cap = comp_info.get('marketCap', 0)
                
                revenue = abs(comp_financials_yf.loc['Total Revenue', comp_financials_yf.columns[0]]) if 'Total Revenue' in comp_financials_yf.index and not comp_financials_yf.empty else 0
                ebitda = comp_info.get('ebitda', 0)
                net_income = abs(comp_financials_yf.loc['Net Income', comp_financials_yf.columns[0]]) if 'Net Income' in comp_financials_yf.index and not comp_financials_yf.empty else 0
                
                total_debt = abs(comp_bs.loc['Long Term Debt', comp_bs.columns[0]]) if 'Long Term Debt' in comp_bs.index and not comp_bs.empty else 0
                cash = abs(comp_bs.loc['Cash And Cash Equivalents', comp_bs.columns[0]]) if 'Cash And Cash Equivalents' in comp_bs.index and not comp_bs.empty else 0
                
                book_value = comp_info.get('bookValue', 0)
                eps = net_income / shares if shares > 0 else 0
                
                # Calculate multiples
                pe = price / eps if eps > 0 else 0
                pb = price / book_value if book_value > 0 else 0
                ps = market_cap / revenue if revenue > 0 else 0
                
                enterprise_value = market_cap + total_debt - cash
                ev_ebitda = enterprise_value / ebitda if ebitda > 0 else 0
                ev_sales = enterprise_value / revenue if revenue > 0 else 0
                
                comp_data.append({
                    'ticker': ticker,
                    'name': comp_info.get('longName', ticker),
                    'price': price,
                    'market_cap': market_cap,
                    'revenue': revenue,
                    'ebitda': ebitda,
                    'net_income': net_income,
                    'eps': eps,
                    'book_value': book_value,
                    'pe': pe,
                    'pb': pb,
                    'ps': ps,
                    'ev_ebitda': ev_ebitda,
                    'ev_sales': ev_sales,
                    'enterprise_value': enterprise_value,
                    'shares': shares
                })
                
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
                continue
        
        results['comparables'] = comp_data
        
        if not comp_data:
            st.error("No comparable company data could be fetched")
            return None
        
        # Calculate statistics for each multiple
        multiples = ['pe', 'pb', 'ps', 'ev_ebitda', 'ev_sales']
        
        for multiple in multiples:
            valid_values = [c[multiple] for c in comp_data if c.get(multiple, 0) > 0]
            
            if not valid_values:
                continue
            
            results['multiples_stats'][multiple] = {
                'average': np.mean(valid_values),
                'median': np.median(valid_values),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'std': np.std(valid_values),
                'values': valid_values
            }
        
        # Calculate implied valuations
        target = results['target']
        valuations_summary = {}
        
        # P/E Method
        if 'pe' in results['multiples_stats'] and target['eps'] > 0:
            stats = results['multiples_stats']['pe']
            
            fair_value_avg = target['eps'] * stats['average']
            fair_value_median = target['eps'] * stats['median']
            
            valuations_summary['pe'] = {
                'method': 'Price-to-Earnings (P/E)',
                'target_metric': target['eps'],
                'metric_name': 'EPS',
                'avg_multiple': stats['average'],
                'median_multiple': stats['median'],
                'fair_value_avg': fair_value_avg,
                'fair_value_median': fair_value_median,
                'current_price': target['current_price'],
                'upside_avg': ((fair_value_avg - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'upside_median': ((fair_value_median - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'formula_avg': f"EPS × Avg P/E = ₹{target['eps']:.2f} × {stats['average']:.2f} = ₹{fair_value_avg:.2f}",
                'formula_median': f"EPS × Median P/E = ₹{target['eps']:.2f} × {stats['median']:.2f} = ₹{fair_value_median:.2f}"
            }
        
        # P/B Method
        if 'pb' in results['multiples_stats'] and target['book_value_per_share'] > 0:
            stats = results['multiples_stats']['pb']
            
            fair_value_avg = target['book_value_per_share'] * stats['average']
            fair_value_median = target['book_value_per_share'] * stats['median']
            
            valuations_summary['pb'] = {
                'method': 'Price-to-Book (P/B)',
                'target_metric': target['book_value_per_share'],
                'metric_name': 'Book Value per Share',
                'avg_multiple': stats['average'],
                'median_multiple': stats['median'],
                'fair_value_avg': fair_value_avg,
                'fair_value_median': fair_value_median,
                'current_price': target['current_price'],
                'upside_avg': ((fair_value_avg - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'upside_median': ((fair_value_median - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'formula_avg': f"BVPS × Avg P/B = ₹{target['book_value_per_share']:.2f} × {stats['average']:.2f} = ₹{fair_value_avg:.2f}",
                'formula_median': f"BVPS × Median P/B = ₹{target['book_value_per_share']:.2f} × {stats['median']:.2f} = ₹{fair_value_median:.2f}"
            }
        
        # P/S Method
        if 'ps' in results['multiples_stats'] and target['revenue'] > 0 and target['shares'] > 0:
            stats = results['multiples_stats']['ps']
            
            revenue_per_share = target['revenue'] / target['shares']
            fair_value_avg = revenue_per_share * stats['average']
            fair_value_median = revenue_per_share * stats['median']
            
            valuations_summary['ps'] = {
                'method': 'Price-to-Sales (P/S)',
                'target_metric': revenue_per_share,
                'metric_name': 'Revenue per Share',
                'avg_multiple': stats['average'],
                'median_multiple': stats['median'],
                'fair_value_avg': fair_value_avg,
                'fair_value_median': fair_value_median,
                'current_price': target['current_price'],
                'upside_avg': ((fair_value_avg - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'upside_median': ((fair_value_median - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'formula_avg': f"Revenue/Share × Avg P/S = ₹{revenue_per_share:.2f} × {stats['average']:.2f} = ₹{fair_value_avg:.2f}",
                'formula_median': f"Revenue/Share × Median P/S = ₹{revenue_per_share:.2f} × {stats['median']:.2f} = ₹{fair_value_median:.2f}"
            }
        
        # EV/EBITDA Method
        if 'ev_ebitda' in results['multiples_stats'] and target['ebitda'] > 0 and target['shares'] > 0:
            stats = results['multiples_stats']['ev_ebitda']
            
            implied_ev_avg = target['ebitda'] * stats['average']
            implied_ev_median = target['ebitda'] * stats['median']
            
            net_debt = target['total_debt'] - target['cash']
            
            equity_value_avg = implied_ev_avg - net_debt
            equity_value_median = implied_ev_median - net_debt
            
            fair_value_avg = equity_value_avg / target['shares']
            fair_value_median = equity_value_median / target['shares']
            
            valuations_summary['ev_ebitda'] = {
                'method': 'EV/EBITDA',
                'target_metric': target['ebitda'],
                'metric_name': 'EBITDA',
                'avg_multiple': stats['average'],
                'median_multiple': stats['median'],
                'implied_ev_avg': implied_ev_avg,
                'implied_ev_median': implied_ev_median,
                'net_debt': net_debt,
                'fair_value_avg': fair_value_avg,
                'fair_value_median': fair_value_median,
                'current_price': target['current_price'],
                'upside_avg': ((fair_value_avg - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'upside_median': ((fair_value_median - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'formula_avg': f"(EBITDA × Avg EV/EBITDA - Net Debt) / Shares = (₹{target['ebitda']/1e7:.2f}Cr × {stats['average']:.2f} - ₹{net_debt/1e7:.2f}Cr) / {target['shares']/1e7:.2f}Cr",
                'formula_median': f"(EBITDA × Median EV/EBITDA - Net Debt) / Shares = (₹{target['ebitda']/1e7:.2f}Cr × {stats['median']:.2f} - ₹{net_debt/1e7:.2f}Cr) / {target['shares']/1e7:.2f}Cr"
            }
        
        # EV/Sales Method
        if 'ev_sales' in results['multiples_stats'] and target['revenue'] > 0 and target['shares'] > 0:
            stats = results['multiples_stats']['ev_sales']
            
            implied_ev_avg = target['revenue'] * stats['average']
            implied_ev_median = target['revenue'] * stats['median']
            
            net_debt = target['total_debt'] - target['cash']
            
            equity_value_avg = implied_ev_avg - net_debt
            equity_value_median = implied_ev_median - net_debt
            
            fair_value_avg = equity_value_avg / target['shares']
            fair_value_median = equity_value_median / target['shares']
            
            valuations_summary['ev_sales'] = {
                'method': 'EV/Sales',
                'target_metric': target['revenue'],
                'metric_name': 'Revenue',
                'avg_multiple': stats['average'],
                'median_multiple': stats['median'],
                'implied_ev_avg': implied_ev_avg,
                'implied_ev_median': implied_ev_median,
                'net_debt': net_debt,
                'fair_value_avg': fair_value_avg,
                'fair_value_median': fair_value_median,
                'current_price': target['current_price'],
                'upside_avg': ((fair_value_avg - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'upside_median': ((fair_value_median - target['current_price']) / target['current_price'] * 100) if target['current_price'] else 0,
                'formula_avg': f"(Revenue × Avg EV/Sales - Net Debt) / Shares = (₹{target['revenue']/1e7:.2f}Cr × {stats['average']:.2f} - ₹{net_debt/1e7:.2f}Cr) / {target['shares']/1e7:.2f}Cr",
                'formula_median': f"(Revenue × Median EV/Sales - Net Debt) / Shares = (₹{target['revenue']/1e7:.2f}Cr × {stats['median']:.2f} - ₹{net_debt/1e7:.2f}Cr) / {target['shares']/1e7:.2f}Cr"
            }
        
        results['valuations'] = valuations_summary
        
        # Calculate Forward P/E using projected earnings
        if target_financials and 'nopat' in target_financials:
            # Use projected year 1 earnings
            forward_eps = (target_financials['nopat'][-1] * 100000) / target['shares'] if target['shares'] > 0 else 0
            
            if 'pe' in results['multiples_stats'] and forward_eps > 0:
                stats = results['multiples_stats']['pe']
                forward_fair_value_avg = forward_eps * stats['average']
                forward_fair_value_median = forward_eps * stats['median']
                
                results['forward_pe'] = {
                    'forward_eps': forward_eps,
                    'fair_value_avg': forward_fair_value_avg,
                    'fair_value_median': forward_fair_value_median,
                    'formula_avg': f"Forward EPS × Avg P/E = ₹{forward_eps:.2f} × {stats['average']:.2f} = ₹{forward_fair_value_avg:.2f}",
                    'formula_median': f"Forward EPS × Median P/E = ₹{forward_eps:.2f} × {stats['median']:.2f} = ₹{forward_fair_value_median:.2f}"
                }
        
        return results
        
    except Exception as e:
        st.error(f"Comparative valuation error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def calculate_working_capital_metrics(financials):
    """Calculate working capital days"""
    wc_metrics = {
        'inventory_days': [],
        'debtor_days': [],
        'creditor_days': []
    }
    
    for i in range(len(financials['years'])):
        revenue = financials['revenue'][i]
        cogs = financials['cogs'][i]
        inventory = financials['inventory'][i]
        receivables = financials['receivables'][i]
        payables = financials['payables'][i]
        
        inv_days = (inventory / cogs * 365) if cogs > 0 else 0
        deb_days = (receivables / revenue * 365) if revenue > 0 else 0
        cred_days = (payables / cogs * 365) if cogs > 0 else 0
        
        wc_metrics['inventory_days'].append(inv_days)
        wc_metrics['debtor_days'].append(deb_days)
        wc_metrics['creditor_days'].append(cred_days)
    
    # Average days
    wc_metrics['avg_inv_days'] = np.mean(wc_metrics['inventory_days']) if wc_metrics['inventory_days'] else 0
    wc_metrics['avg_deb_days'] = np.mean(wc_metrics['debtor_days']) if wc_metrics['debtor_days'] else 0
    wc_metrics['avg_cred_days'] = np.mean(wc_metrics['creditor_days']) if wc_metrics['creditor_days'] else 0
    
    return wc_metrics

def project_financials(financials, wc_metrics, years, tax_rate, rev_growth_override, opex_margin_override):
    """Project financial statements for specified years"""
    # Calculate growth rates
    revenues = financials['revenue']
    growth_rates = [(revenues[i] - revenues[i-1]) / revenues[i-1] * 100 for i in range(1, len(revenues)) if revenues[i-1] != 0]
    avg_growth = np.mean(growth_rates) if growth_rates else 10.0
    
    if rev_growth_override:
        avg_growth = float(rev_growth_override)
    
    # Calculate COGS margin instead of opex margin
    cogs_margins = [financials['cogs'][i] / financials['revenue'][i] * 100 for i in range(len(revenues)) if financials['revenue'][i] > 0]
    avg_cogs_margin = np.mean(cogs_margins) if cogs_margins else 60.0
    
    # Calculate opex margin (opex as % of revenue)
    opex_margins = [financials['opex'][i] / financials['revenue'][i] * 100 for i in range(len(revenues)) if financials['revenue'][i] > 0]
    avg_opex_margin = np.mean(opex_margins) if opex_margins else 20.0
    
    if opex_margin_override:
        avg_opex_margin = float(opex_margin_override)
    
    # Depreciation rate
    dep_rates = [financials['depreciation'][i] / financials['fixed_assets'][i] * 100 for i in range(len(revenues)) if financials['fixed_assets'][i] > 0]
    avg_dep_rate = np.mean(dep_rates) if dep_rates else 5.0
    
    # Finance cost rate
    total_debts = [financials['st_debt'][i] + financials['lt_debt'][i] for i in range(len(revenues))]
    fin_cost_rates = [financials['interest'][i] / total_debts[i] * 100 for i in range(len(revenues)) if total_debts[i] > 0]
    avg_fin_cost_rate = np.mean(fin_cost_rates) if fin_cost_rates else 8.0
    
    # Asset growth rates
    equity_growth = [(financials['equity'][i] - financials['equity'][i-1]) / financials['equity'][i-1] * 100 for i in range(1, len(revenues)) if financials['equity'][i-1] != 0]
    avg_equity_growth = np.mean(equity_growth) if equity_growth else 10.0
    
    fa_growth = [(financials['fixed_assets'][i] - financials['fixed_assets'][i-1]) / financials['fixed_assets'][i-1] * 100 for i in range(1, len(revenues)) if financials['fixed_assets'][i-1] != 0]
    avg_fa_growth = np.mean(fa_growth) if fa_growth else 10.0
    
    debt_growth = [(total_debts[i] - total_debts[i-1]) / total_debts[i-1] * 100 for i in range(1, len(revenues)) if total_debts[i-1] != 0]
    avg_debt_growth = np.mean(debt_growth) if debt_growth else 0.0
    
    # Projections
    projections = {
        'year': [],
        'revenue': [],
        'cogs': [],
        'opex': [],
        'ebitda': [],
        'depreciation': [],
        'ebit': [],
        'interest': [],
        'nopat': [],
        'fixed_assets': [],
        'equity': [],
        'debt': [],
        'wc': [],
        'delta_wc': [],
        'capex': [],
        'fcff': []
    }
    
    last_revenue = revenues[-1]
    last_fa = financials['fixed_assets'][-1]
    last_equity = financials['equity'][-1]
    last_debt = total_debts[-1] if total_debts[-1] > 0 else 0
    last_wc = 0  # Will calculate
    
    for year in range(1, years + 1):
        # Revenue
        projected_revenue = last_revenue * (1 + avg_growth / 100)
        
        # COGS
        projected_cogs = projected_revenue * (avg_cogs_margin / 100)
        
        # Operating expenses
        projected_opex = projected_revenue * (avg_opex_margin / 100)
        
        # EBITDA = Revenue - COGS - Opex
        projected_ebitda = projected_revenue - projected_cogs - projected_opex
        
        # Fixed Assets
        projected_fa = last_fa * (1 + avg_fa_growth / 100)
        
        # Depreciation
        projected_dep = projected_fa * (avg_dep_rate / 100)
        
        # EBIT
        projected_ebit = projected_ebitda - projected_dep
        
        # Debt
        projected_debt = last_debt * (1 + avg_debt_growth / 100) if last_debt > 0 else 0
        
        # Interest
        projected_interest = projected_debt * (avg_fin_cost_rate / 100) if projected_debt > 0 else 0
        
        # NOPAT
        projected_nopat = projected_ebit * (1 - tax_rate / 100)
        
        # Working Capital
        projected_inventory = projected_cogs * wc_metrics['avg_inv_days'] / 365 if projected_cogs > 0 else 0
        projected_receivables = projected_revenue * wc_metrics['avg_deb_days'] / 365 if projected_revenue > 0 else 0
        projected_payables = projected_cogs * wc_metrics['avg_cred_days'] / 365 if projected_cogs > 0 else 0
        projected_wc = projected_inventory + projected_receivables - projected_payables
        
        delta_wc = projected_wc - last_wc
        
        # Capex
        capex = projected_fa - last_fa + projected_dep
        
        # FCFF
        fcff = projected_nopat + projected_dep - delta_wc - capex
        
        projections['year'].append(year)
        projections['revenue'].append(projected_revenue)
        projections['cogs'].append(projected_cogs)
        projections['opex'].append(projected_opex)
        projections['ebitda'].append(projected_ebitda)
        projections['depreciation'].append(projected_dep)
        projections['ebit'].append(projected_ebit)
        projections['interest'].append(projected_interest)
        projections['nopat'].append(projected_nopat)
        projections['fixed_assets'].append(projected_fa)
        projections['equity'].append(last_equity * (1 + avg_equity_growth / 100))
        projections['debt'].append(projected_debt)
        projections['wc'].append(projected_wc)
        projections['delta_wc'].append(delta_wc)
        projections['capex'].append(capex)
        projections['fcff'].append(fcff)
        
        # Update for next iteration
        last_revenue = projected_revenue
        last_fa = projected_fa
        last_equity = projections['equity'][-1]
        last_debt = projected_debt
        last_wc = projected_wc
    
    return projections, {
        'avg_growth': avg_growth,
        'avg_cogs_margin': avg_cogs_margin,
        'avg_opex_margin': avg_opex_margin,
        'avg_dep_rate': avg_dep_rate,
        'avg_fin_cost_rate': avg_fin_cost_rate
    }

def calculate_wacc(financials, tax_rate, peer_tickers=None):
    """Calculate WACC with proper beta calculation from peers"""
    # Cost of Equity (Ke)
    rf = get_risk_free_rate()
    rm = get_market_return()
    
    # Calculate beta from peer tickers
    beta = 1.0
    if peer_tickers and peer_tickers.strip():
        ticker_list = [t.strip() for t in peer_tickers.split(',') if t.strip()]
        betas = []
        
        for ticker in ticker_list:
            try:
                ticker_beta = get_stock_beta(ticker)
                if ticker_beta > 0:
                    betas.append(ticker_beta)
                    st.info(f"Beta for {ticker}: {ticker_beta:.3f}")
            except Exception as e:
                st.warning(f"Could not fetch beta for {ticker}: {str(e)}")
        
        if betas:
            beta = np.mean(betas)
            st.success(f"✅ Average peer beta: {beta:.3f} (from {len(betas)} peers)")
        else:
            st.warning("⚠️ Could not calculate beta from peers, using default β=1.0")
            beta = 1.0
    else:
        st.warning("⚠️ No peer tickers provided, using default β=1.0")
    
    ke = rf + (beta * (rm - rf))
    
    # Cost of Debt (Kd)
    total_debt = financials['st_debt'][-1] + financials['lt_debt'][-1]
    interest = financials['interest'][-1]
    
    kd = (interest / total_debt * 100) if total_debt > 0 else 8.0
    kd_after_tax = kd * (1 - tax_rate / 100)
    
    # WACC
    equity = financials['equity'][-1]
    total_capital = equity + total_debt
    
    we = equity / total_capital if total_capital > 0 else 1.0
    wd = total_debt / total_capital if total_capital > 0 else 0.0
    
    wacc = (we * ke) + (wd * kd_after_tax)
    
    return {
        'wacc': wacc,
        'ke': ke,
        'kd': kd,
        'kd_after_tax': kd_after_tax,
        'rf': rf,
        'rm': rm,
        'beta': beta,
        'we': we * 100,
        'wd': wd * 100,
        'equity': equity,
        'debt': total_debt
    }

def calculate_dcf_valuation(projections, wacc_details, terminal_growth, num_shares):
    """Calculate DCF valuation with Rulebook-compliant validations"""
    wacc = wacc_details['wacc']
    g = terminal_growth
    
    # RULEBOOK SECTION 8.2: Terminal growth must be < WACC
    if g >= wacc:
        return None, "❌ HARD ERROR: Terminal growth rate must be less than WACC (Rulebook 8.2)"
    
    # RULEBOOK SECTION 8.2: Check terminal year FCFF
    last_fcff = projections['fcff'][-1]
    if last_fcff <= 0:
        return None, "❌ HARD ERROR: Terminal year FCFF is negative or zero. Cannot compute Terminal Value (Rulebook 8.2)"
    
    # Present Value of FCFFs
    pv_fcffs = []
    for i, fcff in enumerate(projections['fcff']):
        year = i + 1
        pv = fcff / ((1 + wacc / 100) ** year)
        pv_fcffs.append(pv)
    
    sum_pv_fcff = sum(pv_fcffs)
    
    # Terminal Value (Rulebook Section 8.1)
    fcff_n_plus_1 = last_fcff * (1 + g / 100)
    terminal_value = fcff_n_plus_1 / ((wacc / 100) - (g / 100))
    
    n = len(projections['fcff'])
    pv_terminal_value = terminal_value / ((1 + wacc / 100) ** n)
    
    # Enterprise Value (in Lacs)
    enterprise_value = sum_pv_fcff + pv_terminal_value
    
    # RULEBOOK SECTION 13.1: Terminal Value sanity checks
    tv_percentage = (pv_terminal_value / enterprise_value * 100) if enterprise_value > 0 else 0
    
    if tv_percentage > 100:
        return None, f"❌ ERROR: Terminal Value ({tv_percentage:.1f}%) exceeds 100% of Enterprise Value (Rulebook 13.1)"
    
    # Equity Value (in Lacs)
    net_debt = wacc_details['debt'] - 0  # Assuming cash is 0 for simplicity
    equity_value = enterprise_value - net_debt
    
    # Convert Equity Value from Lacs to absolute Rupees, then divide by shares
    # Equity Value is in Lacs, so multiply by 100,000 to get Rupees
    equity_value_rupees = equity_value * 100000
    fair_value_per_share = equity_value_rupees / num_shares if num_shares > 0 else 0
    
    return {
        'pv_fcffs': pv_fcffs,
        'sum_pv_fcff': sum_pv_fcff,
        'terminal_value': terminal_value,
        'pv_terminal_value': pv_terminal_value,
        'enterprise_value': enterprise_value,
        'net_debt': net_debt,
        'equity_value': equity_value,
        'equity_value_rupees': equity_value_rupees,
        'fair_value_per_share': fair_value_per_share,
        'tv_percentage': tv_percentage,
        'tv_warning': tv_percentage > 90  # Flag for warning
    }, None

# ================================
# STREAMLIT UI
# ================================

st.set_page_config(page_title="DCF Valuation Engine", layout="wide")

st.title("🏦 DCF Valuation Engine")
st.markdown("**Listed + Unlisted | XML-Integrated | Traditional WACC**")

# Mode Selection
mode = st.radio("Select Mode:", ["Listed Company (Yahoo Finance)", "Unlisted Company (XML Upload)"], horizontal=True)

if mode == "Listed Company (Yahoo Finance)":
    st.subheader("📈 Listed Company Valuation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Enter NSE Ticker (e.g., RELIANCE, TATASTEEL, INFY):")
        comp_tickers_listed = st.text_input("Comparable Tickers (comma-separated, e.g., HDFC, ICICIBANK):", key='comp_listed')
    
    with col2:
        tax_rate = st.number_input("Tax Rate (%):", min_value=0.0, max_value=100.0, value=25.0, step=0.5, key='listed_tax')
        terminal_growth = st.number_input("Terminal Growth Rate (%):", min_value=0.0, max_value=10.0, value=4.0, step=0.5, key='listed_tg')
    
    with st.expander("⚙️ Advanced Assumptions (Optional Overrides)"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            rev_growth_override_listed = st.text_input("Revenue Growth Override (%):", placeholder="Leave blank for auto", key='listed_rev')
        with col_b:
            opex_margin_override_listed = st.text_input("Opex Margin Override (%):", placeholder="Leave blank for auto", key='listed_opex')
        with col_c:
            projection_years_listed = st.number_input("Projection Years:", min_value=3, max_value=10, value=5, step=1, key='listed_years')
    
    if ticker:
        if st.button("🚀 Fetch & Analyze", type="primary"):
            with st.spinner("Fetching data from Yahoo Finance..."):
                # Fetch data
                yahoo_data, error = fetch_yahoo_financials(ticker)
                
                if error:
                    st.error(error)
                    st.stop()
                
                shares = yahoo_data['shares']
                company_name = yahoo_data['info'].get('longName', ticker)
                
                st.success(f"✅ Loaded data for **{company_name}**")
                st.info(f"📊 Shares Outstanding: **{shares:,}**")
                
                # Extract financials
                financials = extract_financials_listed(yahoo_data)
                
                if financials is None:
                    st.error("Failed to extract financial data")
                    st.stop()
                
                # ================================
                # BUSINESS MODEL CLASSIFICATION (RULEBOOK SECTION 2)
                # ================================
                st.markdown("---")
                st.subheader("🏢 Business Model Classification")
                
                classification = classify_business_model(
                    financials, 
                    income_stmt=yahoo_data['income_statement'], 
                    balance_sheet=yahoo_data['balance_sheet']
                )
                
                # Show classification and check if FCFF DCF is allowed
                should_stop = show_classification_warning(classification)
                
                if should_stop:
                    # For Interest-Dominant entities, run alternative valuation methods
                    st.markdown("---")
                    st.header("🏦 Bank/NBFC Valuation Methods")
                    st.info("Running alternative valuation methods appropriate for interest-dominant entities...")
                    
                    # Get current price
                    stock = yf.Ticker(f"{ticker}.NS")
                    info = stock.info
                    current_price = info.get('currentPrice', 0)
                    
                    # Calculate Ke for bank valuations
                    wacc_details = calculate_wacc(financials, tax_rate, peer_tickers=None)
                    beta = get_stock_beta(ticker, period_years=3)
                    wacc_details['beta'] = beta
                    wacc_details['ke'] = wacc_details['rf'] + (beta * (wacc_details['rm'] - wacc_details['rf']))
                    cost_of_equity = wacc_details['ke']
                    
                    # Run all bank valuation methods
                    ri_model = calculate_residual_income_model(financials, shares, cost_of_equity)
                    ddm_model = calculate_dividend_discount_model(financials, shares, cost_of_equity, ticker=ticker)
                    pb_roe_model = calculate_pb_roe_valuation(financials, shares, cost_of_equity)
                    rel_val = calculate_relative_valuation(ticker, financials, shares, peer_tickers=comp_tickers_listed)
                    
                    # Display results
                    st.success("✅ Bank Valuation Complete!")
                    
                    # Current Price & Fair Value Display
                    col_price1, col_price2 = st.columns(2)
                    with col_price1:
                        st.metric("📊 Current Market Price", f"₹ {current_price:.2f}")
                    
                    # Collect all fair values
                    fair_values = []
                    if ri_model:
                        fair_values.append(ri_model['value_per_share'])
                    if ddm_model:
                        fair_values.append(ddm_model['value_per_share'])
                    if pb_roe_model:
                        fair_values.append(pb_roe_model['value_per_share'])
                    if rel_val:
                        fair_values.append(rel_val['avg_fair_value'])
                    
                    avg_fair_value = np.mean(fair_values) if fair_values else 0
                    
                    with col_price2:
                        st.metric("🎯 Average Fair Value", f"₹ {avg_fair_value:.2f}",
                                 delta=f"{((avg_fair_value - current_price) / current_price * 100):.1f}%")
                    
                    # Price vs Value Gauge
                    if avg_fair_value > 0:
                        st.plotly_chart(create_price_vs_value_gauge(current_price, avg_fair_value), 
                                      use_container_width=True)
                    
                    # Valuation Methods Tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Summary", 
                        "🏢 Residual Income", 
                        "💰 Dividend Discount",
                        "📈 P/B with ROE",
                        "🔄 Relative Valuation"
                    ])
                    
                    with tab1:
                        st.subheader("Valuation Summary - All Methods")
                        
                        # Comparison chart
                        valuations_dict = {}
                        if ri_model:
                            valuations_dict['Residual Income Model'] = ri_model
                        if ddm_model:
                            valuations_dict['Dividend Discount Model'] = ddm_model
                        if pb_roe_model:
                            valuations_dict['P/B with ROE'] = pb_roe_model
                        if rel_val:
                            valuations_dict['Relative Valuation'] = {'value_per_share': rel_val['avg_fair_value']}
                        
                        if valuations_dict:
                            st.plotly_chart(create_bank_valuation_comparison_chart(valuations_dict), 
                                          use_container_width=True)
                        
                        # Summary table
                        summary_data = []
                        if ri_model:
                            summary_data.append(['Residual Income Model', f"₹{ri_model['value_per_share']:.2f}", 
                                               f"{((ri_model['value_per_share'] - current_price) / current_price * 100):.1f}%"])
                        if ddm_model:
                            summary_data.append(['Dividend Discount Model', f"₹{ddm_model['value_per_share']:.2f}",
                                               f"{((ddm_model['value_per_share'] - current_price) / current_price * 100):.1f}%"])
                        if pb_roe_model:
                            summary_data.append(['P/B with ROE Analysis', f"₹{pb_roe_model['value_per_share']:.2f}",
                                               f"{((pb_roe_model['value_per_share'] - current_price) / current_price * 100):.1f}%"])
                        if rel_val:
                            summary_data.append(['Relative Valuation (Avg)', f"₹{rel_val['avg_fair_value']:.2f}",
                                               f"{((rel_val['avg_fair_value'] - current_price) / current_price * 100):.1f}%"])
                        
                        summary_df = pd.DataFrame(summary_data, columns=['Method', 'Fair Value', 'Upside/Downside'])
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        
                        # Historical charts
                        st.markdown("---")
                        st.subheader("📈 Historical Financial Analysis")
                        st.plotly_chart(create_historical_financials_chart(financials), use_container_width=True)
                    
                    with tab2:
                        if ri_model:
                            st.subheader("Residual Income Model")
                            st.write(f"**Fair Value per Share:** ₹{ri_model['value_per_share']:.2f}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Book Value", f"₹{ri_model['current_book_value']:,.0f}")
                            with col2:
                                st.metric("ROE", f"{ri_model['roe']:.2f}%")
                            with col3:
                                st.metric("BV Growth Rate", f"{ri_model.get('bv_growth', 10):.1f}%",
                                         help="Historical book value growth rate")
                            
                            # Growth assumptions
                            st.markdown("---")
                            st.subheader("Growth Assumptions")
                            col4, col5 = st.columns(2)
                            with col4:
                                st.info(f"**Projection Phase Growth:** {ri_model.get('bv_growth', 10):.1f}%")
                                if ri_model.get('historical_bv_growth'):
                                    hist_growth = ri_model['historical_bv_growth']
                                    st.caption(f"Based on historical growth: {', '.join([f'{g:.1f}%' for g in hist_growth])}")
                                else:
                                    st.caption("Using default 10% growth")
                            
                            with col5:
                                st.info(f"**Terminal Growth:** {ri_model.get('terminal_growth', 4):.1f}%")
                                st.caption("Assumes gradual decline to sustainable long-term rate")
                            
                            # Projections table
                            st.markdown("---")
                            st.subheader("5-Year Residual Income Projections")
                            proj_df = pd.DataFrame(ri_model['projections'])
                            proj_df['book_value'] = proj_df['book_value'].apply(lambda x: f"₹{x:,.0f}")
                            proj_df['net_income'] = proj_df['net_income'].apply(lambda x: f"₹{x:,.0f}")
                            proj_df['residual_income'] = proj_df['residual_income'].apply(lambda x: f"₹{x:,.0f}")
                            proj_df['pv_ri'] = proj_df['pv_ri'].apply(lambda x: f"₹{x:,.0f}")
                            st.dataframe(proj_df, use_container_width=True)
                            
                            # Value breakdown
                            st.markdown("---")
                            st.subheader("Valuation Breakdown")
                            breakdown_df = pd.DataFrame({
                                'Component': ['Current Book Value', 'PV of 5Y Residual Income', 'PV of Terminal Value', 'Total Equity Value'],
                                'Value (₹)': [
                                    f"₹{ri_model['current_book_value']:,.0f}",
                                    f"₹{ri_model['sum_pv_ri']:,.0f}",
                                    f"₹{ri_model['terminal_ri_pv']:,.0f}",
                                    f"₹{ri_model['total_equity_value']:,.0f}"
                                ]
                            })
                            st.table(breakdown_df)
                            
                            # Formula explanation
                            with st.expander("📖 Residual Income Formula"):
                                st.latex(r"RI = NI - (K_e \times BV)")
                                st.latex(r"Value = BV_0 + \sum_{t=1}^{n} \frac{RI_t}{(1+K_e)^t} + \frac{TV}{(1+K_e)^n}")
                                st.write("Where:")
                                st.write("- RI = Residual Income")
                                st.write("- NI = Net Income")
                                st.write(f"- Kₑ = Cost of Equity = {cost_of_equity:.2f}%")
                                st.write("- BV = Book Value of Equity")
                                st.write("- TV = Terminal Value")
                        else:
                            st.warning("Residual Income Model calculation failed")
                    
                    with tab3:
                        if ddm_model:
                            st.subheader("Dividend Discount Model")
                            
                            # Show data source
                            if ddm_model.get('using_actual_data'):
                                st.success("✅ Using actual historical dividend data")
                            else:
                                st.info("ℹ️ Using estimated dividend data (no historical dividends found)")
                            
                            st.write(f"**Fair Value per Share:** ₹{ddm_model['value_per_share']:.2f}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Current DPS", f"₹{ddm_model['current_dps']:.2f}")
                                st.metric("Payout Ratio", f"{ddm_model['payout_ratio']:.1f}%")
                            
                            with col2:
                                st.metric("Dividend Growth", f"{ddm_model['dividend_growth']:.1f}%")
                                st.metric("Next Year DPS (D1)", f"₹{ddm_model['next_year_dps']:.2f}")
                            
                            # Historical dividends if available
                            if ddm_model.get('historical_dividends'):
                                st.markdown("---")
                                st.subheader("Historical Dividends (Annual)")
                                hist_divs = ddm_model['historical_dividends']
                                years_range = list(range(len(hist_divs), 0, -1))
                                hist_df = pd.DataFrame({
                                    'Year': [f"T-{y}" for y in years_range],
                                    'Dividend (₹)': hist_divs
                                })
                                st.dataframe(hist_df, use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            st.subheader("5-Year Dividend Projection")
                            
                            # Dividend projections
                            div_df = pd.DataFrame(ddm_model['projections'])
                            div_df['dividend'] = div_df['dividend'].apply(lambda x: f"₹{x:.2f}")
                            div_df['pv_dividend'] = div_df['pv_dividend'].apply(lambda x: f"₹{x:.2f}")
                            st.dataframe(div_df, use_container_width=True)
                            
                            # DDM formula explanation
                            with st.expander("📖 DDM Formula & Assumptions"):
                                st.latex(r"Value = \frac{D_1}{K_e - g}")
                                st.write(f"Where:")
                                st.write(f"- D₁ = Next year dividend = ₹{ddm_model['next_year_dps']:.2f}")
                                st.write(f"- Kₑ = Cost of Equity = {cost_of_equity:.2f}%")
                                st.write(f"- g = Dividend Growth Rate = {ddm_model['dividend_growth']:.2f}%")
                        else:
                            st.warning("DDM calculation failed or not applicable (cost of equity ≤ growth rate)")
                    
                    with tab4:
                        if pb_roe_model:
                            st.subheader("P/B with ROE Analysis")
                            st.write(f"**Fair Value per Share:** ₹{pb_roe_model['value_per_share']:.2f}")
                            st.write(f"**Book Value per Share:** ₹{pb_roe_model['book_value_per_share']:.2f}")
                            st.write(f"**ROE:** {pb_roe_model['roe']:.2f}%")
                            st.write(f"**Cost of Equity:** {pb_roe_model['cost_of_equity']:.2f}%")
                            st.write(f"**Fair P/B Ratio:** {pb_roe_model['fair_pb_ratio']:.2f}x")
                            
                            # Historical ROE chart
                            roe_years = financials['years']
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=roe_years, y=pb_roe_model['historical_roe'],
                                                    mode='lines+markers', name='ROE',
                                                    line=dict(color='#06A77D', width=3)))
                            fig.add_hline(y=pb_roe_model['cost_of_equity'], line_dash="dash",
                                         annotation_text=f"Cost of Equity: {pb_roe_model['cost_of_equity']:.2f}%")
                            fig.update_layout(title="Historical ROE Trend", xaxis_title="Year",
                                            yaxis_title="ROE %", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("P/B ROE model calculation failed")
                    
                    with tab5:
                        if rel_val:
                            st.subheader("Relative Valuation (Peer-Based)")
                            
                            # Show peer data summary
                            if rel_val.get('peer_count', 0) > 0:
                                st.success(f"✅ Analyzed {rel_val['peer_count']} peer companies")
                            else:
                                st.warning("⚠️ Using default market averages (add peer tickers for better accuracy)")
                            
                            # Main metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current P/E", f"{rel_val['current_pe']:.2f}x")
                                st.metric("Peer Avg P/E", f"{rel_val['sector_avg_pe']:.2f}x")
                                st.caption(f"Range: {rel_val.get('sector_low_pe', 0):.1f}x - {rel_val.get('sector_high_pe', 0):.1f}x")
                            
                            with col2:
                                st.metric("Current P/B", f"{rel_val['current_pb']:.2f}x")
                                st.metric("Peer Avg P/B", f"{rel_val['sector_avg_pb']:.2f}x")
                                st.caption(f"Range: {rel_val.get('sector_low_pb', 0):.1f}x - {rel_val.get('sector_high_pb', 0):.1f}x")
                            
                            with col3:
                                st.metric("Fair Value (P/E)", f"₹{rel_val['fair_value_pe_based']:.2f}")
                                st.metric("Fair Value (P/B)", f"₹{rel_val['fair_value_pb_based']:.2f}")
                            
                            st.markdown("---")
                            
                            # Valuation ranges
                            st.subheader("Valuation Range Analysis")
                            col4, col5, col6 = st.columns(3)
                            with col4:
                                st.metric("Conservative", f"₹{rel_val.get('conservative_value', 0):.2f}",
                                         help="Based on 25th percentile peer P/E")
                            with col5:
                                st.metric("Fair Value", f"₹{rel_val['avg_fair_value']:.2f}",
                                         help="Average of P/E and P/B based valuations")
                            with col6:
                                st.metric("Aggressive", f"₹{rel_val.get('aggressive_value', 0):.2f}",
                                         help="Based on 75th percentile peer P/E")
                            
                            # Peer comparison table
                            if rel_val.get('peer_data') and len(rel_val['peer_data']) > 0:
                                st.markdown("---")
                                st.subheader("Peer Comparison")
                                peer_df = pd.DataFrame(rel_val['peer_data'])
                                peer_df.columns = ['Ticker', 'Price (₹)', 'P/E', 'P/B']
                                st.dataframe(peer_df, use_container_width=True, hide_index=True)
                            
                            # Interpretation
                            st.markdown("---")
                            st.subheader("Interpretation")
                            pe_premium = ((rel_val['current_pe'] - rel_val['sector_avg_pe']) / rel_val['sector_avg_pe'] * 100) if rel_val['sector_avg_pe'] > 0 else 0
                            pb_premium = ((rel_val['current_pb'] - rel_val['sector_avg_pb']) / rel_val['sector_avg_pb'] * 100) if rel_val['sector_avg_pb'] > 0 else 0
                            
                            if pe_premium > 20:
                                st.warning(f"📊 **P/E Analysis:** Trading at {pe_premium:.1f}% premium to peers. May be overvalued unless justified by superior growth.")
                            elif pe_premium < -20:
                                st.success(f"📊 **P/E Analysis:** Trading at {abs(pe_premium):.1f}% discount to peers. Potential undervaluation.")
                            else:
                                st.info(f"📊 **P/E Analysis:** Trading in line with peers ({pe_premium:+.1f}% premium).")
                            
                            if pb_premium > 20:
                                st.warning(f"📈 **P/B Analysis:** Trading at {pb_premium:.1f}% premium to peers. May indicate high growth expectations.")
                            elif pb_premium < -20:
                                st.success(f"📈 **P/B Analysis:** Trading at {abs(pb_premium):.1f}% discount to peers. Potential value opportunity.")
                            else:
                                st.info(f"📈 **P/B Analysis:** Trading in line with peers ({pb_premium:+.1f}% premium).")
                        else:
                            st.warning("Relative valuation calculation failed")
                    
                    st.stop()
                
                st.markdown("---")
                
                # Calculate WC metrics
                wc_metrics = calculate_working_capital_metrics(financials)
                
                # Project financials
                projections, drivers = project_financials(
                    financials, wc_metrics, projection_years_listed, tax_rate,
                    rev_growth_override_listed, opex_margin_override_listed
                )
                
                # Calculate WACC (beta of the company itself)
                st.info("Calculating beta for the stock...")
                beta = get_stock_beta(ticker, period_years=3)
                st.success(f"✅ Beta calculated: {beta:.3f}")
                
                wacc_details = calculate_wacc(financials, tax_rate, peer_tickers=None)
                wacc_details['beta'] = beta  # Override with actual stock beta
                # Recalculate Ke and WACC with actual beta
                wacc_details['ke'] = wacc_details['rf'] + (beta * (wacc_details['rm'] - wacc_details['rf']))
                wacc_details['wacc'] = (wacc_details['we']/100 * wacc_details['ke']) + (wacc_details['wd']/100 * wacc_details['kd_after_tax'])
                
                # DCF Valuation
                valuation, error = calculate_dcf_valuation(
                    projections, wacc_details, terminal_growth, shares
                )
                
                if error:
                    st.error(error)
                    st.stop()
                
                # ================================
                # DISPLAY RESULTS (SAME AS UNLISTED)
                # ================================
                
                st.success("✅ Valuation Complete!")
                
                # Get current price for comparison
                stock = yf.Ticker(f"{ticker}.NS")
                info = stock.info
                current_price = info.get('currentPrice', 0)
                
                # Key Metrics with Current Price
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("📊 Current Price", f"₹ {current_price:.2f}")
                with col2:
                    st.metric("🎯 Fair Value/Share", f"₹ {valuation['fair_value_per_share']:.2f}",
                             delta=f"{((valuation['fair_value_per_share'] - current_price) / current_price * 100):.1f}%")
                with col3:
                    st.metric("Enterprise Value", f"₹ {valuation['enterprise_value']:.2f} Lacs")
                with col4:
                    st.metric("Equity Value", f"₹ {valuation['equity_value']:.2f} Lacs")
                with col5:
                    st.metric("WACC", f"{wacc_details['wacc']:.2f}%")
                
                # Price vs Value Gauge
                if valuation['fair_value_per_share'] > 0:
                    st.plotly_chart(create_price_vs_value_gauge(current_price, valuation['fair_value_per_share']), 
                                  use_container_width=True)
                
                # Tabs for detailed output
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "📊 Historical Analysis",
                    "📈 Projections",
                    "💰 FCF Working",
                    "🎯 WACC Breakdown",
                    "🏆 Valuation Summary",
                    "📉 Sensitivity Analysis",
                    "🔍 Comparative Valuation"
                ])
                
                with tab1:
                    st.subheader("📊 Comprehensive Historical Financial Analysis")
                    
                    # Use advanced charting function
                    st.plotly_chart(create_historical_financials_chart(financials), use_container_width=True)
                    
                    # Data tables below charts
                    with st.expander("📋 View Raw Data Tables"):
                        st.subheader("Historical Financials (Last 3 Years)")
                        
                        hist_df = pd.DataFrame({
                            'Year': [str(y) for y in financials['years']],
                            'Revenue': financials['revenue'],
                            'Operating Expenses': financials['opex'],
                            'EBITDA': financials['ebitda'],
                            'Depreciation': financials['depreciation'],
                            'EBIT': financials['ebit'],
                            'Interest': financials['interest'],
                            'Tax': financials['tax'],
                            'NOPAT': financials['nopat']
                        })
                        
                        numeric_cols = hist_df.select_dtypes(include=[np.number]).columns.tolist()
                        format_dict = {col: '{:.2f}' for col in numeric_cols}
                        st.dataframe(hist_df.style.format(format_dict), use_container_width=True)
                        
                        st.subheader("Balance Sheet Metrics")
                        bs_df = pd.DataFrame({
                            'Year': [str(y) for y in financials['years']],
                            'Fixed Assets': financials['fixed_assets'],
                            'Inventory': financials['inventory'],
                            'Receivables': financials['receivables'],
                            'Payables': financials['payables'],
                            'Equity': financials['equity'],
                            'ST Debt': financials['st_debt'],
                            'LT Debt': financials['lt_debt']
                        })
                        numeric_cols = bs_df.select_dtypes(include=[np.number]).columns.tolist()
                        format_dict = {col: '{:.2f}' for col in numeric_cols}
                        st.dataframe(bs_df.style.format(format_dict), use_container_width=True)
                        
                        st.subheader("Working Capital Days")
                        wc_df = pd.DataFrame({
                            'Year': [str(y) for y in financials['years']],
                            'Inventory Days': wc_metrics['inventory_days'],
                            'Debtor Days': wc_metrics['debtor_days'],
                            'Creditor Days': wc_metrics['creditor_days']
                        })
                        numeric_cols = wc_df.select_dtypes(include=[np.number]).columns.tolist()
                        format_dict = {col: '{:.2f}' for col in numeric_cols}
                        st.dataframe(wc_df.style.format(format_dict), use_container_width=True)
                        
                        st.info(f"**Average Working Capital Days:** Inventory: {wc_metrics['avg_inv_days']:.1f} | Debtors: {wc_metrics['avg_deb_days']:.1f} | Creditors: {wc_metrics['avg_cred_days']:.1f}")
                
                with tab2:
                    st.subheader(f"📈 Projected Financials ({projection_years_listed} Years)")
                    
                    # Use advanced charting function
                    st.plotly_chart(create_fcff_projection_chart(projections), use_container_width=True)
                    
                    # Data table below
                    with st.expander("📋 View Projection Data Table"):
                        proj_df = pd.DataFrame({
                            'Year': [str(y) for y in projections['year']],
                            'Revenue': projections['revenue'],
                            'EBITDA': projections['ebitda'],
                            'Depreciation': projections['depreciation'],
                            'EBIT': projections['ebit'],
                            'NOPAT': projections['nopat'],
                            'Capex': projections['capex'],
                            'Δ WC': projections['delta_wc'],
                            'FCFF': projections['fcff']
                        })
                        numeric_cols = proj_df.select_dtypes(include=[np.number]).columns.tolist()
                        format_dict = {col: '{:.2f}' for col in numeric_cols}
                        st.dataframe(proj_df.style.format(format_dict), use_container_width=True)
                    
                    st.info(f"**Key Drivers:** Revenue Growth: {drivers['avg_growth']:.2f}% | Opex Margin: {drivers['avg_opex_margin']:.2f}% | Depreciation Rate: {drivers['avg_dep_rate']:.2f}%")
                
                with tab3:
                    st.subheader("Free Cash Flow Working")
                    
                    fcff_df = pd.DataFrame({
                        'Year': [str(y) for y in projections['year']],
                        'NOPAT': projections['nopat'],
                        '+ Depreciation': projections['depreciation'],
                        '- Δ WC': projections['delta_wc'],
                        '- Capex': projections['capex'],
                        '= FCFF': projections['fcff'],
                        'Discount Factor': [(1 + wacc_details['wacc']/100)**(-y) for y in projections['year']],
                        'PV(FCFF)': valuation['pv_fcffs']
                    })
                    numeric_cols = fcff_df.select_dtypes(include=[np.number]).columns.tolist()
                    format_dict = {col: '{:.4f}' for col in numeric_cols}
                    st.dataframe(fcff_df.style.format(format_dict), use_container_width=True)
                    
                    st.metric("Sum of PV(FCFF)", f"₹ {valuation['sum_pv_fcff']:.2f} Lacs")
                
                with tab4:
                    st.subheader("🎯 WACC Calculation & Breakdown")
                    
                    # Advanced WACC breakdown chart
                    st.plotly_chart(create_wacc_breakdown_chart(wacc_details), use_container_width=True)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Cost of Equity (Ke)**")
                        st.write(f"Risk-free Rate (Rf): **{wacc_details['rf']:.2f}%**")
                        st.write(f"Market Return (Rm): **{wacc_details['rm']:.2f}%**")
                        st.write(f"Beta (β) - {ticker}: **{wacc_details['beta']:.3f}**")
                        st.write(f"Ke = Rf + β × (Rm - Rf)")
                        st.write(f"Ke = {wacc_details['rf']:.2f}% + {wacc_details['beta']:.3f} × ({wacc_details['rm']:.2f}% - {wacc_details['rf']:.2f}%)")
                        st.write(f"**Ke = {wacc_details['ke']:.2f}%**")
                    
                    with col2:
                        st.markdown("**Cost of Debt (Kd)**")
                        st.write(f"Interest Expense: **₹ {financials['interest'][-1]:.2f} Lacs**")
                        st.write(f"Total Debt: **₹ {wacc_details['debt']:.2f} Lacs**")
                        st.write(f"Kd (pre-tax) = {wacc_details['kd']:.2f}%")
                        st.write(f"Tax Rate = {tax_rate}%")
                        st.write(f"**Kd (after-tax) = {wacc_details['kd_after_tax']:.2f}%**")
                    
                    st.markdown("---")
                    st.markdown("**WACC Calculation**")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write(f"Equity (E): **₹ {wacc_details['equity']:.2f} Lacs** ({wacc_details['we']:.2f}%)")
                        st.write(f"Debt (D): **₹ {wacc_details['debt']:.2f} Lacs** ({wacc_details['wd']:.2f}%)")
                        st.write(f"Total Capital (V): **₹ {wacc_details['equity'] + wacc_details['debt']:.2f} Lacs**")
                    
                    with col4:
                        st.write(f"WACC = (E/V × Ke) + (D/V × Kd × (1-Tax))")
                        st.write(f"WACC = ({wacc_details['we']:.2f}% × {wacc_details['ke']:.2f}%) + ({wacc_details['wd']:.2f}% × {wacc_details['kd_after_tax']:.2f}%)")
                        st.write(f"**WACC = {wacc_details['wacc']:.2f}%**")
                
                with tab5:
                    st.subheader("🏆 DCF Valuation Summary")
                    
                    # Waterfall chart showing value buildup
                    st.plotly_chart(create_waterfall_chart(valuation), use_container_width=True)
                    
                    st.markdown("### Terminal Value Calculation")
                    st.write(f"FCFF (Year {projection_years_listed}): **₹ {projections['fcff'][-1]:.2f} Lacs**")
                    st.write(f"Terminal Growth Rate (g): **{terminal_growth}%**")
                    st.write(f"FCFF (Year {projection_years_listed + 1}) = FCFF{projection_years_listed} × (1 + g)")
                    st.write(f"FCFF (Year {projection_years_listed + 1}) = ₹ {projections['fcff'][-1]:.2f} × (1 + {terminal_growth/100})")
                    st.write(f"FCFF (Year {projection_years_listed + 1}) = **₹ {projections['fcff'][-1] * (1 + terminal_growth/100):.2f} Lacs**")
                    
                    st.write(f"\nTerminal Value = FCFF{projection_years_listed + 1} / (WACC - g)")
                    st.write(f"Terminal Value = ₹ {projections['fcff'][-1] * (1 + terminal_growth/100):.2f} / ({wacc_details['wacc']:.2f}% - {terminal_growth}%)")
                    st.write(f"**Terminal Value = ₹ {valuation['terminal_value']:.2f} Lacs**")
                    
                    st.write(f"\nPV(Terminal Value) = TV / (1 + WACC)^{projection_years_listed}")
                    st.write(f"**PV(Terminal Value) = ₹ {valuation['pv_terminal_value']:.2f} Lacs**")
                    
                    st.markdown("---")
                    st.markdown("### Enterprise Value")
                    
                    ev_df = pd.DataFrame({
                        'Component': ['Sum of PV(FCFF)', 'PV(Terminal Value)', 'Enterprise Value'],
                        'Value (₹ Lacs)': [
                            valuation['sum_pv_fcff'],
                            valuation['pv_terminal_value'],
                            valuation['enterprise_value']
                        ]
                    })
                    st.dataframe(ev_df.style.format({'Value (₹ Lacs)': '{:.2f}'}), use_container_width=True)
                    
                    tv_pct = valuation['tv_percentage']
                    if tv_pct > 90:
                        st.warning(f"⚠️ Terminal Value represents {tv_pct:.1f}% of Enterprise Value (>90% is high)")
                    else:
                        st.info(f"Terminal Value represents {tv_pct:.1f}% of Enterprise Value")
                    
                    st.markdown("---")
                    st.markdown("### Equity Value & Fair Value per Share")
                    
                    equity_calc_df = pd.DataFrame({
                        'Item': ['Enterprise Value', 'Less: Net Debt', 'Equity Value', 'Equity Value (₹)', 'Number of Shares', 'Fair Value per Share'],
                        'Value': [
                            f"₹ {valuation['enterprise_value']:.2f} Lacs",
                            f"₹ {valuation['net_debt']:.2f} Lacs",
                            f"₹ {valuation['equity_value']:.2f} Lacs",
                            f"₹ {valuation['equity_value_rupees']:,.0f}",
                            f"{shares:,}",
                            f"₹ {valuation['fair_value_per_share']:.2f}"
                        ]
                    })
                    st.table(equity_calc_df)
                    
                    st.success(f"### 🎯 Fair Value per Share: ₹ {valuation['fair_value_per_share']:.2f}")
                
                with tab6:
                    st.subheader("📉 Advanced Sensitivity Analysis")
                    
                    wacc_range = np.arange(max(1.0, wacc_details['wacc'] - 3), wacc_details['wacc'] + 3.5, 0.5)
                    g_range = np.arange(max(1.0, terminal_growth - 2), min(terminal_growth + 3, wacc_details['wacc'] - 1), 0.5)
                    
                    if len(g_range) == 0:
                        g_range = np.array([terminal_growth])
                    
                    # Interactive heatmap
                    st.plotly_chart(create_sensitivity_heatmap(projections, wacc_range, g_range, shares),
                                  use_container_width=True)
                    
                    # Traditional table below
                    with st.expander("📋 View Sensitivity Data Table"):
                        sensitivity_data = []
                        
                        for w in wacc_range:
                            row_data = {'WACC →': f"{w:.1f}%"}
                            for g_val in g_range:
                                if g_val >= w - 0.1:  # Need at least 0.1% gap
                                    row_data[f"g={g_val:.1f}%"] = "N/A"
                                else:
                                    try:
                                        fcff_n_plus_1 = projections['fcff'][-1] * (1 + g_val / 100)
                                        tv = fcff_n_plus_1 / ((w / 100) - (g_val / 100))
                                        pv_tv = tv / ((1 + w / 100) ** projection_years_listed)
                                        ev = valuation['sum_pv_fcff'] + pv_tv
                                        eq_val = ev - valuation['net_debt']
                                        eq_val_rupees = eq_val * 100000
                                        fv = eq_val_rupees / shares if shares > 0 else 0
                                        row_data[f"g={g_val:.1f}%"] = f"₹{fv:.2f}"
                                    except:
                                        row_data[f"g={g_val:.1f}%"] = "Error"
                            sensitivity_data.append(row_data)
                        
                        sensitivity_df = pd.DataFrame(sensitivity_data)
                        st.dataframe(sensitivity_df, use_container_width=True)
                        
                        st.caption("Sensitivity table shows Fair Value per Share for different WACC and terminal growth rate combinations")
                
                with tab7:
                    st.subheader("🔍 Comparative (Relative) Valuation")
                    
                    if comp_tickers_listed:
                        with st.spinner("Fetching comparable companies data..."):
                            comp_results = perform_comparative_valuation(ticker, comp_tickers_listed, None, shares)
                        
                        if comp_results:
                            # Show comparables table
                            st.markdown("### Comparable Companies")
                            comp_df = pd.DataFrame(comp_results['comparables'])
                            if not comp_df.empty:
                                display_comp_df = comp_df[['ticker', 'name', 'price', 'pe', 'pb', 'ps', 'ev_ebitda', 'ev_sales']]
                                st.dataframe(display_comp_df.style.format({
                                    'price': '₹{:.2f}',
                                    'pe': '{:.2f}x',
                                    'pb': '{:.2f}x',
                                    'ps': '{:.2f}x',
                                    'ev_ebitda': '{:.2f}x',
                                    'ev_sales': '{:.2f}x'
                                }), use_container_width=True)
                            
                            # Show multiples statistics
                            st.markdown("### Peer Multiples Statistics")
                            for multiple, stats in comp_results['multiples_stats'].items():
                                with st.expander(f"📊 {multiple.upper()} - Avg: {stats['average']:.2f}x, Median: {stats['median']:.2f}x"):
                                    st.write(f"**Range:** {stats['min']:.2f}x - {stats['max']:.2f}x")
                                    st.write(f"**Std Dev:** {stats['std']:.2f}x")
                                    st.write(f"**Peer Values:** {', '.join([f'{v:.2f}x' for v in stats['values']])}")
                            
                            # Show implied valuations
                            st.markdown("### Implied Fair Values")
                            
                            all_avg_values = []
                            all_median_values = []
                            
                            for method_key, val_data in comp_results['valuations'].items():
                                st.markdown(f"#### {val_data['method']}")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Using Average Multiple:**")
                                    st.write(val_data['formula_avg'])
                                    st.metric("Fair Value (Avg)", f"₹{val_data['fair_value_avg']:.2f}", 
                                            f"{val_data['upside_avg']:.1f}%" if val_data['current_price'] else None)
                                    all_avg_values.append(val_data['fair_value_avg'])
                                
                                with col2:
                                    st.markdown("**Using Median Multiple:**")
                                    st.write(val_data['formula_median'])
                                    st.metric("Fair Value (Median)", f"₹{val_data['fair_value_median']:.2f}",
                                            f"{val_data['upside_median']:.1f}%" if val_data['current_price'] else None)
                                    all_median_values.append(val_data['fair_value_median'])
                                
                                st.markdown("---")
                            
                            # Forward P/E if available
                            if 'forward_pe' in comp_results:
                                st.markdown("#### Forward P/E Valuation")
                                fpe = comp_results['forward_pe']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(fpe['formula_avg'])
                                    st.metric("Forward Fair Value (Avg)", f"₹{fpe['fair_value_avg']:.2f}")
                                with col2:
                                    st.write(fpe['formula_median'])
                                    st.metric("Forward Fair Value (Median)", f"₹{fpe['fair_value_median']:.2f}")
                            
                            # Summary statistics
                            if all_avg_values and all_median_values:
                                st.markdown("### 📈 Comparative Valuation Summary")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Average (All Methods)", f"₹{np.mean(all_avg_values):.2f}")
                                    st.metric("Median (All Methods)", f"₹{np.median(all_median_values):.2f}")
                                
                                with col2:
                                    st.metric("Min Fair Value", f"₹{min(all_avg_values + all_median_values):.2f}")
                                    st.metric("Max Fair Value", f"₹{max(all_avg_values + all_median_values):.2f}")
                                
                                with col3:
                                    if valuation['fair_value_per_share'] > 0:
                                        st.metric("DCF Fair Value", f"₹{valuation['fair_value_per_share']:.2f}")
                                        combined_avg = (np.mean(all_avg_values) + valuation['fair_value_per_share']) / 2
                                        st.metric("DCF + Comp Avg", f"₹{combined_avg:.2f}")
                    
                    else:
                        st.info("Enter comparable tickers above to see relative valuation")

else:  # Unlisted Mode
    st.subheader("📄 Unlisted Company Valuation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name:")
        xml_file = st.file_uploader("Upload Financial XML (All Sheets)", type=['xml'])
        peer_tickers = st.text_input("Peer Listed Tickers (comma-separated, for beta):", placeholder="RELIANCE, TATASTEEL")
    
    with col2:
        num_shares = st.number_input("Number of Shares Outstanding:", min_value=1, value=100, step=1)
        tax_rate = st.number_input("Tax Rate (%):", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        terminal_growth = st.number_input("Terminal Growth Rate (%):", min_value=0.0, max_value=10.0, value=4.0, step=0.5)
    
    with st.expander("⚙️ Advanced Assumptions (Optional Overrides)"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            rev_growth_override = st.text_input("Revenue Growth Override (%):", placeholder="Leave blank for auto")
        with col_b:
            opex_margin_override = st.text_input("Opex Margin Override (%):", placeholder="Leave blank for auto")
        with col_c:
            projection_years = st.number_input("Projection Years:", min_value=3, max_value=10, value=5, step=1)
    
    if xml_file and company_name and num_shares:
        if st.button("🚀 Run DCF Valuation", type="primary"):
            with st.spinner("Processing..."):
                # Parse XML
                xml_content = xml_file.read()
                df_bs, df_pl = parse_xml_to_dataframes(xml_content)
                
                if df_bs is None or df_pl is None:
                    st.error("Failed to parse XML file")
                    st.stop()
                
                # Detect year columns
                year_cols = detect_year_columns(df_bs)
                
                if len(year_cols) < 3:
                    st.error("Need at least 3 years of historical data")
                    st.stop()
                
                st.success(f"✅ Loaded {len(year_cols)} years of data")
                
                # Extract financials
                financials = extract_financials_unlisted(df_bs, df_pl, year_cols)
                
                # ================================
                # BUSINESS MODEL CLASSIFICATION (RULEBOOK SECTION 2)
                # ================================
                st.markdown("---")
                st.subheader("🏢 Business Model Classification")
                
                classification = classify_business_model(financials, income_stmt=None, balance_sheet=None)
                
                # Show classification and check if FCFF DCF is allowed
                should_stop = show_classification_warning(classification)
                
                if should_stop:
                    st.stop()
                
                st.markdown("---")
                
                # Calculate WC metrics
                wc_metrics = calculate_working_capital_metrics(financials)
                
                # Project financials
                projections, drivers = project_financials(
                    financials, wc_metrics, projection_years, tax_rate,
                    rev_growth_override, opex_margin_override
                )
                
                # Calculate WACC
                wacc_details = calculate_wacc(financials, tax_rate, peer_tickers=peer_tickers)
                
                # DCF Valuation
                valuation, error = calculate_dcf_valuation(
                    projections, wacc_details, terminal_growth, num_shares
                )
                
                if error:
                    st.error(error)
                    st.stop()
                
                # ================================
                # DISPLAY RESULTS
                # ================================
                
                st.success("✅ Valuation Complete!")
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Enterprise Value", f"₹ {valuation['enterprise_value']:.2f} Lacs")
                with col2:
                    st.metric("Equity Value", f"₹ {valuation['equity_value']:.2f} Lacs")
                with col3:
                    st.metric("Fair Value/Share", f"₹ {valuation['fair_value_per_share']:.2f}")
                with col4:
                    st.metric("WACC", f"{wacc_details['wacc']:.2f}%")
                
                # Tabs for detailed output
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Historical Financials",
                    "📈 Projections",
                    "💰 FCF Working",
                    "🎯 WACC Calculation",
                    "🏆 Valuation Summary"
                ])
                
                with tab1:
                    st.subheader("Historical Financials (Last 3 Years)")
                    
                    hist_df = pd.DataFrame({
                        'Year': [str(y) for y in financials['years']],
                        'Revenue': financials['revenue'],
                        'Operating Expenses': financials['opex'],
                        'EBITDA': financials['ebitda'],
                        'Depreciation': financials['depreciation'],
                        'EBIT': financials['ebit'],
                        'Interest': financials['interest'],
                        'Tax': financials['tax'],
                        'NOPAT': financials['nopat']
                    })
                    
                    # Format numeric columns only
                    numeric_cols = hist_df.select_dtypes(include=[np.number]).columns.tolist()
                    format_dict = {col: '{:.2f}' for col in numeric_cols}
                    st.dataframe(hist_df.style.format(format_dict), use_container_width=True)
                    
                    st.subheader("Balance Sheet Metrics")
                    bs_df = pd.DataFrame({
                        'Year': [str(y) for y in financials['years']],
                        'Fixed Assets': financials['fixed_assets'],
                        'Inventory': financials['inventory'],
                        'Receivables': financials['receivables'],
                        'Payables': financials['payables'],
                        'Equity': financials['equity'],
                        'ST Debt': financials['st_debt'],
                        'LT Debt': financials['lt_debt']
                    })
                    numeric_cols = bs_df.select_dtypes(include=[np.number]).columns.tolist()
                    format_dict = {col: '{:.2f}' for col in numeric_cols}
                    st.dataframe(bs_df.style.format(format_dict), use_container_width=True)
                    
                    st.subheader("Working Capital Days")
                    wc_df = pd.DataFrame({
                        'Year': [str(y) for y in financials['years']],
                        'Inventory Days': wc_metrics['inventory_days'],
                        'Debtor Days': wc_metrics['debtor_days'],
                        'Creditor Days': wc_metrics['creditor_days']
                    })
                    numeric_cols = wc_df.select_dtypes(include=[np.number]).columns.tolist()
                    format_dict = {col: '{:.2f}' for col in numeric_cols}
                    st.dataframe(wc_df.style.format(format_dict), use_container_width=True)
                    
                    st.info(f"**Average Working Capital Days:** Inventory: {wc_metrics['avg_inv_days']:.1f} | Debtors: {wc_metrics['avg_deb_days']:.1f} | Creditors: {wc_metrics['avg_cred_days']:.1f}")
                
                with tab2:
                    st.subheader(f"Projected Financials ({projection_years} Years)")
                    
                    proj_df = pd.DataFrame({
                        'Year': [str(y) for y in projections['year']],
                        'Revenue': projections['revenue'],
                        'EBITDA': projections['ebitda'],
                        'Depreciation': projections['depreciation'],
                        'EBIT': projections['ebit'],
                        'NOPAT': projections['nopat'],
                        'Capex': projections['capex'],
                        'Δ WC': projections['delta_wc'],
                        'FCFF': projections['fcff']
                    })
                    numeric_cols = proj_df.select_dtypes(include=[np.number]).columns.tolist()
                    format_dict = {col: '{:.2f}' for col in numeric_cols}
                    st.dataframe(proj_df.style.format(format_dict), use_container_width=True)
                    
                    st.info(f"**Key Drivers:** Revenue Growth: {drivers['avg_growth']:.2f}% | Opex Margin: {drivers['avg_opex_margin']:.2f}% | Depreciation Rate: {drivers['avg_dep_rate']:.2f}%")
                
                with tab3:
                    st.subheader("Free Cash Flow Working")
                    
                    fcff_df = pd.DataFrame({
                        'Year': [str(y) for y in projections['year']],
                        'NOPAT': projections['nopat'],
                        '+ Depreciation': projections['depreciation'],
                        '- Δ WC': projections['delta_wc'],
                        '- Capex': projections['capex'],
                        '= FCFF': projections['fcff'],
                        'Discount Factor': [(1 + wacc_details['wacc']/100)**(-y) for y in projections['year']],
                        'PV(FCFF)': valuation['pv_fcffs']
                    })
                    numeric_cols = fcff_df.select_dtypes(include=[np.number]).columns.tolist()
                    format_dict = {col: '{:.4f}' for col in numeric_cols}
                    st.dataframe(fcff_df.style.format(format_dict), use_container_width=True)
                    
                    st.metric("Sum of PV(FCFF)", f"₹ {valuation['sum_pv_fcff']:.2f} Lacs")
                
                with tab4:
                    st.subheader("WACC Calculation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Cost of Equity (Ke)**")
                        st.write(f"Risk-free Rate (Rf): **{wacc_details['rf']:.2f}%**")
                        st.write(f"Market Return (Rm): **{wacc_details['rm']:.2f}%**")
                        st.write(f"Beta (β): **{wacc_details['beta']:.2f}**")
                        st.write(f"Ke = Rf + β × (Rm - Rf)")
                        st.write(f"Ke = {wacc_details['rf']:.2f}% + {wacc_details['beta']:.2f} × ({wacc_details['rm']:.2f}% - {wacc_details['rf']:.2f}%)")
                        st.write(f"**Ke = {wacc_details['ke']:.2f}%**")
                    
                    with col2:
                        st.markdown("**Cost of Debt (Kd)**")
                        st.write(f"Interest Expense: **₹ {financials['interest'][-1]:.2f} Lacs**")
                        st.write(f"Total Debt: **₹ {wacc_details['debt']:.2f} Lacs**")
                        st.write(f"Kd (pre-tax) = {wacc_details['kd']:.2f}%")
                        st.write(f"Tax Rate = {tax_rate}%")
                        st.write(f"**Kd (after-tax) = {wacc_details['kd_after_tax']:.2f}%**")
                    
                    st.markdown("---")
                    st.markdown("**WACC Calculation**")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write(f"Equity (E): **₹ {wacc_details['equity']:.2f} Lacs** ({wacc_details['we']:.2f}%)")
                        st.write(f"Debt (D): **₹ {wacc_details['debt']:.2f} Lacs** ({wacc_details['wd']:.2f}%)")
                        st.write(f"Total Capital (V): **₹ {wacc_details['equity'] + wacc_details['debt']:.2f} Lacs**")
                    
                    with col4:
                        st.write(f"WACC = (E/V × Ke) + (D/V × Kd × (1-Tax))")
                        st.write(f"WACC = ({wacc_details['we']:.2f}% × {wacc_details['ke']:.2f}%) + ({wacc_details['wd']:.2f}% × {wacc_details['kd_after_tax']:.2f}%)")
                        st.write(f"**WACC = {wacc_details['wacc']:.2f}%**")
                
                with tab5:
                    st.subheader("DCF Valuation Summary")
                    
                    st.markdown("### Terminal Value Calculation")
                    st.write(f"FCFF (Year {projection_years}): **₹ {projections['fcff'][-1]:.2f} Lacs**")
                    st.write(f"Terminal Growth Rate (g): **{terminal_growth}%**")
                    st.write(f"FCFF (Year {projection_years + 1}) = FCFF{projection_years} × (1 + g)")
                    st.write(f"FCFF (Year {projection_years + 1}) = ₹ {projections['fcff'][-1]:.2f} × (1 + {terminal_growth/100})")
                    st.write(f"FCFF (Year {projection_years + 1}) = **₹ {projections['fcff'][-1] * (1 + terminal_growth/100):.2f} Lacs**")
                    
                    st.write(f"\nTerminal Value = FCFF{projection_years + 1} / (WACC - g)")
                    st.write(f"Terminal Value = ₹ {projections['fcff'][-1] * (1 + terminal_growth/100):.2f} / ({wacc_details['wacc']:.2f}% - {terminal_growth}%)")
                    st.write(f"**Terminal Value = ₹ {valuation['terminal_value']:.2f} Lacs**")
                    
                    st.write(f"\nPV(Terminal Value) = TV / (1 + WACC)^{projection_years}")
                    st.write(f"**PV(Terminal Value) = ₹ {valuation['pv_terminal_value']:.2f} Lacs**")
                    
                    st.markdown("---")
                    st.markdown("### Enterprise Value")
                    
                    ev_df = pd.DataFrame({
                        'Component': ['Sum of PV(FCFF)', 'PV(Terminal Value)', 'Enterprise Value'],
                        'Value (₹ Lacs)': [
                            valuation['sum_pv_fcff'],
                            valuation['pv_terminal_value'],
                            valuation['enterprise_value']
                        ]
                    })
                    st.dataframe(ev_df.style.format({'Value (₹ Lacs)': '{:.2f}'}), use_container_width=True)
                    
                    tv_pct = valuation['tv_percentage']
                    if tv_pct > 90:
                        st.warning(f"⚠️ Terminal Value represents {tv_pct:.1f}% of Enterprise Value (>90% is high)")
                    else:
                        st.info(f"Terminal Value represents {tv_pct:.1f}% of Enterprise Value")
                    
                    st.markdown("---")
                    st.markdown("### Equity Value & Fair Value per Share")
                    
                    equity_calc_df = pd.DataFrame({
                        'Item': ['Enterprise Value', 'Less: Net Debt', 'Equity Value', 'Number of Shares', 'Fair Value per Share'],
                        'Value': [
                            f"₹ {valuation['enterprise_value']:.2f} Lacs",
                            f"₹ {valuation['net_debt']:.2f} Lacs",
                            f"₹ {valuation['equity_value']:.2f} Lacs",
                            f"{num_shares:,.0f}",
                            f"₹ {valuation['fair_value_per_share']:.2f}"
                        ]
                    })
                    st.table(equity_calc_df)
                    
                    st.success(f"### 🎯 Fair Value per Share: ₹ {valuation['fair_value_per_share']:.2f}")
                    
                    # Sensitivity Analysis
                    st.markdown("---")
                    st.subheader("📊 Sensitivity Analysis")
                    
                    wacc_range = np.arange(max(1.0, wacc_details['wacc'] - 3), wacc_details['wacc'] + 3.5, 0.5)
                    g_range = np.arange(max(1.0, terminal_growth - 2), min(terminal_growth + 3, wacc_details['wacc'] - 1), 0.5)
                    
                    if len(g_range) == 0:
                        g_range = np.array([terminal_growth])
                    
                    sensitivity_data = []
                    
                    for w in wacc_range:
                        row_data = {'WACC →': f"{w:.1f}%"}
                        for g_val in g_range:
                            if g_val >= w - 0.1:  # Need at least 0.1% gap
                                row_data[f"g={g_val:.1f}%"] = "N/A"
                            else:
                                try:
                                    fcff_n_plus_1 = projections['fcff'][-1] * (1 + g_val / 100)
                                    tv = fcff_n_plus_1 / ((w / 100) - (g_val / 100))
                                    pv_tv = tv / ((1 + w / 100) ** projection_years)
                                    ev = valuation['sum_pv_fcff'] + pv_tv
                                    eq_val = ev - valuation['net_debt']
                                    eq_val_rupees = eq_val * 100000
                                    fv = eq_val_rupees / num_shares if num_shares > 0 else 0
                                    row_data[f"g={g_val:.1f}%"] = f"₹{fv:.2f}"
                                except:
                                    row_data[f"g={g_val:.1f}%"] = "Error"
                        sensitivity_data.append(row_data)
                    
                    sensitivity_df = pd.DataFrame(sensitivity_data)
                    st.dataframe(sensitivity_df, use_container_width=True)
                    
                    st.caption("Sensitivity table shows Fair Value per Share for different WACC and terminal growth rate combinations")

# Footer
st.markdown("---")
st.caption("💡 **Note:** All values in ₹ Lacs unless specified otherwise | Built with traditional DCF methodology")