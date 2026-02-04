"""
Screener Excel Handler
======================
Handles all functionality related to uploaded Screener Excel files.

This module provides:
1. Template download
2. Excel file upload and parsing
3. Financial data extraction from Screener format
4. Integration with existing DCF logic

Author: Shesh Ultimate
Version: 1.0
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def show_screener_upload_section():
    """
    Display the Screener Excel upload section with template download
    Returns: (use_screener_excel, uploaded_file)
    """
    st.markdown("---")
    st.markdown("### 📊 Screener Excel Upload Mode")
    
    use_screener_excel = st.checkbox(
        "📥 Upload Screener Excel", 
        help="Use data from a Screener.in Excel export instead of fetching from Yahoo Finance"
    )
    
    uploaded_file = None
    
    if use_screener_excel:
        st.info("📌 **Excel Upload Mode Active** - All calculations will use your uploaded data only")
        
        # Download template button
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                # Read template from uploads folder
                with open('/mnt/user-data/uploads/Screener_template.xlsx', 'rb') as f:
                    template_bytes = f.read()
                
                st.download_button(
                    label="📥 Download Template",
                    data=template_bytes,
                    file_name="Screener_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download this template and fill it with data from Screener.in"
                )
            except Exception as e:
                st.warning(f"⚠️ Template download unavailable. Make sure Screener_template.xlsx is in uploads folder.")
        
        with col2:
            st.caption("""
            **Instructions:**
            1. Download the template
            2. Go to Screener.in → Select company
            3. Copy Balance Sheet and P&L data
            4. Paste into template (keep same format)
            5. Upload below
            """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Screener Excel File",
            type=['xlsx', 'xls'],
            help="Upload the filled template with Screener data"
        )
        
        if uploaded_file:
            st.success("✅ File uploaded successfully!")
    
    return use_screener_excel, uploaded_file


def parse_screener_excel(uploaded_file):
    """
    Parse uploaded Screener Excel file and extract financial data
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        dict: Parsed financial data in the same format as Yahoo Finance extraction
              {
                'years': [...],
                'revenue': [...],
                'cogs': [...],
                'opex': [...],
                'ebitda': [...],
                'depreciation': [...],
                'ebit': [...],
                'interest': [...],
                'tax': [...],
                'nopat': [...],
                'fixed_assets': [...],
                'inventory': [...],
                'receivables': [...],
                'payables': [...],
                'cash': [...],
                'equity': [...],
                'st_debt': [...],
                'lt_debt': [...],
                'interest_income': [...]
              }
    """
    try:
        # Read both sheets
        df_bs = pd.read_excel(uploaded_file, sheet_name='BalanceSheet')
        df_pl = pd.read_excel(uploaded_file, sheet_name='P&L')
        
        st.write("📊 Excel file loaded successfully")
        st.write(f"  - P&L sheet: {df_pl.shape[0]} rows x {df_pl.shape[1]} columns")
        st.write(f"  - Balance Sheet: {df_bs.shape[0]} rows x {df_bs.shape[1]} columns")
        
        # Detect year columns - first row should have "Mar 2024" style headers
        year_columns = []
        header_row = df_pl.iloc[0]  # First row contains headers like "Mar 2024"
        
        for col_idx, col in enumerate(df_pl.columns[1:], start=1):  # Skip first column (item names)
            try:
                date_text = str(header_row.iloc[col_idx])
                if pd.notna(date_text) and date_text not in ['nan', 'TTM']:
                    # Extract year from "Mar 2024" or similar format
                    if any(month in date_text for month in ['Mar', 'Sep', 'Dec', 'Jun']):
                        parts = date_text.split()
                        if len(parts) >= 2 and parts[-1].isdigit():
                            year = int(parts[-1])
                            year_columns.append((col, year))
                            st.write(f"  ✓ Found year: {date_text} → {year}")
            except Exception as e:
                continue
        
        if len(year_columns) == 0:
            raise ValueError("Could not detect year columns in Excel file. Ensure first row has 'Mar 2024' style headers.")
        
        # Extract years
        years = [year for _, year in year_columns]
        col_indices = [col for col, _ in year_columns]
        
        st.success(f"✅ Detected {len(years)} years: {years}")
        
        # Helper function to extract value from dataframe matching exact Screener field names
        def get_value_by_exact_name(df, item_name, col):
            """Extract value for a specific item and column by exact field name match"""
            try:
                # Look through all rows for exact match (case-insensitive)
                for idx, row in df.iterrows():
                    label = str(row.iloc[0]).strip()
                    if label.lower() == item_name.lower():
                        val = row.iloc[col]
                        if pd.notna(val) and val != '' and val != '-':
                            # Handle negative values
                            if isinstance(val, str) and val.startswith('-'):
                                return -float(val[1:].replace(',', ''))
                            return float(str(val).replace(',', ''))
                        return 0.0
            except Exception as e:
                st.write(f"  ⚠️ Error extracting {item_name}: {e}")
            return 0.0
        
        # Initialize financials dictionary
        financials = {
            'years': years,
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
            'lt_debt': []
        }
        
        st.write("### 📊 Extracting P&L Data")
        
        # Extract data for each year using EXACT Screener field names
        for col_idx, col in enumerate(col_indices):
            # P&L Items - Use exact Screener field names
            sales = get_value_by_exact_name(df_pl, 'Sales', col)
            st.write(f"  Year {years[col_idx]}: Sales = {sales}")
            
            expenses = get_value_by_exact_name(df_pl, 'Expenses', col)
            operating_profit = get_value_by_exact_name(df_pl, 'Operating Profit', col)
            other_income = get_value_by_exact_name(df_pl, 'Other Income', col)
            interest = get_value_by_exact_name(df_pl, 'Interest', col)
            depreciation = get_value_by_exact_name(df_pl, 'Depreciation', col)
            profit_before_tax = get_value_by_exact_name(df_pl, 'Profit before tax', col)
            
            # Try to get tax percentage
            tax_pct_text = get_value_by_exact_name(df_pl, 'Tax %', col)
            # Convert percentage to actual tax amount
            if tax_pct_text and profit_before_tax:
                # If tax_pct_text is like "42%" or "42", convert to decimal
                try:
                    tax_pct = float(str(tax_pct_text).replace('%', ''))
                    tax = profit_before_tax * (tax_pct / 100)
                except:
                    tax = 0
            else:
                tax = 0
            
            net_profit = get_value_by_exact_name(df_pl, 'Net Profit', col)
            
            # If we don't have tax from percentage, calculate from PBT - Net Profit
            if tax == 0 and profit_before_tax > 0 and net_profit > 0:
                tax = profit_before_tax - net_profit
            
            # Calculate derived P&L items
            # EBITDA = Operating Profit + Depreciation
            ebitda = operating_profit + depreciation
            
            # EBIT = Operating Profit (since Operating Profit = EBITDA - Depreciation)
            ebit = operating_profit
            
            # COGS and OpEx estimation
            # Total Cost = Sales - Operating Profit
            total_cost = sales - operating_profit if sales > operating_profit else expenses
            
            # Default split: 60% COGS, 40% OpEx
            cogs = total_cost * 0.6
            opex = total_cost * 0.4
            
            # NOPAT = EBIT * (1 - Tax Rate)
            tax_rate = (tax / profit_before_tax) if profit_before_tax > 0 and tax > 0 else 0.25
            nopat = ebit * (1 - tax_rate)
            
            # Interest income (assume 50% of other income)
            interest_income = other_income * 0.5
            
            st.write("### 🏦 Extracting Balance Sheet Data")
            
            # Balance Sheet Items - Use exact Screener field names
            equity_capital = get_value_by_exact_name(df_bs, 'Equity Capital', col)
            reserves = get_value_by_exact_name(df_bs, 'Reserves', col)
            equity = equity_capital + reserves
            
            borrowings = get_value_by_exact_name(df_bs, 'Borrowings', col)
            # Split borrowings into ST and LT (assume 30% ST, 70% LT)
            st_debt = borrowings * 0.3
            lt_debt = borrowings * 0.7
            
            # Assets
            fixed_assets = get_value_by_exact_name(df_bs, 'Fixed Assets', col)
            cwip = get_value_by_exact_name(df_bs, 'CWIP', col)
            fixed_assets += cwip  # Add CWIP to fixed assets
            
            other_assets = get_value_by_exact_name(df_bs, 'Other Assets', col)
            other_liabilities = get_value_by_exact_name(df_bs, 'Other Liabilities', col)
            
            # Working capital items (estimated from aggregated values)
            # Screener aggregates these, so we estimate
            inventory = other_assets * 0.2
            receivables = other_assets * 0.3
            cash = other_assets * 0.2
            payables = other_liabilities * 0.4
            
            # Store in financials dict
            financials['revenue'].append(sales)
            financials['cogs'].append(cogs)
            financials['opex'].append(opex)
            financials['ebitda'].append(ebitda)
            financials['depreciation'].append(depreciation)
            financials['ebit'].append(ebit)
            financials['interest'].append(interest)
            financials['interest_income'].append(interest_income)
            financials['tax'].append(tax)
            financials['nopat'].append(nopat)
            financials['fixed_assets'].append(fixed_assets)
            financials['inventory'].append(inventory)
            financials['receivables'].append(receivables)
            financials['payables'].append(payables)
            financials['cash'].append(cash)
            financials['equity'].append(equity)
            financials['st_debt'].append(st_debt)
            financials['lt_debt'].append(lt_debt)
        
        st.success(f"✅ Extracted financial data for {len(years)} years")
        return financials
    
    except Exception as e:
        st.error(f"❌ Error parsing Excel file: {str(e)}")
        st.error("Please ensure the file matches the template format")
        return None


def show_manual_peer_input():
    """
    Display manual peer input section for comparable valuation
    
    Returns:
        list: List of peer dictionaries with name, pe, ev_ebitda
    """
    st.markdown("### 📊 Manual Peer Comparison")
    st.info("💡 In Excel upload mode, you must manually enter peer data")
    
    num_peers = st.number_input(
        "Number of Peers",
        min_value=1,
        max_value=10,
        value=3,
        help="How many peer companies do you want to compare?"
    )
    
    peers = []
    
    for i in range(int(num_peers)):
        st.markdown(f"**Peer {i+1}**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input(
                f"Company Name",
                key=f"peer_name_{i}",
                placeholder="e.g., HDFC Bank"
            )
        
        with col2:
            pe = st.number_input(
                f"P/E Ratio",
                min_value=0.0,
                value=20.0,
                step=0.1,
                key=f"peer_pe_{i}"
            )
        
        with col3:
            ev_ebitda = st.number_input(
                f"EV/EBITDA",
                min_value=0.0,
                value=12.0,
                step=0.1,
                key=f"peer_ev_ebitda_{i}"
            )
        
        if name:
            peers.append({
                'name': name,
                'pe': pe,
                'ev_ebitda': ev_ebitda
            })
    
    return peers


def calculate_manual_peer_valuation(financials, shares, peers):
    """
    Calculate relative valuation using manually entered peer multiples
    
    Args:
        financials: dict - Financial data
        shares: float - Number of shares
        peers: list - List of peer dictionaries
    
    Returns:
        dict: Valuation results
    """
    if not peers or len(peers) == 0:
        return None
    
    # Calculate average multiples
    avg_pe = np.mean([p['pe'] for p in peers])
    avg_ev_ebitda = np.mean([p['ev_ebitda'] for p in peers])
    
    # Latest financials (index 0 is newest)
    latest_net_profit = financials['revenue'][0] - financials['cogs'][0] - financials['opex'][0] - financials['depreciation'][0] - financials['interest'][0] - financials['tax'][0]
    latest_ebitda = financials['ebitda'][0]
    
    # P/E Valuation
    eps = latest_net_profit / shares if shares > 0 else 0
    pe_value_per_share = eps * avg_pe
    
    # EV/EBITDA Valuation
    enterprise_value = latest_ebitda * avg_ev_ebitda
    net_debt = (financials['st_debt'][0] + financials['lt_debt'][0]) - financials['cash'][0]
    equity_value = enterprise_value - net_debt
    ev_ebitda_value_per_share = equity_value / shares if shares > 0 else 0
    
    # Average valuation
    avg_value = (pe_value_per_share + ev_ebitda_value_per_share) / 2
    
    return {
        'pe_multiple': avg_pe,
        'ev_ebitda_multiple': avg_ev_ebitda,
        'pe_value': pe_value_per_share,
        'ev_ebitda_value': ev_ebitda_value_per_share,
        'avg_value': avg_value,
        'peers': peers
    }


def display_screener_data_summary(financials):
    """Display summary of extracted Screener data"""
    st.markdown("### 📊 Extracted Financial Data")
    
    # Create summary dataframe
    years_display = [str(y) for y in financials['years']]
    
    summary_data = {
        'Year': years_display,
        'Revenue': [f"{v:.0f}" for v in financials['revenue']],
        'EBITDA': [f"{v:.0f}" for v in financials['ebitda']],
        'EBIT': [f"{v:.0f}" for v in financials['ebit']],
        'Interest': [f"{v:.0f}" for v in financials['interest']],
        'Depreciation': [f"{v:.0f}" for v in financials['depreciation']],
        'Equity': [f"{v:.0f}" for v in financials['equity']],
        'Debt': [f"{v:.0f}" for v in [financials['st_debt'][i] + financials['lt_debt'][i] for i in range(len(financials['years']))]],
        'Fixed Assets': [f"{v:.0f}" for v in financials['fixed_assets']]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)
    
    # Working Capital Summary
    st.markdown("**Working Capital Components:**")
    wc_data = {
        'Year': years_display,
        'Inventory': [f"{v:.0f}" for v in financials['inventory']],
        'Receivables': [f"{v:.0f}" for v in financials['receivables']],
        'Payables': [f"{v:.0f}" for v in financials['payables']],
        'Cash': [f"{v:.0f}" for v in financials['cash']]
    }
    
    df_wc = pd.DataFrame(wc_data)
    st.dataframe(df_wc, use_container_width=True)
    
    st.caption("💡 All figures in Rs. Crores (as per Screener format)")
