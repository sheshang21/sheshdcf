"""
DCF Integration Module for Screener Data
=========================================
Integrates Screener HTML parser and Excel handler with DCF valuation

This module provides:
1. Unified interface for both Screener web and Excel data
2. Data source selection UI
3. Integration with existing DCF calculation functions

Author: Shesh Ultimate
Version: 1.0
"""

import streamlit as st
from screener_data_parser import fetch_screener_financials_v2, convert_screener_to_dcf_format
from screener_excel_handler import (
    show_screener_upload_section,
    parse_screener_excel,
    display_screener_data_summary
)


def show_data_source_selector():
    """
    Display UI for selecting data source (Yahoo Finance, Screener, or Excel)
    
    Returns:
        tuple: (source_type, data_dict)
               source_type: 'yahoo', 'screener_web', or 'screener_excel'
               data_dict: Financial data in DCF format or None
    """
    
    st.markdown("### 📊 Data Source Selection")
    
    data_source = st.radio(
        "Choose your data source:",
        ["Yahoo Finance (Default)", "Screener.in Web", "Screener Excel Upload"],
        horizontal=True,
        help="Select how you want to fetch financial data"
    )
    
    if data_source == "Yahoo Finance (Default)":
        st.info("💡 Will fetch data from Yahoo Finance - Standard flow")
        return 'yahoo', None
    
    elif data_source == "Screener.in Web":
        st.markdown("---")
        st.markdown("### 🌐 Screener.in Web Scraping")
        st.warning("⚠️ **Note:** Web scraping may be unreliable. Consider using Excel upload for better accuracy.")
        
        symbol = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., NYKAA, RELIANCE, HDFCBANK",
            help="Enter the Screener.in symbol for the company"
        )
        
        num_years = st.number_input(
            "Number of years",
            min_value=2,
            max_value=10,
            value=5,
            help="How many years of historical data to extract"
        )
        
        if st.button("🔍 Fetch from Screener.in", type="primary"):
            if not symbol:
                st.error("❌ Please enter a stock symbol")
                return 'screener_web', None
            
            with st.spinner(f"Fetching data for {symbol} from Screener.in..."):
                # Fetch raw Screener data
                screener_data = fetch_screener_financials_v2(symbol, num_years)
                
                if screener_data:
                    # Convert to DCF format
                    dcf_data = convert_screener_to_dcf_format(screener_data)
                    
                    if dcf_data:
                        st.success("✅ Successfully fetched and converted Screener data!")
                        
                        # Display summary
                        with st.expander("📊 View Extracted Data", expanded=True):
                            display_screener_data_summary(dcf_data)
                        
                        return 'screener_web', dcf_data
                    else:
                        st.error("❌ Failed to convert Screener data to DCF format")
                        return 'screener_web', None
                else:
                    st.error("❌ Failed to fetch data from Screener.in")
                    return 'screener_web', None
        
        return 'screener_web', None
    
    elif data_source == "Screener Excel Upload":
        st.markdown("---")
        
        # Show upload section with template download
        use_excel, uploaded_file = show_screener_upload_section()
        
        if use_excel and uploaded_file:
            with st.spinner("📊 Parsing Excel file..."):
                # Parse the Excel file
                financials = parse_screener_excel(uploaded_file)
                
                if financials:
                    st.success("✅ Successfully parsed Excel file!")
                    
                    # Display summary
                    with st.expander("📊 View Extracted Data", expanded=True):
                        display_screener_data_summary(financials)
                    
                    return 'screener_excel', financials
                else:
                    st.error("❌ Failed to parse Excel file")
                    return 'screener_excel', None
        
        return 'screener_excel', None
    
    return 'yahoo', None


def integrate_screener_with_dcf(ticker, exchange_suffix, historical_years, use_screener=False, screener_data=None):
    """
    Integrate Screener data with existing DCF valuation flow
    
    Args:
        ticker: Stock ticker symbol
        exchange_suffix: 'NS' for NSE or 'BO' for BSE
        historical_years: Number of years for historical data
        use_screener: Boolean - whether to use Screener data
        screener_data: Dict with financial data from Screener (if use_screener=True)
    
    Returns:
        dict: Yahoo-compatible data structure with Screener data embedded
              or None to trigger standard Yahoo Finance fetch
    """
    
    if not use_screener or screener_data is None:
        # Return None to trigger standard Yahoo Finance fetch
        return None
    
    # Embed Screener data in a Yahoo-compatible structure
    # This allows the existing extract_financials_listed function to work
    yahoo_compatible = {
        '_screener_financials': screener_data,
        '_source': 'screener',
        '_ticker': ticker
    }
    
    st.info("📊 Using Screener data instead of Yahoo Finance")
    
    return yahoo_compatible


def get_shares_outstanding_from_screener(screener_data, ticker, exchange_suffix):
    """
    Calculate shares outstanding from Screener data
    
    Args:
        screener_data: Dict with Screener financial data
        ticker: Stock ticker
        exchange_suffix: Exchange suffix
    
    Returns:
        float: Number of shares in lakhs (or 0 if cannot determine)
    """
    
    if not screener_data:
        return 0
    
    try:
        # Method 1: Calculate from EPS and Net Profit
        if 'eps' in screener_data and 'net_profit' in screener_data:
            latest_eps = screener_data['eps'][0] if isinstance(screener_data['eps'], list) else 0
            latest_net_profit = screener_data['net_profit'][0] if isinstance(screener_data['net_profit'], list) else 0
            
            if latest_eps > 0 and latest_net_profit > 0:
                # Net Profit is in Rs. Crores, EPS is in Rs.
                # Shares = Net Profit (Cr) * 10^7 / EPS
                shares_actual = (latest_net_profit * 1e7) / latest_eps
                # Convert to lakhs
                shares_lakhs = shares_actual / 1e5
                
                st.info(f"📊 Calculated Shares Outstanding from EPS: {shares_lakhs:.2f} lakhs")
                return shares_lakhs
        
        # Method 2: Calculate from Equity Capital
        if 'equity_capital' in screener_data:
            equity_capital = screener_data['equity_capital'][0] if isinstance(screener_data['equity_capital'], list) else 0
            
            if equity_capital > 0:
                # Equity capital is in Rs. Crores
                # Assuming face value of Rs. 10 (most common in India)
                # Shares = (Equity Capital in Cr * 10^7) / Face Value / 10^5 (to convert to lakhs)
                shares_lakhs = (equity_capital * 1e7) / 10 / 1e5
                
                st.info(f"📊 Calculated Shares Outstanding from Equity Capital: {shares_lakhs:.2f} lakhs (assuming Rs. 10 face value)")
                return shares_lakhs
        
        # Method 3: Fallback to Yahoo Finance
        st.warning("⚠️ Could not determine shares outstanding from Screener data. Will fetch from Yahoo Finance.")
        import yfinance as yf
        
        ticker_symbol = f"{ticker}.{exchange_suffix}"
        stock = yf.Ticker(ticker_symbol)
        shares_outstanding = stock.info.get('sharesOutstanding', 0)
        
        if shares_outstanding > 0:
            shares_lakhs = shares_outstanding / 1e5
            st.success(f"✅ Fetched Shares Outstanding from Yahoo Finance: {shares_lakhs:.2f} lakhs")
            return shares_lakhs
    
    except Exception as e:
        st.error(f"❌ Error calculating shares outstanding: {e}")
    
    return 0
