"""
Advanced Peer Comparison Visualizations
========================================
Interactive 3D charts and comprehensive peer analysis
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st


def fetch_peer_financials(ticker_list, target_ticker=None, exchange_suffix="NS"):
    """
    Fetch comprehensive financial data for all peer companies
    Returns: DataFrame with all metrics
    """
    # Use the shared CachedTickerData from DCF engine to prevent rate limits
    try:
        from PHASE5_DCF_valuation import CachedTickerData
    except ImportError:
        CachedTickerData = None

    peer_data = []
    
    for ticker in ticker_list:
        try:
            # Add exchange suffix if not already present
            if '.NS' not in ticker and '.BO' not in ticker:
                ticker_full = f"{ticker}.{exchange_suffix}"
            else:
                ticker_full = ticker
            
            stock = CachedTickerData(ticker_full) if CachedTickerData else yf.Ticker(ticker_full)
            info = stock.info
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            
            # Extract latest year data
            if not financials.empty and not balance_sheet.empty:
                latest_year = financials.columns[0]
                
                # Revenue
                revenue = abs(financials.loc['Total Revenue', latest_year]) if 'Total Revenue' in financials.index else 0
                
                # Operating metrics
                operating_income = abs(financials.loc['Operating Income', latest_year]) if 'Operating Income' in financials.index else 0
                net_income = abs(financials.loc['Net Income', latest_year]) if 'Net Income' in financials.index else 0
                ebitda = info.get('ebitda', 0)
                
                # Balance sheet
                total_assets = abs(balance_sheet.loc['Total Assets', latest_year]) if 'Total Assets' in balance_sheet.index else 0
                total_debt = abs(balance_sheet.loc['Long Term Debt', latest_year]) if 'Long Term Debt' in balance_sheet.index else 0
                equity = abs(balance_sheet.loc['Stockholders Equity', latest_year]) if 'Stockholders Equity' in balance_sheet.index else 0
                
                # Market data
                market_cap = info.get('marketCap', 0)
                current_price = info.get('currentPrice', 0)
                
                # Calculate metrics
                operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
                net_margin = (net_income / revenue * 100) if revenue > 0 else 0
                roe = (net_income / equity * 100) if equity > 0 else 0
                roa = (net_income / total_assets * 100) if total_assets > 0 else 0
                debt_to_equity = (total_debt / equity) if equity > 0 else 0
                
                # Valuation multiples
                pe = info.get('trailingPE', 0)
                pb = info.get('priceToBook', 0)
                ps = (market_cap / revenue) if revenue > 0 else 0
                ev_ebitda = info.get('enterpriseToEbitda', 0)
                
                peer_data.append({
                    'ticker': ticker,
                    'company': info.get('longName', ticker),
                    'revenue': revenue / 100000,  # Convert Rupees to Lacs
                    'operating_income': operating_income / 100000,
                    'net_income': net_income / 100000,
                    'ebitda': ebitda / 100000 if ebitda else 0,
                    'market_cap': market_cap / 100000,
                    'total_assets': total_assets / 100000,
                    'equity': equity / 100000,
                    'total_debt': total_debt / 100000,
                    'current_price': current_price,
                    'operating_margin': operating_margin,
                    'net_margin': net_margin,
                    'roe': roe,
                    'roa': roa,
                    'debt_to_equity': debt_to_equity,
                    'pe': pe if pe and pe > 0 and pe < 100 else 0,
                    'pb': pb if pb and pb > 0 and pb < 20 else 0,
                    'ps': ps,
                    'ev_ebitda': ev_ebitda if ev_ebitda and ev_ebitda > 0 else 0,
                    'is_target': ticker == target_ticker
                })
                
                st.success(f"✅ {ticker}: ₹{revenue/100000:.0f} Lacs revenue")
            else:
                st.warning(f"⚠️ {ticker}: No financial data")
                
        except Exception as e:
            st.error(f"❌ {ticker}: {str(e)}")
            continue
    
    if not peer_data:
        return None
    
    return pd.DataFrame(peer_data)


def create_3d_scatter_revenue_margin_valuation(df, target_ticker=None):
    """
    3D Scatter: Revenue vs Operating Margin vs P/E Ratio
    Bubble size = Market Cap
    """
    # Filter valid data
    df_plot = df[(df['revenue'] > 0) & (df['operating_margin'] > 0) & (df['pe'] > 0)].copy()
    
    if df_plot.empty:
        return None
    
    # Color based on whether it's target or peer
    df_plot['color'] = df_plot['ticker'].apply(
        lambda x: 'Target Company' if x == target_ticker else 'Peer'
    )
    
    fig = go.Figure()
    
    # Add peers
    peers = df_plot[df_plot['color'] == 'Peer']
    if not peers.empty:
        fig.add_trace(go.Scatter3d(
            x=peers['revenue'],
            y=peers['operating_margin'],
            z=peers['pe'],
            mode='markers+text',
            marker=dict(
                size=peers['market_cap'] / peers['market_cap'].max() * 30 + 10,
                color=peers['market_cap'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Market Cap<br>(₹ Lacs)", x=1.1),
                line=dict(width=0.5, color='white')
            ),
            text=peers['ticker'],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            name='Peers',
            hovertemplate='<b>%{text}</b><br>' +
                         'Revenue: ₹%{x:.2f} Lacs<br>' +
                         'Op. Margin: %{y:.1f}%<br>' +
                         'P/E: %{z:.1f}x<br>' +
                         '<extra></extra>'
        ))
    
    # Add target
    target = df_plot[df_plot['color'] == 'Target Company']
    if not target.empty:
        fig.add_trace(go.Scatter3d(
            x=target['revenue'],
            y=target['operating_margin'],
            z=target['pe'],
            mode='markers+text',
            marker=dict(
                size=30,
                color='red',
                symbol='diamond',
                line=dict(width=2, color='yellow')
            ),
            text=target['ticker'],
            textposition='top center',
            textfont=dict(size=14, color='red', family='Arial Black'),
            name='Target',
            hovertemplate='<b>%{text} (TARGET)</b><br>' +
                         'Revenue: ₹%{x:.2f} Lacs<br>' +
                         'Op. Margin: %{y:.1f}%<br>' +
                         'P/E: %{z:.1f}x<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='3D Peer Comparison: Revenue vs Profitability vs Valuation',
        scene=dict(
            xaxis=dict(title='Revenue (₹ Lacsores)', gridcolor='lightgray'),
            yaxis=dict(title='Operating Margin (%)', gridcolor='lightgray'),
            zaxis=dict(title='P/E Ratio (x)', gridcolor='lightgray'),
            bgcolor='rgb(240,240,240)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    return fig


def create_revenue_comparison_bar(df, target_ticker=None):
    """
    Latest Financial Year Revenue Comparison
    With growth indicators if available
    """
    df_sorted = df.sort_values('revenue', ascending=False)
    
    # Create color array
    colors = ['red' if row['ticker'] == target_ticker else '#2E86AB' 
              for _, row in df_sorted.iterrows()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted['ticker'],
        y=df_sorted['revenue'],
        marker_color=colors,
        text=[f"₹{val:.2f} Lacs" for val in df_sorted['revenue']],
        textposition='outside',
        textfont=dict(size=11, color='black'),
        hovertemplate='<b>%{x}</b><br>' +
                     'Revenue: ₹%{y:.2f} Lacs<br>' +
                     '<extra></extra>'
    ))
    
    # Add target indicator
    if target_ticker:
        target_revenue = df_sorted[df_sorted['ticker'] == target_ticker]['revenue'].values
        if len(target_revenue) > 0:
            fig.add_hline(
                y=target_revenue[0],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target: ₹{target_revenue[0]:.0f}Cr",
                annotation_position="right"
            )
    
    fig.update_layout(
        title=dict(
            text='Latest Financial Year Revenue Comparison',
            font=dict(size=20, color='#2C3E50', family='Arial Black')
        ),
        xaxis_title='Company',
        yaxis_title='Revenue (₹ Lacsores)',
        height=500,
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor='lightgray'),
        plot_bgcolor='white',
        showlegend=False
    )
    
    return fig


def create_profitability_comparison(df, target_ticker=None):
    """
    Multi-dimensional profitability comparison
    Operating Margin, Net Margin, ROE, ROA
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Operating Margin (%)', 'Net Margin (%)', 
                       'Return on Equity (%)', 'Return on Assets (%)'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Sort by revenue for consistent ordering
    df_sorted = df.sort_values('revenue', ascending=False)
    
    colors = ['red' if row['ticker'] == target_ticker else '#06A77D' 
              for _, row in df_sorted.iterrows()]
    
    # Operating Margin
    fig.add_trace(
        go.Bar(x=df_sorted['ticker'], y=df_sorted['operating_margin'],
               marker_color=colors, name='Op. Margin',
               text=[f"{val:.1f}%" for val in df_sorted['operating_margin']],
               textposition='outside'),
        row=1, col=1
    )
    
    # Net Margin
    colors2 = ['red' if row['ticker'] == target_ticker else '#F77F00' 
               for _, row in df_sorted.iterrows()]
    fig.add_trace(
        go.Bar(x=df_sorted['ticker'], y=df_sorted['net_margin'],
               marker_color=colors2, name='Net Margin',
               text=[f"{val:.1f}%" for val in df_sorted['net_margin']],
               textposition='outside'),
        row=1, col=2
    )
    
    # ROE
    colors3 = ['red' if row['ticker'] == target_ticker else '#9D4EDD' 
               for _, row in df_sorted.iterrows()]
    fig.add_trace(
        go.Bar(x=df_sorted['ticker'], y=df_sorted['roe'],
               marker_color=colors3, name='ROE',
               text=[f"{val:.1f}%" for val in df_sorted['roe']],
               textposition='outside'),
        row=2, col=1
    )
    
    # ROA
    colors4 = ['red' if row['ticker'] == target_ticker else '#D62828' 
               for _, row in df_sorted.iterrows()]
    fig.add_trace(
        go.Bar(x=df_sorted['ticker'], y=df_sorted['roa'],
               marker_color=colors4, name='ROA',
               text=[f"{val:.1f}%" for val in df_sorted['roa']],
               textposition='outside'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Comprehensive Profitability Comparison"
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig


def create_valuation_multiples_radar(df, target_ticker):
    """
    Radar chart comparing target vs peer average on valuation multiples
    Normalized to show relative positioning
    """
    if target_ticker not in df['ticker'].values:
        return None
    
    target = df[df['ticker'] == target_ticker].iloc[0]
    peers = df[df['ticker'] != target_ticker]
    
    if peers.empty:
        return None
    
    # Calculate peer averages
    peer_avg = peers[['pe', 'pb', 'ps', 'ev_ebitda', 'operating_margin', 'net_margin']].mean()
    
    # Get target values
    target_vals = target[['pe', 'pb', 'ps', 'ev_ebitda', 'operating_margin', 'net_margin']]
    
    categories = ['P/E', 'P/B', 'P/S', 'EV/EBITDA', 'Op. Margin', 'Net Margin']
    
    fig = go.Figure()
    
    # Peer average
    fig.add_trace(go.Scatterpolar(
        r=peer_avg.values,
        theta=categories,
        fill='toself',
        name='Peer Average',
        line=dict(color='#2E86AB', width=2),
        fillcolor='rgba(46, 134, 171, 0.3)'
    ))
    
    # Target company
    fig.add_trace(go.Scatterpolar(
        r=target_vals.values,
        theta=categories,
        fill='toself',
        name=f'{target_ticker} (Target)',
        line=dict(color='red', width=3),
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(peer_avg.max(), target_vals.max()) * 1.2])
        ),
        showlegend=True,
        title='Target vs Peer Average - Valuation & Profitability Radar',
        height=600
    )
    
    return fig


def create_balance_sheet_comparison(df, target_ticker=None):
    """
    Stacked bar chart: Assets, Equity, Debt comparison
    """
    df_sorted = df.sort_values('total_assets', ascending=False)
    
    fig = go.Figure()
    
    # Equity
    fig.add_trace(go.Bar(
        name='Equity',
        x=df_sorted['ticker'],
        y=df_sorted['equity'],
        marker_color='#06A77D',
        text=[f"₹{val:.2f} Lacs" for val in df_sorted['equity']],
        textposition='inside'
    ))
    
    # Debt
    fig.add_trace(go.Bar(
        name='Debt',
        x=df_sorted['ticker'],
        y=df_sorted['total_debt'],
        marker_color='#D62828',
        text=[f"₹{val:.2f} Lacs" for val in df_sorted['total_debt']],
        textposition='inside'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Capital Structure Comparison',
        xaxis_title='Company',
        yaxis_title='Amount (₹ Lacsores)',
        height=500,
        xaxis=dict(tickangle=-45),
        legend=dict(x=0.8, y=0.95)
    )
    
    return fig


def create_3d_bubble_market_cap_revenue_margin(df, target_ticker=None):
    """
    3D Bubble Chart: Market Cap vs Revenue vs Net Margin
    Color by debt-to-equity ratio
    """
    df_plot = df[(df['market_cap'] > 0) & (df['revenue'] > 0)].copy()
    
    if df_plot.empty:
        return None
    
    # Normalize debt to equity for color scale
    df_plot['d_e_normalized'] = df_plot['debt_to_equity'].clip(0, 3)
    
    fig = go.Figure()
    
    # Peers
    peers = df_plot[df_plot['ticker'] != target_ticker]
    if not peers.empty:
        fig.add_trace(go.Scatter3d(
            x=peers['market_cap'],
            y=peers['revenue'],
            z=peers['net_margin'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=peers['d_e_normalized'],
                colorscale='RdYlGn_r',  # Red = high debt
                showscale=True,
                colorbar=dict(title="Debt/Equity", x=1.1),
                line=dict(width=0.5, color='white')
            ),
            text=peers['ticker'],
            textposition='top center',
            name='Peers',
            hovertemplate='<b>%{text}</b><br>' +
                         'Market Cap: ₹%{x:.2f} Lacs<br>' +
                         'Revenue: ₹%{y:.2f} Lacs<br>' +
                         'Net Margin: %{z:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    # Target
    if target_ticker in df_plot['ticker'].values:
        target = df_plot[df_plot['ticker'] == target_ticker]
        fig.add_trace(go.Scatter3d(
            x=target['market_cap'],
            y=target['revenue'],
            z=target['net_margin'],
            mode='markers+text',
            marker=dict(
                size=25,
                color='red',
                symbol='diamond',
                line=dict(width=3, color='yellow')
            ),
            text=target['ticker'],
            textposition='top center',
            textfont=dict(size=14, color='red', family='Arial Black'),
            name='Target',
            hovertemplate='<b>%{text} (TARGET)</b><br>' +
                         'Market Cap: ₹%{x:.2f} Lacs<br>' +
                         'Revenue: ₹%{y:.2f} Lacs<br>' +
                         'Net Margin: %{z:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='3D Analysis: Market Cap vs Revenue vs Profitability (Color = Leverage)',
        scene=dict(
            xaxis=dict(title='Market Cap (₹ Lacs)'),
            yaxis=dict(title='Revenue (₹ Lacs)'),
            zaxis=dict(title='Net Margin (%)'),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.3))
        ),
        height=700,
        showlegend=True
    )
    
    return fig


def create_peer_comparison_dashboard(ticker, peer_tickers_str, unlisted_data=None, exchange_suffix="NS"):
    """
    Main function to generate complete peer comparison dashboard
    
    Parameters:
    -----------
    ticker : str
        Target company ticker (for listed) or synthetic name (for unlisted)
    peer_tickers_str : str
        Comma-separated peer tickers (can include .NS or .BO suffixes)
    unlisted_data : dict, optional
        Financial data for unlisted company. If provided, creates synthetic peer data.
        Expected keys: 'revenue', 'ebitda', 'nopat', 'equity', 'st_debt', 'lt_debt', etc.
    exchange_suffix : str
        Exchange suffix - only used if tickers don't already have .NS or .BO
    """
    if unlisted_data:
        st.header(f"🏢 Comprehensive Peer Analysis: {ticker} (Unlisted)")
    else:
        st.header(f"🏢 Comprehensive Peer Analysis: {ticker}")
    
    # Parse peer tickers
    peer_list = [t.strip() for t in peer_tickers_str.split(',') if t.strip()]
    
    if unlisted_data:
        # For unlisted companies, only fetch peer data
        all_tickers = peer_list
        st.info(f"📊 Analyzing unlisted company vs {len(peer_list)} listed peers")
    else:
        # For listed companies, include target
        all_tickers = [ticker] + peer_list
        st.info(f"📊 Analyzing {len(all_tickers)} companies")
    
    # Fetch data for peers (and target if listed)
    with st.spinner("Fetching financial data for peer companies..."):
        if unlisted_data:
            df_peers = fetch_peer_financials(peer_list, target_ticker=None, exchange_suffix=exchange_suffix)
        else:
            df_peers = fetch_peer_financials(all_tickers, target_ticker=ticker, exchange_suffix=exchange_suffix)
    
    if df_peers is None or df_peers.empty:
        st.error("❌ Could not fetch peer data")
        return
    
    # If unlisted, create synthetic data row for target company
    if unlisted_data:
        st.info("🔧 Creating synthetic data for unlisted company from financial inputs...")
        
        # Extract data from unlisted_data (values are in Lacs)
        revenue_lacs = unlisted_data['revenue'][-1] if 'revenue' in unlisted_data else 0
        ebitda_lacs = unlisted_data['ebitda'][-1] if 'ebitda' in unlisted_data else 0
        nopat_lacs = unlisted_data['nopat'][-1] if 'nopat' in unlisted_data else 0
        equity_lacs = unlisted_data['equity'][-1] if 'equity' in unlisted_data else 0
        st_debt_lacs = unlisted_data['st_debt'][-1] if 'st_debt' in unlisted_data else 0
        lt_debt_lacs = unlisted_data['lt_debt'][-1] if 'lt_debt' in unlisted_data else 0
        fixed_assets_lacs = unlisted_data['fixed_assets'][-1] if 'fixed_assets' in unlisted_data else 0
        
        # Keep values in Lacs (same unit as peer data)
        revenue = revenue_lacs
        ebitda = ebitda_lacs
        nopat = nopat_lacs
        equity = equity_lacs
        total_debt = st_debt_lacs + lt_debt_lacs
        total_assets = fixed_assets_lacs  # Simplified
        
        # Calculate metrics
        operating_income = ebitda  # Approximation
        net_income = nopat  # Using NOPAT as proxy
        
        operating_margin = (operating_income / revenue * 100) if revenue > 0 else 0
        net_margin = (net_income / revenue * 100) if revenue > 0 else 0
        roe = (net_income / equity * 100) if equity > 0 else 0
        roa = (net_income / total_assets * 100) if total_assets > 0 else 0
        debt_to_equity = (total_debt / equity) if equity > 0 else 0
        
        # For valuation multiples, use peer averages as benchmarks
        peer_avg_pe = df_peers['pe'].mean() if not df_peers.empty else 15
        peer_avg_pb = df_peers['pb'].mean() if not df_peers.empty else 2
        peer_avg_ps = df_peers['ps'].mean() if not df_peers.empty else 1
        peer_avg_ev_ebitda = df_peers['ev_ebitda'].mean() if not df_peers.empty else 10
        
        # Estimated market cap using peer average P/E (in Lacs)
        estimated_market_cap = net_income * peer_avg_pe if net_income > 0 else equity * peer_avg_pb
        estimated_price = 100  # Placeholder - no real price for unlisted
        
        # Create synthetic row
        unlisted_row = {
            'ticker': ticker,
            'company': f'{ticker} (Unlisted)',
            'revenue': revenue,
            'operating_income': operating_income,
            'net_income': net_income,
            'ebitda': ebitda,
            'market_cap': estimated_market_cap,
            'total_assets': total_assets,
            'equity': equity,
            'total_debt': total_debt,
            'current_price': estimated_price,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'roe': roe,
            'roa': roa,
            'debt_to_equity': debt_to_equity,
            'pe': peer_avg_pe,  # Using peer average
            'pb': peer_avg_pb,  # Using peer average
            'ps': peer_avg_ps,  # Using peer average
            'ev_ebitda': peer_avg_ev_ebitda,  # Using peer average
            'is_target': True
        }
        
        # Combine with peer data
        df = pd.concat([pd.DataFrame([unlisted_row]), df_peers], ignore_index=True)
        
        st.success(f"✅ Created synthetic data for {ticker}")
        st.caption("📌 Note: Valuation multiples (P/E, P/B, etc.) are estimated using peer averages since the company is unlisted")
        st.caption(f"   Revenue: ₹{revenue:.2f} Lacs | EBITDA: ₹{ebitda:.2f} Lacs | Equity: ₹{equity:.2f} Lacs")
    else:
        df = df_peers
    
    st.success(f"✅ Successfully loaded data for {len(df)} companies")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Revenue & Size",
        "💰 Profitability",
        "📈 Valuation",
        "🏦 Balance Sheet",
        "🌐 3D Analysis",
        "📋 Data Table"
    ])
    
    with tab1:
        st.subheader("Revenue Comparison - Latest FY")
        fig1 = create_revenue_comparison_bar(df, ticker)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        
        # Market cap comparison
        st.subheader("Market Capitalization")
        df_sorted = df.sort_values('market_cap', ascending=False)
        fig_mc = go.Figure(go.Bar(
            x=df_sorted['ticker'],
            y=df_sorted['market_cap'],
            marker_color=['red' if t == ticker else '#2E86AB' for t in df_sorted['ticker']],
            text=[f"₹{val:.2f} Lacs" for val in df_sorted['market_cap']],
            textposition='outside'
        ))
        fig_mc.update_layout(
            title='Market Cap Comparison',
            xaxis_title='Company',
            yaxis_title='Market Cap (₹ Lacsores)',
            height=400,
            xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig_mc, use_container_width=True)
    
    with tab2:
        st.subheader("Profitability Metrics Comparison")
        fig2 = create_profitability_comparison(df, ticker)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Valuation Multiples - Radar View")
        fig3 = create_valuation_multiples_radar(df, ticker)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Target company not found in dataset")
        
        # Valuation multiples table
        st.subheader("Valuation Multiples Summary")
        val_df = df[['ticker', 'pe', 'pb', 'ps', 'ev_ebitda']].copy()
        val_df.columns = ['Ticker', 'P/E', 'P/B', 'P/S', 'EV/EBITDA']
        st.dataframe(
            val_df.style.format({
                'P/E': '{:.2f}x',
                'P/B': '{:.2f}x',
                'P/S': '{:.2f}x',
                'EV/EBITDA': '{:.2f}x'
            }).background_gradient(subset=['P/E', 'P/B', 'P/S', 'EV/EBITDA'], cmap='RdYlGn_r'),
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Capital Structure Comparison")
        fig4 = create_balance_sheet_comparison(df, ticker)
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)
        
        # Debt to Equity chart
        st.subheader("Leverage Ratio (Debt/Equity)")
        df_sorted = df.sort_values('debt_to_equity', ascending=True)
        fig_de = go.Figure(go.Bar(
            x=df_sorted['ticker'],
            y=df_sorted['debt_to_equity'],
            marker_color=['red' if t == ticker else '#F77F00' for t in df_sorted['ticker']],
            text=[f"{val:.2f}x" for val in df_sorted['debt_to_equity']],
            textposition='outside'
        ))
        fig_de.update_layout(
            title='Debt-to-Equity Ratio',
            xaxis_title='Company',
            yaxis_title='D/E Ratio',
            height=400,
            xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig_de, use_container_width=True)
    
    with tab5:
        st.subheader("🌐 3D Interactive Analysis")
        
        st.markdown("### Chart 1: Revenue vs Operating Margin vs P/E")
        st.caption("Bubble size = Market Cap | Explore by rotating the chart")
        fig5 = create_3d_scatter_revenue_margin_valuation(df, ticker)
        if fig5:
            st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Chart 2: Market Cap vs Revenue vs Net Margin")
        st.caption("Color = Debt/Equity (Red = High Leverage)")
        fig6 = create_3d_bubble_market_cap_revenue_margin(df, ticker)
        if fig6:
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab6:
        st.subheader("Complete Peer Data Table")
        
        # Reorder columns for better readability
        display_cols = ['ticker', 'company', 'revenue', 'operating_margin', 'net_margin', 
                       'market_cap', 'pe', 'pb', 'roe', 'debt_to_equity']
        display_df = df[display_cols].copy()
        display_df.columns = ['Ticker', 'Company', 'Revenue (Lacs)', 'Op. Margin %', 'Net Margin %',
                              'Market Cap (Lacs)', 'P/E', 'P/B', 'ROE %', 'D/E']
        
        st.dataframe(
            display_df.style.format({
                'Revenue (Lacs)': '{:.0f}',
                'Op. Margin %': '{:.1f}%',
                'Net Margin %': '{:.1f}%',
                'Market Cap (Lacs)': '{:.0f}',
                'P/E': '{:.2f}x',
                'P/B': '{:.2f}x',
                'ROE %': '{:.1f}%',
                'D/E': '{:.2f}x'
            }).background_gradient(subset=['Op. Margin %', 'Net Margin %', 'ROE %'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Peer Data as CSV",
            data=csv,
            file_name=f"{ticker}_peer_comparison.csv",
            mime="text/csv"
        )


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    st.set_page_config(page_title="Peer Comparison Dashboard", layout="wide")
    
    st.title("🏢 Advanced Peer Comparison Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Target Company Ticker:", value="TATASTEEL")
    
    with col2:
        peers = st.text_input(
            "Peer Tickers (comma-separated):",
            value="JSWSTEEL,SAIL,JINDALSTEL,NMDC"
        )
    
    if st.button("🚀 Generate Dashboard", type="primary"):
        create_peer_comparison_dashboard(ticker, peers)
