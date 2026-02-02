"""
Enhanced Peer Metrics Visualizations & Fair Value Comparison
=============================================================
Elegant, modern visualizations for peer analysis and valuation results
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st


def create_peer_metrics_elegant_display(df, ticker):
    """
    Create an elegant dashboard-style display for peer metrics
    Uses modern card-style layouts and clean visualizations
    """
    st.markdown("### 📊 Peer Metrics Dashboard")
    
    # Get target company data
    target = df[df['ticker'] == ticker].iloc[0] if ticker in df['ticker'].values else None
    peers = df[df['ticker'] != ticker]
    
    if target is None:
        st.error("Target company not found in peer data")
        return
    
    # Calculate percentile ranks
    metrics_for_ranking = ['revenue', 'operating_margin', 'net_margin', 'roe', 'market_cap']
    percentiles = {}
    for metric in metrics_for_ranking:
        if metric in df.columns:
            target_value = target[metric]
            percentile = (df[metric] < target_value).sum() / len(df) * 100
            percentiles[metric] = percentile
    
    # Create modern metric cards
    st.markdown("#### 🎯 Target Company Position")
    
    # Row 1: Size metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Revenue",
            f"₹{target['revenue']:.0f} L",
            percentiles.get('revenue', 0),
            peers['revenue'].median(),
            "Lacs"
        )
    
    with col2:
        create_metric_card(
            "Market Cap",
            f"₹{target['market_cap']:.0f} L",
            percentiles.get('market_cap', 0),
            peers['market_cap'].median(),
            "Lacs"
        )
    
    with col3:
        create_metric_card(
            "Operating Margin",
            f"{target['operating_margin']:.1f}%",
            percentiles.get('operating_margin', 0),
            peers['operating_margin'].median(),
            "%"
        )
    
    with col4:
        create_metric_card(
            "ROE",
            f"{target['roe']:.1f}%",
            percentiles.get('roe', 0),
            peers['roe'].median(),
            "%"
        )
    
    st.markdown("---")
    
    # Create comprehensive comparison visualizations
    create_peer_comparison_heatmap(df, ticker)
    
    st.markdown("---")
    
    create_valuation_multiples_comparison(df, ticker)


def create_metric_card(title, value, percentile, peer_median, unit):
    """
    Create an elegant metric card with percentile indicator
    """
    # Determine color based on percentile
    if percentile >= 75:
        color = "#06A77D"  # Green
        emoji = "🟢"
        position = "Top Quartile"
    elif percentile >= 50:
        color = "#F4D35E"  # Yellow
        emoji = "🟡"
        position = "Above Median"
    elif percentile >= 25:
        color = "#F77F00"  # Orange
        emoji = "🟠"
        position = "Below Median"
    else:
        color = "#D62828"  # Red
        emoji = "🔴"
        position = "Bottom Quartile"
    
    # Create styled card using HTML/CSS
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
        border-left: 4px solid {color};
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
    ">
        <div style="color: #666; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">
            {emoji} {title}
        </div>
        <div style="color: #1a1a1a; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;">
            {value}
        </div>
        <div style="color: {color}; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem;">
            {position}
        </div>
        <div style="color: #999; font-size: 0.75rem;">
            Peer Median: {peer_median:.1f} {unit}
        </div>
        <div style="margin-top: 0.5rem;">
            <div style="background: #e0e0e0; height: 4px; border-radius: 2px; overflow: hidden;">
                <div style="background: {color}; height: 100%; width: {percentile}%; transition: width 0.3s;"></div>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def create_peer_comparison_heatmap(df, ticker):
    """
    Create an elegant heatmap comparing all metrics across peers
    """
    st.markdown("#### 🔥 Comprehensive Peer Metrics Heatmap")
    
    # Select metrics for heatmap
    metrics = ['operating_margin', 'net_margin', 'roe', 'roa', 'pe', 'pb', 'debt_to_equity']
    metric_labels = ['Op. Margin %', 'Net Margin %', 'ROE %', 'ROA %', 'P/E', 'P/B', 'D/E']
    
    # Prepare data
    heatmap_data = df[['ticker'] + metrics].copy()
    heatmap_data = heatmap_data.set_index('ticker')
    
    # Normalize data for better visualization (percentile rank)
    normalized_data = heatmap_data.rank(pct=True) * 100
    
    # Create custom colorscale
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data.values.T,
        x=normalized_data.index,
        y=metric_labels,
        colorscale=[
            [0, '#D62828'],      # Red - Bottom
            [0.25, '#F77F00'],   # Orange
            [0.5, '#F4D35E'],    # Yellow
            [0.75, '#90BE6D'],   # Light Green
            [1, '#06A77D']       # Green - Top
        ],
        text=heatmap_data.values.T,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        colorbar=dict(
            title="Percentile<br>Rank",
            titleside="right",
            tickmode="array",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            len=0.7
        ),
        hovertemplate='<b>%{x}</b><br>%{y}: %{text:.2f}<br>Percentile: %{z:.0f}%<extra></extra>'
    ))
    
    # Highlight target company
    target_idx = list(normalized_data.index).index(ticker) if ticker in normalized_data.index else -1
    if target_idx >= 0:
        fig.add_shape(
            type="rect",
            x0=target_idx - 0.5,
            x1=target_idx + 0.5,
            y0=-0.5,
            y1=len(metric_labels) - 0.5,
            line=dict(color="red", width=3),
            layer="above"
        )
    
    fig.update_layout(
        title="Peer Comparison: Percentile Rankings Across Key Metrics",
        xaxis_title="Company",
        yaxis_title="Metric",
        height=450,
        font=dict(size=11),
        xaxis=dict(tickangle=-45)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_valuation_multiples_comparison(df, ticker):
    """
    Create elegant valuation multiples comparison with spider chart and bars
    """
    st.markdown("#### 💎 Valuation Multiples Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Spider/Radar chart for target vs peer average
        target_row = df[df['ticker'] == ticker].iloc[0] if ticker in df['ticker'].values else None
        peers = df[df['ticker'] != ticker]
        
        if target_row is not None and not peers.empty:
            categories = ['P/E', 'P/B', 'P/S', 'EV/EBITDA']
            
            # Normalize to percentiles for comparison
            target_values = []
            peer_median_values = []
            
            for metric in ['pe', 'pb', 'ps', 'ev_ebitda']:
                target_val = target_row[metric]
                peer_median = peers[metric].median()
                
                # For valuation multiples, lower is often better
                # But we'll show actual values
                target_values.append(target_val)
                peer_median_values.append(peer_median)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=target_values,
                theta=categories,
                fill='toself',
                name='Target Company',
                line=dict(color='#D62828', width=2),
                fillcolor='rgba(214, 40, 40, 0.2)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=peer_median_values,
                theta=categories,
                fill='toself',
                name='Peer Median',
                line=dict(color='#2E86AB', width=2),
                fillcolor='rgba(46, 134, 171, 0.2)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(max(target_values), max(peer_median_values)) * 1.2]
                    )
                ),
                showlegend=True,
                title="Valuation Multiples: Target vs Peer Median",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart showing all companies
        multiples_df = df[['ticker', 'pe', 'pb']].copy()
        multiples_df = multiples_df[multiples_df['pe'] > 0]  # Filter invalid values
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("P/E Ratio Comparison", "P/B Ratio Comparison"),
            vertical_spacing=0.15
        )
        
        # P/E comparison
        colors_pe = ['#D62828' if t == ticker else '#2E86AB' for t in multiples_df['ticker']]
        fig.add_trace(
            go.Bar(
                x=multiples_df['ticker'],
                y=multiples_df['pe'],
                marker_color=colors_pe,
                text=multiples_df['pe'].round(2),
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>P/E: %{y:.2f}x<extra></extra>'
            ),
            row=1, col=1
        )
        
        # P/B comparison
        colors_pb = ['#D62828' if t == ticker else '#06A77D' for t in multiples_df['ticker']]
        fig.add_trace(
            go.Bar(
                x=multiples_df['ticker'],
                y=multiples_df['pb'],
                marker_color=colors_pb,
                text=multiples_df['pb'].round(2),
                textposition='outside',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>P/B: %{y:.2f}x<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=2, col=1)
        fig.update_yaxes(title_text="P/E Ratio", row=1, col=1)
        fig.update_yaxes(title_text="P/B Ratio", row=2, col=1)
        
        fig.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)


def create_fair_value_comparison_chart(valuation_results, current_price=None):
    """
    Create an elegant comparison chart for all fair values from different methods
    
    Args:
        valuation_results: dict with structure:
            {
                'DCF': fair_value,
                'P/E Method': fair_value,
                'P/B Method': fair_value,
                'P/S Method': fair_value,
                'DDM': fair_value,
                'Residual Income': fair_value,
                etc.
            }
        current_price: Current market price (optional)
    """
    st.markdown("### 🎯 Fair Value Comparison Across Methods")
    
    # Prepare data
    methods = []
    fair_values = []
    colors = []
    
    # Color scheme for different methods
    color_map = {
        'DCF': '#06A77D',
        'P/E': '#2E86AB',
        'P/B': '#4ECDC4',
        'P/S': '#FF6B6B',
        'EV/EBITDA': '#95E1D3',
        'DDM': '#F38181',
        'Residual Income': '#AA96DA',
        'Average': '#FCBAD3'
    }
    
    for method, value in valuation_results.items():
        if value and value > 0:
            methods.append(method)
            fair_values.append(value)
            # Assign color based on method name
            color = color_map.get(method, '#A8DADC')
            colors.append(color)
    
    if not methods:
        st.warning("No valuation results available")
        return
    
    # Calculate average
    avg_fair_value = np.mean(fair_values)
    
    # Create subplots: main bar chart and radial gauge
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "bar", "colspan": 2}, None],
            [{"type": "indicator"}, {"type": "indicator"}]
        ],
        subplot_titles=[
            "Fair Value Estimates by Valuation Method",
            "Average Fair Value",
            "vs Current Price"
        ],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15
    )
    
    # Main bar chart
    fig.add_trace(
        go.Bar(
            x=methods,
            y=fair_values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"₹{v:.2f}" for v in fair_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Fair Value: ₹%{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add average line
    fig.add_hline(
        y=avg_fair_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: ₹{avg_fair_value:.2f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Add current price line if provided
    if current_price and current_price > 0:
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Current: ₹{current_price:.2f}",
            annotation_position="left",
            row=1, col=1
        )
    
    # Average fair value indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=avg_fair_value,
            title={"text": "Average Fair Value"},
            number={'prefix': "₹", 'valueformat': '.2f', 'font': {'size': 40}},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=2, col=1
    )
    
    # Current price vs fair value indicator
    if current_price and current_price > 0:
        upside = ((avg_fair_value - current_price) / current_price) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=current_price,
                delta={
                    'reference': avg_fair_value,
                    'valueformat': '.2f',
                    'relative': False,
                    'prefix': '₹',
                    'font': {'size': 20}
                },
                title={"text": f"Current Price<br><sub>Upside: {upside:+.1f}%</sub>"},
                number={'prefix': "₹", 'valueformat': '.2f', 'font': {'size': 40}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=2
        )
    
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text="Fair Value (₹)", row=1, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### 📊 Valuation Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Minimum", f"₹{min(fair_values):.2f}")
    
    with col2:
        st.metric("Maximum", f"₹{max(fair_values):.2f}")
    
    with col3:
        st.metric("Average", f"₹{avg_fair_value:.2f}")
    
    with col4:
        st.metric("Median", f"₹{np.median(fair_values):.2f}")
    
    with col5:
        std_dev = np.std(fair_values)
        st.metric("Std Dev", f"₹{std_dev:.2f}")
    
    # Confidence level display
    if current_price and current_price > 0:
        st.markdown("---")
        st.markdown("#### 🎲 Investment Recommendation")
        
        methods_above = sum(1 for v in fair_values if v > current_price)
        confidence = (methods_above / len(fair_values)) * 100
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if confidence >= 75:
                st.success(f"🟢 **STRONG BUY** - {methods_above}/{len(fair_values)} methods suggest undervaluation")
                recommendation = "The stock appears significantly undervalued by multiple methods."
            elif confidence >= 50:
                st.info(f"🟡 **BUY** - {methods_above}/{len(fair_values)} methods suggest undervaluation")
                recommendation = "The stock appears moderately undervalued."
            elif confidence >= 25:
                st.warning(f"🟠 **HOLD** - {methods_above}/{len(fair_values)} methods suggest undervaluation")
                recommendation = "Fair value estimates are mixed. Consider other factors."
            else:
                st.error(f"🔴 **SELL/AVOID** - Only {methods_above}/{len(fair_values)} methods suggest undervaluation")
                recommendation = "The stock appears overvalued by most methods."
            
            st.caption(recommendation)
        
        with col2:
            # Create a simple gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#06A77D" if confidence >= 50 else "#D62828"},
                    'steps': [
                        {'range': [0, 25], 'color': "#FFE5E5"},
                        {'range': [25, 50], 'color': "#FFF4E5"},
                        {'range': [50, 75], 'color': "#E5F9F5"},
                        {'range': [75, 100], 'color': "#D4F4DD"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            gauge_fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge_fig, use_container_width=True)


def create_valuation_waterfall_chart(valuation_breakdown):
    """
    Create a waterfall chart showing how valuation builds up
    Useful for DCF showing components
    
    Args:
        valuation_breakdown: dict with steps
            {
                'Enterprise Value': value,
                'Less: Debt': -value,
                'Plus: Cash': value,
                'Equity Value': value,
                'Shares Outstanding': -shares (as divisor),
                'Fair Value per Share': final_value
            }
    """
    st.markdown("### 💧 Valuation Waterfall: How We Got There")
    
    measures = []
    x_labels = []
    y_values = []
    text_values = []
    
    cumulative = 0
    
    for label, value in valuation_breakdown.items():
        x_labels.append(label)
        
        if 'Less' in label or 'Shares' in label:
            measures.append('relative')
            y_values.append(value)
            text_values.append(f"₹{value:.2f}")
            cumulative += value
        elif label == 'Fair Value per Share' or label == 'Equity Value':
            measures.append('total')
            y_values.append(cumulative)
            text_values.append(f"₹{cumulative:.2f}")
        else:
            measures.append('relative')
            y_values.append(value)
            text_values.append(f"₹{value:.2f}")
            cumulative += value
    
    fig = go.Figure(go.Waterfall(
        x=x_labels,
        y=y_values,
        measure=measures,
        text=text_values,
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#06A77D"}},
        decreasing={"marker": {"color": "#D62828"}},
        totals={"marker": {"color": "#2E86AB"}}
    ))
    
    fig.update_layout(
        title="Step-by-Step Valuation Build-up",
        showlegend=False,
        height=500,
        xaxis={'tickangle': -45}
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Integration function to replace lame displays in main app
def display_elegant_fair_values(dcf_value=None, pe_value=None, pb_value=None, 
                                ps_value=None, ddm_value=None, ri_value=None,
                                ev_ebitda_value=None, current_price=None):
    """
    Replace all lame st.metric displays with this elegant visualization
    
    Usage in main app:
        display_elegant_fair_values(
            dcf_value=dcf_result,
            pe_value=pe_result,
            pb_value=pb_result,
            current_price=100
        )
    """
    valuation_results = {}
    
    if dcf_value and dcf_value > 0:
        valuation_results['DCF'] = dcf_value
    if pe_value and pe_value > 0:
        valuation_results['P/E Method'] = pe_value
    if pb_value and pb_value > 0:
        valuation_results['P/B Method'] = pb_value
    if ps_value and ps_value > 0:
        valuation_results['P/S Method'] = ps_value
    if ddm_value and ddm_value > 0:
        valuation_results['DDM'] = ddm_value
    if ri_value and ri_value > 0:
        valuation_results['Residual Income'] = ri_value
    if ev_ebitda_value and ev_ebitda_value > 0:
        valuation_results['EV/EBITDA'] = ev_ebitda_value
    
    create_fair_value_comparison_chart(valuation_results, current_price)
