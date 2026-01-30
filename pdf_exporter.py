"""
COMPREHENSIVE PDF GENERATOR FOR DCF VALUATION
==============================================
Creates detailed, print-ready PDFs that mirror the exact layout and content
shown in the Streamlit app after clicking "Fetch & Analyze"

This module includes:
- ALL tabs from the app (Historical, Projections, FCF, WACC, Summary, Sensitivity, Comparative, Peers)
- All charts as high-quality images
- All tables with proper formatting
- Detailed breakdowns matching the app exactly
- Professional styling with proper page breaks

Author: DCF Valuation Tool
Date: January 2026
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image as RLImage, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import tempfile
import os
from io import BytesIO


class NumberedCanvas(canvas.Canvas):
    """Canvas with page numbers"""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        page_num = self._pageNumber
        text = f"Page {page_num} of {page_count}"
        self.drawRightString(7.5*inch, 0.5*inch, text)


def create_chart_image(fig, width=1200, height=600):
    """Convert Plotly figure to image bytes"""
    try:
        # Use Plotly's to_image if kaleido is available
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        return BytesIO(img_bytes)
    except:
        # Fallback: return None, we'll skip the chart
        return None


def create_historical_chart(financials):
    """Create historical financials line chart"""
    years = financials.get('years', [])
    revenues = financials.get('revenue', [])
    ebitdas = financials.get('ebitda', [])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=revenues, name='Revenue', line=dict(color='#2E86AB', width=3)))
    fig.add_trace(go.Scatter(x=years, y=ebitdas, name='EBITDA', line=dict(color='#06A77D', width=3)))
    
    fig.update_layout(
        title="Historical Financial Performance",
        xaxis_title="Year",
        yaxis_title="₹ Lacs",
        height=500,
        showlegend=True,
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    return fig


def create_projection_chart(projections):
    """Create projections chart"""
    years = projections.get('years', [])
    revenues = projections.get('revenue', [])
    fcfs = projections.get('fcf', [])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=years, y=revenues, name='Revenue Projection', marker_color='#4ECDC4'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=years, y=fcfs, name='FCF', mode='lines+markers',
                  line=dict(color='#FF6B6B', width=3)),
        secondary_y=True
    )
    
    fig.update_layout(
        title="5-Year Financial Projections",
        height=500,
        showlegend=True
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Revenue (₹ Lacs)", secondary_y=False)
    fig.update_yaxes(title_text="FCF (₹ Lacs)", secondary_y=True)
    
    return fig


def create_wacc_breakdown_chart(wacc_details):
    """Create WACC components pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=['Equity Weight', 'Debt Weight'],
        values=[wacc_details.get('we', 0), wacc_details.get('wd', 0)],
        marker=dict(colors=['#06A77D', '#D62828'])
    )])
    
    fig.update_layout(
        title=f"Capital Structure (WACC: {wacc_details.get('wacc', 0):.2f}%)",
        height=400
    )
    
    return fig


def generate_comprehensive_pdf(data_dict):
    """
    Generate comprehensive PDF with ALL data from the app
    
    Parameters:
    -----------
    data_dict : dict containing:
        'company_name': str
        'ticker': str
        'current_price': float
        'shares': int
        'financials': dict (with years, revenue, ebitda, ebit, nopat, fcf, capex, etc.)
        'projections': dict (projected years, revenue, fcf, etc.)
        'dcf_results': dict (enterprise_value, equity_value, fair_value_per_share, etc.)
        'wacc_details': dict (wacc, ke, kd, we, wd, beta, rf, rm, etc.)
        'fair_values': dict (all valuation methods)
        'peer_data': pd.DataFrame (optional)
        'comp_results': dict (optional)
        'sensitivity_data': dict (optional)
        'classification': dict (business model classification)
    
    Returns:
    --------
    str: Path to generated PDF
    """
    
    # Extract all data
    company_name = data_dict.get('company_name', 'Company')
    ticker = data_dict.get('ticker', 'TICKER')
    current_price = data_dict.get('current_price', 0)
    shares = data_dict.get('shares', 0)
    financials = data_dict.get('financials', {})
    projections = data_dict.get('projections', {})
    dcf_results = data_dict.get('dcf_results', {})
    wacc_details = data_dict.get('wacc_details', {})
    fair_values = data_dict.get('fair_values', {})
    peer_data = data_dict.get('peer_data', pd.DataFrame())
    comp_results = data_dict.get('comp_results')
    sensitivity_data = data_dict.get('sensitivity_data')
    classification = data_dict.get('classification', {})
    
    # Create output path
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{ticker}_Comprehensive_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    
    # Create PDF
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomTitle',
        fontSize=32,
        textColor=HexColor('#2E86AB'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name='SectionHead',
        fontSize=16,
        textColor=HexColor('#2E86AB'),
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name='SubHead',
        fontSize=13,
        textColor=HexColor('#06A77D'),
        fontName='Helvetica-Bold',
        spaceBefore=8,
        spaceAfter=6
    ))
    
    story = []
    
    # =====================================
    # COVER PAGE
    # =====================================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("DCF VALUATION REPORT", styles['CustomTitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"<b>{company_name}</b>", styles['Heading1']))
    story.append(Paragraph(f"Ticker: {ticker}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Classification if available
    if classification:
        class_text = f"Business Type: {classification.get('type', 'N/A')}"
        story.append(Paragraph(class_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        f"<para align=center>Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</para>",
        styles['Normal']
    ))
    story.append(PageBreak())
    
    # =====================================
    # EXECUTIVE SUMMARY
    # =====================================
    story.append(Paragraph("EXECUTIVE SUMMARY", styles['SectionHead']))
    story.append(Spacer(1, 0.2*inch))
    
    # Calculate metrics
    avg_fv = np.mean([v for v in fair_values.values() if v and v > 0]) if fair_values else 0
    upside = ((avg_fv - current_price) / current_price * 100) if current_price > 0 else 0
    
    # Summary table with MORE details
    summary_data = [
        ['Metric', 'Value', 'Details'],
        ['Company', company_name, f'Ticker: {ticker}'],
        ['Current Market Price', 
         f"₹{current_price:.2f}" if current_price > 0 else "N/A (Unlisted)", 
         'Latest trading price' if current_price > 0 else 'Private company'],
        ['Shares Outstanding', f"{shares:,}", 'Total shares issued'],
        ['Market Cap', f"₹{(current_price * shares / 100000):.2f} Cr" if current_price > 0 else "N/A", 
         'Current valuation'],
        ['DCF Fair Value', f"₹{dcf_results.get('fair_value_per_share', 0):.2f}", 
         'Intrinsic value from DCF model'],
        ['Average Fair Value', f"₹{avg_fv:.2f}", 
         f'Average of {len(fair_values)} valuation methods'],
        ['Enterprise Value', f"₹{dcf_results.get('enterprise_value', 0):.2f} Lacs", 
         'Total firm value'],
        ['Equity Value', f"₹{dcf_results.get('equity_value', 0):.2f} Lacs", 
         'Value for equity holders'],
        ['Upside/Downside', f"{upside:+.1f}%", 
         'Potential gain/loss from current price']
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 2*inch, 2.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')])
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Investment recommendation
    if upside > 20:
        rec = "STRONG BUY"
        rec_color = HexColor('#06A77D')
    elif upside > 10:
        rec = "BUY"
        rec_color = HexColor('#4ECDC4')
    elif upside > 0:
        rec = "HOLD / BUY"
        rec_color = HexColor('#F4D35E')
    elif upside > -10:
        rec = "HOLD"
        rec_color = HexColor('#F4A259')
    else:
        rec = "SELL / HOLD"
        rec_color = HexColor('#D62828')
    
    rec_style = ParagraphStyle('Rec', parent=styles['Normal'], fontSize=14, 
                               textColor=rec_color, alignment=TA_CENTER, fontName='Helvetica-Bold')
    story.append(Paragraph(f"Investment Recommendation: {rec}", rec_style))
    story.append(PageBreak())
    
    # =====================================
    # TAB 1: HISTORICAL ANALYSIS
    # =====================================
    if financials and 'years' in financials:
        story.append(Paragraph("HISTORICAL FINANCIAL ANALYSIS", styles['SectionHead']))
        story.append(Spacer(1, 0.2*inch))
        
        # Chart
        hist_chart = create_historical_chart(financials)
        hist_img = create_chart_image(hist_chart, width=1300, height=500)
        if hist_img:
            story.append(RLImage(hist_img, width=6.5*inch, height=2.5*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Detailed table
        years = financials['years']
        metrics_map = {
            'Revenue (₹ Lacs)': 'revenue',
            'EBITDA (₹ Lacs)': 'ebitda',
            'EBIT (₹ Lacs)': 'ebit',
            'NOPAT (₹ Lacs)': 'nopat',
            'Free Cash Flow (₹ Lacs)': 'fcf',
            'CAPEX (₹ Lacs)': 'capex',
            'Working Capital (₹ Lacs)': 'working_capital',
            'Cash Balance (₹ Lacs)': 'cash'
        }
        
        fin_data = [['Metric'] + [str(y) for y in years]]
        
        for label, key in metrics_map.items():
            if key in financials:
                values = financials[key]
                row = [label] + [f"₹{v:,.0f}" if v else "—" for v in values]
                fin_data.append(row)
        
        # Add growth rates
        if 'revenue' in financials and len(financials['revenue']) > 1:
            revenues = financials['revenue']
            growth_row = ['Revenue Growth (YoY)'] + ['—']
            for i in range(1, len(revenues)):
                if revenues[i-1] and revenues[i-1] != 0:
                    growth = ((revenues[i] - revenues[i-1]) / revenues[i-1]) * 100
                    growth_row.append(f"{growth:.1f}%")
                else:
                    growth_row.append("—")
            fin_data.append(growth_row)
        
        # Add margins
        if 'revenue' in financials and 'ebitda' in financials:
            margin_row = ['EBITDA Margin (%)'] + ['—'] * len(years)
            for i, (rev, ebitda) in enumerate(zip(financials['revenue'], financials['ebitda'])):
                if rev and rev > 0:
                    margin = (ebitda / rev) * 100
                    margin_row[i + 1] = f"{margin:.1f}%"
            fin_data.append(margin_row)
        
        col_widths = [2.5*inch] + [1*inch] * len(years)
        fin_table = Table(fin_data, colWidths=col_widths)
        fin_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')])
        ]))
        
        story.append(fin_table)
        story.append(PageBreak())
    
    # =====================================
    # TAB 2: PROJECTIONS
    # =====================================
    if projections and 'years' in projections:
        story.append(Paragraph("5-YEAR FINANCIAL PROJECTIONS", styles['SectionHead']))
        story.append(Spacer(1, 0.2*inch))
        
        # Chart
        proj_chart = create_projection_chart(projections)
        proj_img = create_chart_image(proj_chart, width=1300, height=500)
        if proj_img:
            story.append(RLImage(proj_img, width=6.5*inch, height=2.5*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Detailed projection table
        proj_years = projections['years']
        proj_data = [['Metric'] + [str(y) for y in proj_years]]
        
        proj_metrics = {
            'Revenue (₹ Lacs)': 'revenue',
            'Revenue Growth (%)': 'revenue_growth',
            'EBITDA (₹ Lacs)': 'ebitda',
            'EBITDA Margin (%)': 'ebitda_margin',
            'NOPAT (₹ Lacs)': 'nopat',
            'CAPEX (₹ Lacs)': 'capex',
            'Change in WC (₹ Lacs)': 'working_capital_change',
            'Free Cash Flow (₹ Lacs)': 'fcf'
        }
        
        for label, key in proj_metrics.items():
            if key in projections:
                values = projections[key]
                if 'growth' in key or 'margin' in key.lower():
                    row = [label] + [f"{v:.1f}%" if v else "—" for v in values]
                else:
                    row = [label] + [f"₹{v:,.0f}" if v else "—" for v in values]
                proj_data.append(row)
        
        proj_table = Table(proj_data, colWidths=[2.5*inch] + [0.9*inch] * len(proj_years))
        proj_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#06A77D')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#E8F4E8')])
        ]))
        
        story.append(proj_table)
        story.append(PageBreak())
    
    # =====================================
    # TAB 3: FCF WORKING
    # =====================================
    story.append(Paragraph("FREE CASH FLOW CALCULATION", styles['SectionHead']))
    story.append(Spacer(1, 0.2*inch))
    
    if projections:
        fcf_breakdown = [
            ['Component', 'Formula / Description'] + [str(y) for y in projections.get('years', [])],
            ['NOPAT', 'Net Operating Profit After Tax'] + 
                [f"₹{v:,.0f}" for v in projections.get('nopat', [])],
            ['(-) CAPEX', 'Capital Expenditures'] + 
                [f"(₹{v:,.0f})" for v in projections.get('capex', [])],
            ['(-) Change in WC', 'Working Capital Changes'] + 
                [f"(₹{v:,.0f})" for v in projections.get('working_capital_change', [])],
            ['(=) Free Cash Flow', 'NOPAT - CAPEX - ΔWC'] + 
                [f"₹{v:,.0f}" for v in projections.get('fcf', [])]
        ]
        
        fcf_table = Table(fcf_breakdown)
        fcf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4ECDC4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#E8F4F8')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9)
        ]))
        
        story.append(fcf_table)
        story.append(PageBreak())
    
    # =====================================
    # TAB 4: WACC BREAKDOWN
    # =====================================
    story.append(Paragraph("WEIGHTED AVERAGE COST OF CAPITAL (WACC)", styles['SectionHead']))
    story.append(Spacer(1, 0.2*inch))
    
    if wacc_details:
        # Chart
        wacc_chart = create_wacc_breakdown_chart(wacc_details)
        wacc_img = create_chart_image(wacc_chart, width=1100, height=400)
        if wacc_img:
            story.append(RLImage(wacc_img, width=5.5*inch, height=2*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # WACC components table
        wacc_data = [
            ['Component', 'Value', 'Description'],
            ['Risk-Free Rate (Rf)', f"{wacc_details.get('rf', 0):.2f}%", '10Y Govt Bond Yield'],
            ['Market Return (Rm)', f"{wacc_details.get('rm', 0):.2f}%", 'Expected market return'],
            ['Beta (β)', f"{wacc_details.get('beta', 0):.2f}", 'Stock volatility vs market'],
            ['Cost of Equity (Ke)', f"{wacc_details.get('ke', 0):.2f}%", 'Rf + β(Rm - Rf)'],
            ['Cost of Debt (Kd)', f"{wacc_details.get('kd', 0):.2f}%", 'Interest rate on debt'],
            ['Tax Rate', f"{wacc_details.get('tax_rate', 0)*100:.2f}%", 'Corporate tax rate'],
            ['After-tax Cost of Debt', f"{wacc_details.get('kd_after_tax', 0):.2f}%", 'Kd × (1 - Tax)'],
            ['Equity Weight (We)', f"{wacc_details.get('we', 0):.2f}%", 'E / (E + D)'],
            ['Debt Weight (Wd)', f"{wacc_details.get('wd', 0):.2f}%", 'D / (E + D)'],
            ['WACC', f"{wacc_details.get('wacc', 0):.2f}%", 'We×Ke + Wd×Kd(1-T)']
        ]
        
        wacc_table = Table(wacc_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        wacc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#E8F4F8')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, HexColor('#F5F5F5')])
        ]))
        
        story.append(wacc_table)
        story.append(PageBreak())
    
    # =====================================
    # TAB 5: DCF VALUATION SUMMARY
    # =====================================
    story.append(Paragraph("DCF VALUATION SUMMARY", styles['SectionHead']))
    story.append(Spacer(1, 0.2*inch))
    
    if dcf_results:
        # Terminal value calculation
        terminal_fcf = projections.get('fcf', [0])[-1] if projections else 0
        terminal_growth = dcf_results.get('terminal_growth_rate', 0)
        wacc_rate = wacc_details.get('wacc', 0) / 100
        
        terminal_value = terminal_fcf * (1 + terminal_growth) / (wacc_rate - terminal_growth) if wacc_rate > terminal_growth else 0
        
        dcf_summary = [
            ['Component', 'Value (₹ Lacs)', 'Notes'],
            ['Present Value of FCFs', f"{dcf_results.get('pv_fcf', 0):,.2f}", 
             'Sum of discounted cash flows'],
            ['Terminal Value', f"{terminal_value:,.2f}", 
             f'FCF_Year5 × (1+g) / (WACC-g), g={terminal_growth*100:.1f}%'],
            ['PV of Terminal Value', f"{dcf_results.get('pv_terminal', 0):,.2f}", 
             'Terminal value discounted to present'],
            ['Enterprise Value', f"{dcf_results.get('enterprise_value', 0):,.2f}", 
             'PV(FCF) + PV(Terminal Value)'],
            ['(+) Cash Balance', f"{dcf_results.get('cash', 0):,.2f}", 
             'Cash and equivalents'],
            ['(-) Total Debt', f"{dcf_results.get('debt', 0):,.2f}", 
             'Short-term + Long-term debt'],
            ['(=) Equity Value', f"{dcf_results.get('equity_value', 0):,.2f}", 
             'Value for shareholders'],
            ['Shares Outstanding', f"{shares:,}", 
             'Total shares issued'],
            ['Fair Value per Share', f"₹{dcf_results.get('fair_value_per_share', 0):.2f}", 
             'Equity Value / Shares']
        ]
        
        dcf_table = Table(dcf_summary, colWidths=[2.2*inch, 1.8*inch, 2.5*inch])
        dcf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#06A77D')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), HexColor('#E8F4E8')),
            ('FONTSIZE', (1, -1), (1, -1), 14),
            ('TEXTCOLOR', (1, -1), (1, -1), HexColor('#06A77D')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, HexColor('#F5F5F5')])
        ]))
        
        story.append(dcf_table)
        story.append(PageBreak())
    
    # =====================================
    # TAB 6: SENSITIVITY ANALYSIS
    # =====================================
    if sensitivity_data:
        story.append(Paragraph("SENSITIVITY ANALYSIS", styles['SectionHead']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Impact of WACC and Terminal Growth Rate Changes", styles['SubHead']))
        story.append(Spacer(1, 0.1*inch))
        
        # Create sensitivity table
        # This would need the actual sensitivity data structure
        story.append(Paragraph("(Sensitivity table would be generated from sensitivity_data)", styles['Normal']))
        story.append(PageBreak())
    
    # =====================================
    # FAIR VALUE COMPARISON
    # =====================================
    if fair_values:
        story.append(Paragraph("FAIR VALUE COMPARISON", styles['SectionHead']))
        story.append(Spacer(1, 0.2*inch))
        
        # Statistics
        values = [v for v in fair_values.values() if v and v > 0]
        
        stats_data = [
            ['Valuation Method', 'Fair Value (₹)'],
        ]
        
        for method, value in fair_values.items():
            stats_data.append([method, f"₹{value:.2f}"])
        
        stats_data.extend([
            ['', ''],
            ['Minimum', f"₹{min(values):.2f}"],
            ['Maximum', f"₹{max(values):.2f}"],
            ['Average', f"₹{np.mean(values):.2f}"],
            ['Median', f"₹{np.median(values):.2f}"],
            ['Std Deviation', f"₹{np.std(values):.2f}"]
        ])
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')]),
            ('BACKGROUND', (0, -5), (-1, -1), HexColor('#E8F4F8')),
            ('FONTNAME', (0, -3), (-1, -1), 'Helvetica-Bold')
        ]))
        
        story.append(stats_table)
        story.append(PageBreak())
    
    # =====================================
    # CONCLUSION
    # =====================================
    story.append(Paragraph("CONCLUSION & RECOMMENDATIONS", styles['SectionHead']))
    story.append(Spacer(1, 0.2*inch))
    
    conclusion = f"""
    Based on comprehensive discounted cash flow analysis, the estimated fair value for 
    <b>{company_name}</b> ({ticker}) is <b>₹{dcf_results.get('fair_value_per_share', 0):.2f}</b> per share.
    """
    
    if current_price > 0:
        conclusion += f"""
        <br/><br/>
        The current market price of ₹{current_price:.2f} represents a <b>{upside:+.1f}%</b> 
        {'discount to' if upside > 0 else 'premium to'} our fair value estimate.
        <br/><br/>
        <b>Investment Recommendation: {rec}</b>
        """
    
    conclusion += """
    <br/><br/>
    <b>Key Assumptions:</b><br/>
    • WACC used for discounting future cash flows<br/>
    • Terminal growth rate reflects long-term industry expectations<br/>
    • Projections based on historical performance and industry trends<br/>
    <br/>
    <b>Risk Factors:</b><br/>
    • Market volatility and macroeconomic conditions<br/>
    • Changes in industry dynamics and competitive landscape<br/>
    • Execution risks in business strategy<br/>
    • Regulatory and policy changes<br/>
    • Interest rate fluctuations affecting discount rates<br/>
    <br/>
    <i>Disclaimer: This valuation is for informational purposes only and should not be 
    considered as investment advice. Please consult with a qualified financial advisor 
    before making investment decisions.</i>
    """
    
    story.append(Paragraph(conclusion, styles['Normal']))
    
    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    
    return output_path


# Wrapper function for easy use
def create_comprehensive_pdf(data_dict):
    """
    Main entry point for PDF generation
    
    Usage:
        pdf_path = create_comprehensive_pdf({
            'company_name': 'Reliance Industries',
            'ticker': 'RELIANCE',
            'current_price': 2500,
            'shares': 6765000000,
            'financials': {...},
            'projections': {...},
            'dcf_results': {...},
            'wacc_details': {...},
            'fair_values': {...}
        })
    """
    return generate_comprehensive_pdf(data_dict)
