import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="FX Relative Rotation Graph (RRG) Dashboard")

@st.cache_data
def ma(data, period):
    return data.rolling(window=period).mean()

@st.cache_data
def calculate_rrg_values(data, benchmark):
    sbr = data / benchmark
    rs1 = ma(sbr, 10)
    rs2 = ma(sbr, 26)
    rs = 100 * ((rs1 - rs2) / rs2 + 1)
    rm1 = ma(rs, 1)
    rm2 = ma(rs, 4)
    rm = 100 * ((rm1 - rm2) / rm2 + 1)
    return rs, rm

@st.cache_data
def get_fx_data(timeframe):
    end_date = datetime.now()
    if timeframe == "Weekly":
        start_date = end_date - timedelta(weeks=100)
    elif timeframe == "Hourly":
        start_date = end_date - timedelta(days=7)
    else:  # Daily
        start_date = end_date - timedelta(days=500)

    benchmark = "HKDUSD=X"
    fx_pairs = ["GBPUSD=X", "EURUSD=X", "AUDUSD=X", "NZDUSD=X", "CADUSD=X", "CHFUSD=X", "JPYUSD=X", "CNYUSD=X", 
                "EURGBP=X", "AUDNZD=X", "AUDCAD=X", "NZDCAD=X", "DX-Y.NYB", "AUDJPY=X"]
    fx_names = {
        "GBPUSD=X": "GBP", "EURUSD=X": "EUR", "AUDUSD=X": "AUD", "NZDUSD=X": "NZD",
        "CADUSD=X": "CAD", "JPYUSD=X": "JPY", "CHFUSD=X": "CHF", "CNYUSD=X": "CNY",
        "EURGBP=X": "EURGBP", "AUDNZD=X": "AUDNZD", "AUDCAD=X": "AUDCAD", "NZDCAD=X": "NZDCAD", 
        "DX-Y.NYB": "DXY", "AUDJPY=X": "AUDJPY"
    }

    tickers_to_download = [benchmark] + fx_pairs
    interval = "1h" if timeframe == "Hourly" else "1d"
    data = yf.download(tickers_to_download, start=start_date, end=end_date, interval=interval)['Close']
    
    return data, benchmark, fx_pairs, fx_names

def get_quadrant(x, y):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.iloc[-1]
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.iloc[-1]
    x = float(x)
    y = float(y)
    if x < 100 and y < 100: return "Lagging"
    elif x >= 100 and y < 100: return "Weakening"
    elif x < 100 and y >= 100: return "Improving"
    else: return "Leading"

def create_rrg_chart(data, benchmark, fx_pairs, fx_names, timeframe, tail_length):
    if timeframe != "Weighted Composite":
        rrg_data = pd.DataFrame()
        for pair in fx_pairs:
            rs_ratio, rs_momentum = calculate_rrg_values(data[pair], data[benchmark])
            rrg_data[f"{pair}_RS-Ratio"] = rs_ratio
            rrg_data[f"{pair}_RS-Momentum"] = rs_momentum
    else:
        rrg_data = data

    fig = go.Figure()

    quadrant_colors = {"Lagging": "pink", "Weakening": "lightyellow", "Improving": "lightblue", "Leading": "lightgreen"}
    curve_colors = {"Lagging": "red", "Weakening": "orange", "Improving": "darkblue", "Leading": "darkgreen"}

    for pair in fx_pairs:
        x_values = rrg_data[f"{pair}_RS-Ratio"].dropna().iloc[-tail_length:]
        y_values = rrg_data[f"{pair}_RS-Momentum"].dropna().iloc[-tail_length:]
        
        if not x_values.empty and not y_values.empty:
            current_quadrant = get_quadrant(x_values.iloc[-1], y_values.iloc[-1])
            color = curve_colors[current_quadrant]
            
            chart_label = fx_names.get(pair, pair)
            
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values, mode='lines+markers', name=chart_label,
                line=dict(color=color, width=2), marker=dict(size=5, symbol='circle'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_values.iloc[-1]], y=[y_values.iloc[-1]], mode='markers+text',
                name=f"{pair} (latest)", marker=dict(color=color, size=9, symbol='circle'),
                text=[chart_label], textposition="top center", showlegend=False,
                textfont=dict(color='black', size=10, family='Arial Black')
            ))

    min_x, max_x = rrg_data[[f"{pair}_RS-Ratio" for pair in fx_pairs]].min().min(), rrg_data[[f"{pair}_RS-Ratio" for pair in fx_pairs]].max().max()
    min_y, max_y = rrg_data[[f"{pair}_RS-Momentum" for pair in fx_pairs]].min().min(), rrg_data[[f"{pair}_RS-Momentum" for pair in fx_pairs]].max().max()

    fig.update_layout(
        title=f"FX Relative Rotation Graph (RRG) ({timeframe})",
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        width=700,
        height=600,
        xaxis=dict(range=[min_x, max_x], title_font=dict(size=14)),
        yaxis=dict(range=[min_y, max_y], title_font=dict(size=14)),
        plot_bgcolor='white',
        showlegend=False,
        shapes=[
            dict(type="rect", xref="x", yref="y", x0=min_x, y0=100, x1=100, y1=max_y, fillcolor="lightblue", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=100, y0=100, x1=max_x, y1=max_y, fillcolor="lightgreen", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=min_x, y0=min_y, x1=100, y1=100, fillcolor="pink", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=100, y0=min_y, x1=max_x, y1=100, fillcolor="lightyellow", opacity=0.35, line_width=0),
            dict(type="line", xref="x", yref="y", x0=100, y0=min_y, x1=100, y1=max_y, line=dict(color="black", width=1)),
            dict(type="line", xref="x", yref="y", x0=min_x, y0=100, x1=max_x, y1=100, line=dict(color="black", width=1)),
        ]
    )

    label_font = dict(size=24, color='black', family='Arial Black')
    fig.add_annotation(x=min_x, y=min_y, text="Lagging", showarrow=False, font=label_font, xanchor="left", yanchor="bottom")
    fig.add_annotation(x=max_x, y=min_y, text="Weakening", showarrow=False, font=label_font, xanchor="right", yanchor="bottom")
    fig.add_annotation(x=min_x, y=max_y, text="Improving", showarrow=False, font=label_font, xanchor="left", yanchor="top")
    fig.add_annotation(x=max_x, y=max_y, text="Leading", showarrow=False, font=label_font, xanchor="right", yanchor="top")

    return fig

@st.cache_data
def calculate_weighted_rrg(weekly_data, daily_data, hourly_data, weekly_weight, daily_weight, hourly_weight):
    common_dates = sorted(set(weekly_data.index) & set(daily_data.index) & set(hourly_data.index))[-10:]
    
    weighted_rs_dict = {}
    weighted_rm_dict = {}
    
    for pair in fx_pairs:
        weighted_rs_list = []
        weighted_rm_list = []
        
        for date in common_dates:
            weekly_rs, weekly_rm = calculate_rrg_values(weekly_data[pair].loc[:date], weekly_data[benchmark].loc[:date])
            daily_rs, daily_rm = calculate_rrg_values(daily_data[pair].loc[:date], daily_data[benchmark].loc[:date])
            hourly_rs, hourly_rm = calculate_rrg_values(hourly_data[pair].loc[:date], hourly_data[benchmark].loc[:date])
            
            total_weight = weekly_weight + daily_weight + hourly_weight
            weighted_rs = (weekly_rs.iloc[-1] * weekly_weight + 
                           daily_rs.iloc[-1] * daily_weight + 
                           hourly_rs.iloc[-1] * hourly_weight) / total_weight
            
            weighted_rm = (weekly_rm.iloc[-1] * weekly_weight + 
                           daily_rm.iloc[-1] * daily_weight + 
                           hourly_rm.iloc[-1] * hourly_weight) / total_weight
            
            weighted_rs_list.append(weighted_rs)
            weighted_rm_list.append(weighted_rm)
        
        weighted_rs_dict[pair] = pd.Series(weighted_rs_list, index=common_dates)
        weighted_rm_dict[pair] = pd.Series(weighted_rm_list, index=common_dates)
    
    return weighted_rs_dict, weighted_rm_dict
# Main Streamlit app
st.title("FX Relative Rotation Graph (RRG) Dashboard")

# Sidebar
st.sidebar.header("FX Pairs")

# Get FX data
daily_data, benchmark, fx_pairs, fx_names = get_fx_data("Daily")
hourly_data, _, _, _ = get_fx_data("Hourly")
weekly_data = daily_data.resample('W-FRI').last()

# Add weight sliders
st.sidebar.header("Weighted Composite RRG Settings")
weekly_weight = st.sidebar.slider("Weekly Weight", 0.0, 2.0, 1.2, 0.1)
daily_weight = st.sidebar.slider("Daily Weight", 0.0, 2.0, 1.0, 0.1)
hourly_weight = st.sidebar.slider("Hourly Weight", 0.0, 2.0, 0.8, 0.1)

# Create FX pair buttons
col1, col2 = st.sidebar.columns(2)
columns = [col1, col2]

for i, pair in enumerate(fx_pairs):
    if columns[i % 2].button(fx_names.get(pair, pair)):
        st.session_state.selected_pair = pair

# Trigger Level Input
if 'trigger_level' not in st.session_state:
    st.session_state.trigger_level = ""

st.session_state.trigger_level = st.sidebar.text_input("Trigger Level Input", st.session_state.trigger_level)

# Refresh button
refresh_button = st.sidebar.button("Refresh Data")

if refresh_button:
    st.cache_data.clear()
    st.rerun()

# Main content area
col_daily, col_weekly = st.columns(2)

with col_daily:
    fig_daily = create_rrg_chart(daily_data, benchmark, fx_pairs, fx_names, "Daily", 5)
    st.plotly_chart(fig_daily, use_container_width=True)

with col_weekly:
    fig_weekly = create_rrg_chart(weekly_data, benchmark, fx_pairs, fx_names, "Weekly", 5)
    st.plotly_chart(fig_weekly, use_container_width=True)

# New row for Hourly RRG and Weighted Composite RRG
col_hourly_rrg, col_weighted_rrg = st.columns(2)

with col_hourly_rrg:
    fig_hourly = create_rrg_chart(hourly_data, benchmark, fx_pairs, fx_names, "Hourly", 5)
    st.plotly_chart(fig_hourly, use_container_width=True)

with col_weighted_rrg:
    weighted_rs_dict, weighted_rm_dict = calculate_weighted_rrg(weekly_data, daily_data, hourly_data, weekly_weight, daily_weight, hourly_weight)
    weighted_data = pd.DataFrame()
    for pair in fx_pairs:
        weighted_data[f"{pair}_RS-Ratio"] = weighted_rs_dict[pair]
        weighted_data[f"{pair}_RS-Momentum"] = weighted_rm_dict[pair]
    
    if not weighted_data.empty:
        fig_weighted = create_rrg_chart(weighted_data, benchmark, fx_pairs, fx_names, "Weighted Composite", 10)
        st.plotly_chart(fig_weighted, use_container_width=True)
    else:
        st.warning("Unable to create Weighted Composite RRG chart. Please ensure there is sufficient data.")
    
    # Debug: Print weighted_data
    st.write("Weighted Composite Data:")
    st.write(weighted_data)

# Show raw data if checkbox is selected
if st.checkbox("Show raw data"):
    st.write("Daily Raw data:")
    st.write(daily_data)
    st.write("Hourly Raw data:")
    st.write(hourly_data)
    st.write("FX Pairs:")
    st.write(fx_pairs)
    st.write("Benchmark:")
    st.write(benchmark)
    st.write("Weighted Composite data:")
    st.write(weighted_data)
