import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="FX Relative Rotation Graph (RRG)")

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
    else:  # Daily
        start_date = end_date - timedelta(days=500)

    benchmark = "HKDUSD=X"
    fx_pairs = ["GBPUSD=X", "EURUSD=X", "AUDUSD=X", "NZDUSD=X", "CADUSD=X", "CHFUSD=X", "JPYUSD=X", "CNYUSD=X", 
                "EURGBP=X", "AUDNZD=X", "AUDCAD=X", "NZDCAD=X", "DX-Y.NYB"]
    
    fx_names = {
        "GBPUSD=X": "GBP", "EURUSD=X": "EUR", "AUDUSD=X": "AUD", "NZDUSD=X": "NZD",
        "CADUSD=X": "CAD", "JPYUSD=X": "JPY", "EURGBP=X": "EURGBP", "AUDNZD=X": "AUDNZD",
        "AUDCAD=X": "AUDCAD", "NZDCAD=X": "NZDCAD", "DX-Y.NYB": "DXY", "CHFUSD=X": "CHF", "CNYUSD=X": "CNY"
    }

    tickers_to_download = [benchmark] + fx_pairs
    data = yf.download(tickers_to_download, start=start_date, end=end_date)['Close']
    
    return data, benchmark, fx_pairs, fx_names

def create_rrg_chart(data, benchmark, fx_pairs, fx_names, timeframe, tail_length):
    if timeframe == "Weekly":
        data_resampled = data.resample('W-FRI').last()
    else:  # Daily
        data_resampled = data

    rrg_data = pd.DataFrame()
    for pair in fx_pairs:
        rs_ratio, rs_momentum = calculate_rrg_values(data_resampled[pair], data_resampled[benchmark])
        rrg_data[f"{pair}_RS-Ratio"] = rs_ratio
        rrg_data[f"{pair}_RS-Momentum"] = rs_momentum

    boundary_data = rrg_data.iloc[-10:]
    
    padding = 0.1
    min_x = boundary_data[[f"{pair}_RS-Ratio" for pair in fx_pairs]].min().min()
    max_x = boundary_data[[f"{pair}_RS-Ratio" for pair in fx_pairs]].max().max()
    min_y = boundary_data[[f"{pair}_RS-Momentum" for pair in fx_pairs]].min().min()
    max_y = boundary_data[[f"{pair}_RS-Momentum" for pair in fx_pairs]].max().max()

    range_x = max_x - min_x
    range_y = max_y - min_y
    min_x = max(min_x - range_x * padding, 60)
    max_x = min(max_x + range_x * padding, 140)
    min_y = max(min_y - range_y * padding, 70)
    max_y = min(max_y + range_y * padding, 130)

    fig = go.Figure()

    quadrant_colors = {"Lagging": "pink", "Weakening": "lightyellow", "Improving": "lightblue", "Leading": "lightgreen"}
    curve_colors = {"Lagging": "red", "Weakening": "orange", "Improving": "darkblue", "Leading": "darkgreen"}

    def get_quadrant(x, y):
        if x < 100 and y < 100: return "Lagging"
        elif x >= 100 and y < 100: return "Weakening"
        elif x < 100 and y >= 100: return "Improving"
        else: return "Leading"

    for pair in fx_pairs:
        x_values = rrg_data[f"{pair}_RS-Ratio"].iloc[-tail_length:].dropna()
        y_values = rrg_data[f"{pair}_RS-Momentum"].iloc[-tail_length:].dropna()
        if len(x_values) > 0 and len(y_values) > 0:
            current_quadrant = get_quadrant(x_values.iloc[-1], y_values.iloc[-1])
            color = curve_colors[current_quadrant]
            
            legend_label = f"{pair} ({fx_names.get(pair, '')})"
            chart_label = fx_names.get(pair, pair)
            
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values, mode='lines+markers', name=legend_label,
                line=dict(color=color, width=2), marker=dict(size=6, symbol='circle'),
                legendgroup=pair, showlegend=True
            ))
            
            text_position = "top center" if y_values.iloc[-1] > y_values.iloc[-2] else "bottom center"
            
            fig.add_trace(go.Scatter(
                x=[x_values.iloc[-1]], y=[y_values.iloc[-1]], mode='markers+text',
                name=f"{pair} (latest)", marker=dict(color=color, size=12, symbol='circle'),
                text=[chart_label], textposition=text_position, legendgroup=pair, showlegend=False,
                textfont=dict(color='black', size=12, family='Arial Black')
            ))

    fig.update_layout(
        title=f"FX Relative Rotation Graph (RRG) ({timeframe})",
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        width=1200,
        height=800,
        xaxis=dict(range=[min_x, max_x], title_font=dict(size=14)),
        yaxis=dict(range=[min_y, max_y], title_font=dict(size=14)),
        plot_bgcolor='white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, title=f"Legend<br>Benchmark: {benchmark}"),
        shapes=[
            dict(type="rect", xref="x", yref="y", x0=min_x, y0=100, x1=100, y1=max_y, fillcolor="lightblue", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=100, y0=100, x1=max_x, y1=max_y, fillcolor="lightgreen", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=min_x, y0=min_y, x1=100, y1=100, fillcolor="pink", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=100, y0=min_y, x1=max_x, y1=100, fillcolor="lightyellow", opacity=0.35, line_width=0),
            dict(type="line", xref="x", yref="y", x0=100, y0=min_y, x1=100, y1=max_y, line=dict(color="black", width=1)),
            dict(type="line", xref="x", yref="y", x0=min_x, y0=100, x1=max_x, y1=100, line=dict(color="black", width=1)),
        ]
    )

    label_font = dict(size=32, color='black', family='Arial Black')
    fig.add_annotation(x=min_x, y=min_y, text="落後", showarrow=False, font=label_font, xanchor="left", yanchor="bottom")
    fig.add_annotation(x=max_x, y=min_y, text="轉弱", showarrow=False, font=label_font, xanchor="right", yanchor="bottom")
    fig.add_annotation(x=min_x, y=max_y, text="改善", showarrow=False, font=label_font, xanchor="left", yanchor="top")
    fig.add_annotation(x=max_x, y=max_y, text="領先", showarrow=False, font=label_font, xanchor="right", yanchor="top")

    return fig

# Main Streamlit app
st.title("FX Relative Rotation Graph (RRG)")

# Sidebar
st.sidebar.header("Chart Settings")

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    options=["Weekly", "Daily"],
    key="timeframe_selector"
)

tail_length = st.sidebar.slider(
    "Tail Length",
    min_value=1,
    max_value=52,
    value=5,
    step=1,
    help="Number of data points to show in the chart"
)

# Main content area
data, benchmark, fx_pairs, fx_names = get_fx_data(timeframe)
if data is not None and not data.empty:
    fig = create_rrg_chart(data, benchmark, fx_pairs, fx_names, timeframe, tail_length)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Latest Data")
    st.dataframe(data.tail())
else:
    st.error("No data available for the selected timeframe. Please try a different selection.")

if st.checkbox("Show raw data"):
    st.write("Raw data:")
    st.write(data)
    st.write("FX Pairs:")
    st.write(fx_pairs)
    st.write("Benchmark:")
    st.write(benchmark)
