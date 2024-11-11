import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="FX Relative Rotation Graph (RRG) Dashboard")

st.warning("""
    **Disclaimer:**
    - This app is for educational purposes only and should not be considered as financial advice.
    - We do not guarantee the accuracy of the data. The data source is Yahoo Finance, which may have limitations or inaccuracies.
    - Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.
""")


# 初始化 session state 變量
if 'new_pair_selected' not in st.session_state:
    st.session_state.new_pair_selected = False

if 'trigger_level' not in st.session_state:
    st.session_state.trigger_level = ""

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

def create_rrg_chart(data, benchmark, fx_pairs, fx_names, timeframe, tail_length):
    if timeframe == "Weekly":
        data_resampled = data.resample('W-FRI').last()
    elif timeframe == "Hourly":
        data_resampled = data
    else:  # Daily
        data_resampled = data

    rrg_data = pd.DataFrame()
    for pair in fx_pairs:
        rs_ratio, rs_momentum = calculate_rrg_values(data_resampled[pair], data_resampled[benchmark])
        rrg_data[f"{pair}_RS-Ratio"] = rs_ratio
        rrg_data[f"{pair}_RS-Momentum"] = rs_momentum

    # Calculate the min and max values for the plotted data points
    plotted_data = rrg_data.iloc[-max(tail_length, 15):]
    min_x = plotted_data[[f"{pair}_RS-Ratio" for pair in fx_pairs]].min().min()
    max_x = plotted_data[[f"{pair}_RS-Ratio" for pair in fx_pairs]].max().max()
    min_y = plotted_data[[f"{pair}_RS-Momentum" for pair in fx_pairs]].min().min()
    max_y = plotted_data[[f"{pair}_RS-Momentum" for pair in fx_pairs]].max().max()

    padding = 0.05  # Increased padding
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    min_x -= range_x * padding
    max_x += range_x * padding
    min_y -= range_y * padding
    max_y += range_y * padding
    
    center_x = 100
    center_y = 100

    fig = go.Figure()

    quadrant_colors = {"Lagging": "pink", "Weakening": "lightyellow", "Improving": "lightblue", "Leading": "lightgreen"}
    curve_colors = {"Lagging": "red", "Weakening": "orange", "Improving": "darkblue", "Leading": "darkgreen"}

    def get_quadrant(x, y):
        if x < center_x and y < center_y: return "Lagging"
        elif x >= center_x and y < center_y: return "Weakening"
        elif x < center_x and y >= center_y: return "Improving"
        else: return "Leading"

    for pair in fx_pairs:
        x_values = rrg_data[f"{pair}_RS-Ratio"].dropna()
        y_values = rrg_data[f"{pair}_RS-Momentum"].dropna()
        
        if len(x_values) > 0 and len(y_values) > 0:
            x_values = x_values.iloc[-tail_length:]
            y_values = y_values.iloc[-tail_length:]
            
            current_quadrant = get_quadrant(x_values.iloc[-1], y_values.iloc[-1])
            color = curve_colors[current_quadrant]
            
            chart_label = fx_names.get(pair, pair)
            
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values, mode='lines+markers', name=chart_label,
                line=dict(color=color, width=2), marker=dict(size=5, symbol='circle'),
                showlegend=False
            ))
            
            if len(y_values) > 1:
                text_position = "top center" if y_values.iloc[-1] > y_values.iloc[-2] else "bottom center"
            else:
                text_position = "top center"
            
            fig.add_trace(go.Scatter(
                x=[x_values.iloc[-1]], y=[y_values.iloc[-1]], mode='markers+text',
                name=f"{pair} (latest)", marker=dict(color=color, size=9, symbol='circle'),
                text=[chart_label], textposition=text_position, showlegend=False,
                textfont=dict(color='black', size=10, family='Arial Black')
            ))

    fig.update_layout(
        title=f"FX Relative Rotation Graph (RRG) ({timeframe})",
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        width=700,
        height=600,  # Adjusted to match original layout
        xaxis=dict(range=[min_x, max_x], title_font=dict(size=14)),
        yaxis=dict(range=[min_y, max_y], title_font=dict(size=14)),
        plot_bgcolor='white',
        showlegend=False,
        shapes=[
            dict(type="rect", xref="x", yref="y", x0=min_x, y0=center_y, x1=center_x, y1=max_y, fillcolor="lightblue", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=center_x, y0=center_y, x1=max_x, y1=max_y, fillcolor="lightgreen", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=min_x, y0=min_y, x1=center_x, y1=center_y, fillcolor="pink", opacity=0.35, line_width=0),
            dict(type="rect", xref="x", yref="y", x0=center_x, y0=min_y, x1=max_x, y1=center_y, fillcolor="lightyellow", opacity=0.35, line_width=0),
            dict(type="line", xref="x", yref="y", x0=center_x, y0=min_y, x1=center_x, y1=max_y, line=dict(color="black", width=1)),
            dict(type="line", xref="x", yref="y", x0=min_x, y0=center_y, x1=max_x, y1=center_y, line=dict(color="black", width=1)),
        ]
    )

    label_font = dict(size=24, color='black', family='Arial Black')
    fig.add_annotation(x=min_x, y=min_y, text="落後", showarrow=False, font=label_font, xanchor="left", yanchor="bottom")
    fig.add_annotation(x=max_x, y=min_y, text="轉弱", showarrow=False, font=label_font, xanchor="right", yanchor="bottom")
    fig.add_annotation(x=min_x, y=max_y, text="改善", showarrow=False, font=label_font, xanchor="left", yanchor="top")
    fig.add_annotation(x=max_x, y=max_y, text="領先", showarrow=False, font=label_font, xanchor="right", yanchor="top")

    return fig

@st.cache_data
def get_hourly_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20)
    
    usd_based_tickers = {
        "CADUSD=X": "USDCAD=X",
        "JPYUSD=X": "USDJPY=X",
        "CNYUSD=X": "USDCNY=X",
        "CHFUSD=X": "USDCHF=X"
    }
    
    # Use the USD-based ticker if it's one of the special cases
    download_ticker = usd_based_tickers.get(ticker, ticker)
    
    try:
        # Get all OHLC data
        data = yf.download(
            download_ticker, 
            start=start_date, 
            end=end_date, 
            interval="1h",
            progress=False  # Disable progress bar
        )  # By default, yf.download gets all OHLC columns
        
        if data.empty:
            st.warning(f"No data available for {download_ticker}")
            return pd.DataFrame()

        # Ensure we have all required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            st.warning(f"Missing required columns in data. Available columns: {data.columns}")
            return pd.DataFrame()

        # Handle datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Remove weekends and NaN values
        data = data.dropna()
        data = data[data.index.dayofweek < 5]
        
        # Handle inverse pairs
        if ticker in usd_based_tickers:
            for col in ['Open', 'High', 'Low', 'Close']:
                data[col] = 1 / data[col]
        
        return data
        
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return pd.DataFrame()

# Test the data structure (add this temporarily for debugging)
if st.button("Test Data Download"):
    test_ticker = "GBPUSD=X"  # or any other pair you want to test
    test_data = get_hourly_data(test_ticker)
    if not test_data.empty:
        st.write("Available columns:", test_data.columns)
        st.write("First few rows of data:")
        st.write(test_data.head())

def create_line_chart(data, ticker, trigger_level=None):
    usd_based_tickers = {
        "CADUSD=X": "USDCAD=X",
        "JPYUSD=X": "USDJPY=X",
        "CNYUSD=X": "USDCNY=X",
        "CHFUSD=X": "USDCHF=X"
    }
    display_ticker = usd_based_tickers.get(ticker, ticker)
    
    fig = go.Figure(data=[go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price'
    )])
    
    fig.update_layout(
        title=f"{display_ticker} - Hourly Price Chart (Last 20 Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=700,
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickformat='%Y-%m-%d %H:%M',
            tickmode='auto',
            nticks=10,
        )
    )
    
    if trigger_level is not None:
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=trigger_level,
            x1=data.index[-1],
            y1=trigger_level,
            line=dict(color="blue", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=data.index[-1],
            y=trigger_level,
            text=f"Trigger: {trigger_level}",
            showarrow=False,
            yshift=10,
            xshift=10,
            font=dict(color="blue"),
        )
    
    return fig


# Main Streamlit app
st.title("FX Relative Rotation Graph (RRG) Dashboard")

# Sidebar
st.sidebar.header("FX Pairs")

# Get FX data
daily_data, benchmark, fx_pairs, fx_names = get_fx_data("Daily")
hourly_data, _, _, _ = get_fx_data("Hourly")

# Sidebar FX pair buttons
col1, col2 = st.sidebar.columns(2)
columns = [col1, col2]

# Debug the selected pair
st.sidebar.write("Current selected pair:", st.session_state.get('selected_pair', 'None'))

for i, pair in enumerate(fx_pairs):
    button_label = fx_names.get(pair, pair)
    if columns[i % 2].button(button_label, key=f"btn_{pair}"):  # Add unique key
        st.session_state.selected_pair = pair
        st.session_state.trigger_level = ""  # Clear the trigger level input
        st.rerun()


# Trigger Level Input
if 'trigger_level' not in st.session_state:
    st.session_state.trigger_level = ""

new_trigger_level = st.sidebar.text_input("Trigger Level Input", value=st.session_state.trigger_level, key="trigger_level_input")

if new_trigger_level != st.session_state.trigger_level:
    st.session_state.trigger_level = new_trigger_level
    st.rerun()
    
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
    fig_weekly = create_rrg_chart(daily_data.resample('W-FRI').last(), benchmark, fx_pairs, fx_names, "Weekly", 5)
    st.plotly_chart(fig_weekly, use_container_width=True)

# New row for Hourly RRG and Candlestick chart
col_hourly_rrg, col_candlestick = st.columns(2)

with col_hourly_rrg:
    fig_hourly = create_rrg_chart(hourly_data, benchmark, fx_pairs, fx_names, "Hourly", 5)
    st.plotly_chart(fig_hourly, use_container_width=True)


with col_candlestick:
    if 'selected_pair' in st.session_state:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=20)
            
            # Get data
            data = yf.download(
                st.session_state.selected_pair,
                start=start_date,
                end=end_date,
                interval="1h",
                progress=False
            )
            
            if not data.empty:
                # Fix multi-level columns if they exist
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Convert index to datetime if not already
                data.index = pd.to_datetime(data.index)
                
                # Filter out weekends and non-trading hours
                mask = (data.index.dayofweek < 5) & (data['Close'].notna())
                data = data.loc[mask]
                
                # Sort index to ensure proper sequence
                data = data.sort_index()
                
                # Calculate price range for scaling
                price_min = min(data['Low'].min(), data['Close'].min())
                price_max = max(data['High'].max(), data['Close'].max())
                padding = (price_max - price_min) * 0.05
                
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    increasing_line_color='red',
                    decreasing_line_color='green'
                )])
                
                # Update layout
                fig.update_layout(
                    title=f"{st.session_state.selected_pair} - Hourly Candlestick Chart",
                    yaxis=dict(
                        title="Price",
                        range=[price_min - padding, price_max + padding],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGrey',
                        zeroline=True
                    ),
                    xaxis=dict(
                        title="Date",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGrey',
                        rangeslider=dict(visible=False),
                        type='date',
                        tickformat='%Y-%m-%d\n%H:%M',
                        dtick=86400000.0,  # Show one tick per day
                    ),
                    height=700,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Add trigger level if specified
                if st.session_state.trigger_level:
                    try:
                        trigger_value = float(st.session_state.trigger_level)
                        fig.add_hline(
                            y=trigger_value,
                            line_dash="dash",
                            line_color="blue",
                            annotation_text=f"Trigger: {trigger_value}",
                            annotation_position="right"
                        )
                    except ValueError:
                        st.warning("Invalid trigger level value")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data info
                with st.expander("Show data info"):
                    st.write(f"Trading points: {len(data)}")
                    st.write(f"Date range: {data.index.min()} to {data.index.max()}")
            else:
                st.warning(f"No data available for {st.session_state.selected_pair}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.write("Select an FX pair to view the candlestick chart.")
       
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




















