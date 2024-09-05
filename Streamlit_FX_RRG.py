import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="外匯相對旋轉圖（RRG）儀表板")

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
    if timeframe != "加權複合":
        rrg_data = pd.DataFrame()
        for pair in fx_pairs:
            rs_ratio, rs_momentum = calculate_rrg_values(data[pair], data[benchmark])
            rrg_data[f"{pair}_RS-Ratio"] = rs_ratio
            rrg_data[f"{pair}_RS-Momentum"] = rs_momentum
    else:
        rrg_data = data

    def get_quadrant(x, y):
        x = float(x)
        y = float(y)
        if x < 100 and y < 100: return "落後"
        elif x >= 100 and y < 100: return "轉弱"
        elif x < 100 and y >= 100: return "改善"
        else: return "領先"

    fig = go.Figure()

    quadrant_colors = {"落後": "pink", "轉弱": "lightyellow", "改善": "lightblue", "領先": "lightgreen"}
    curve_colors = {"落後": "red", "轉弱": "orange", "改善": "darkblue", "領先": "darkgreen"}

    for pair in fx_pairs:
        x_values = rrg_data[f"{pair}_RS-Ratio"].dropna().iloc[-tail_length:].tolist()
        y_values = rrg_data[f"{pair}_RS-Momentum"].dropna().iloc[-tail_length:].tolist()
        
        if x_values and y_values:
            current_quadrant = get_quadrant(x_values[-1], y_values[-1])
            color = curve_colors[current_quadrant]
            
            chart_label = fx_names.get(pair, pair)
            
            fig.add_trace(go.Scatter(
                x=x_values, y=y_values, mode='lines+markers', name=chart_label,
                line=dict(color=color, width=2), marker=dict(size=5, symbol='circle'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_values[-1]], y=[y_values[-1]], mode='markers+text',
                name=f"{pair} (最新)", marker=dict(color=color, size=9, symbol='circle'),
                text=[chart_label], textposition="top center", showlegend=False,
                textfont=dict(color='black', size=10, family='Arial Black')
            ))

            
    fig.update_layout(
        title=f"外匯相對旋轉圖（RRG）({timeframe})",
        xaxis_title="RS-比率",
        yaxis_title="RS-動量",
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
    
    download_ticker = usd_based_tickers.get(ticker, ticker)
    
    data = yf.download(download_ticker, start=start_date, end=end_date, interval="1h")
    
    if data.empty:
        st.warning(f"無法獲取 {download_ticker} 的數據")
        return pd.DataFrame()

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    data = data.dropna()
    data = data[data.index.dayofweek < 5]
    
    if data.empty:
        st.warning(f"移除週末和空值後，{download_ticker} 沒有有效數據")
        return pd.DataFrame()
    
    data = data.reset_index()
    data['continuous_datetime'] = pd.date_range(start=data['Datetime'].min(), periods=len(data), freq='H')
    data.set_index('continuous_datetime', inplace=True)
    
    return data

def create_candlestick_chart(data, ticker, trigger_level=None):
    usd_based_tickers = {
        "CADUSD=X": "USDCAD=X",
        "JPYUSD=X": "USDJPY=X",
        "CNYUSD=X": "USDCNY=X",
        "CHFUSD=X": "USDCHF=X"
    }
    display_ticker = usd_based_tickers.get(ticker, ticker)
    
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'])])
    
    fig.update_layout(
        title=f"{display_ticker} - 小時蠟燭圖（最近20天）",
        xaxis_title="日期",
        yaxis_title="價格",
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
            text=f"觸發: {trigger_level}",
            showarrow=False,
            yshift=10,
            xshift=10,
            font=dict(color="blue"),
        )
    
    return fig

@st.cache_data
def calculate_weighted_rrg(weekly_data, daily_data, hourly_data, weekly_weight, daily_weight, hourly_weight):
    # 確保所有數據都有相同的最新 10 個時間點
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
            
            weighted_rs = (weekly_rs.iloc[-1] * weekly_weight + 
                           daily_rs.iloc[-1] * daily_weight + 
                           hourly_rs.iloc[-1] * hourly_weight) / (weekly_weight + daily_weight + hourly_weight)
            
            weighted_rm = (weekly_rm.iloc[-1] * weekly_weight + 
                           daily_rm.iloc[-1] * daily_weight + 
                           hourly_rm.iloc[-1] * hourly_weight) / (weekly_weight + daily_weight + hourly_weight)
            
            weighted_rs_list.append(weighted_rs)
            weighted_rm_list.append(weighted_rm)
        
        weighted_rs_dict[pair] = pd.Series(weighted_rs_list, index=common_dates)
        weighted_rm_dict[pair] = pd.Series(weighted_rm_list, index=common_dates)
    
    return weighted_rs_dict, weighted_rm_dict

# 主 Streamlit 應用程序
st.title("外匯相對旋轉圖（RRG）儀表板")

# 側邊欄
st.sidebar.header("外匯對")

# 獲取外匯數據
daily_data, benchmark, fx_pairs, fx_names = get_fx_data("Daily")
hourly_data, _, _, _ = get_fx_data("Hourly")
weekly_data = daily_data.resample('W-FRI').last()

# 添加權重滑塊
st.sidebar.header("加權複合 RRG 設置")
weekly_weight = st.sidebar.slider("週線權重", 0.0, 2.0, 1.2, 0.1)
daily_weight = st.sidebar.slider("日線權重", 0.0, 2.0, 1.0, 0.1)
hourly_weight = st.sidebar.slider("小時線權重", 0.0, 2.0, 0.8, 0.1)

# 創建外匯對按鈕
col1, col2 = st.sidebar.columns(2)
columns = [col1, col2]

for i, pair in enumerate(fx_pairs):
    if columns[i % 2].button(fx_names.get(pair, pair)):
        st.session_state.selected_pair = pair

# 觸發水平輸入
if 'trigger_level' not in st.session_state:
    st.session_state.trigger_level = ""

st.session_state.trigger_level = st.sidebar.text_input("觸發水平輸入", st.session_state.trigger_level)

# 刷新按鈕
refresh_button = st.sidebar.button("刷新數據")

if refresh_button:
    st.cache_data.clear()
    st.rerun()

# 主內容區域
col_daily, col_weekly = st.columns(2)

with col_daily:
    fig_daily = create_rrg_chart(daily_data, benchmark, fx_pairs, fx_names, "日線", 5)
    st.plotly_chart(fig_daily, use_container_width=True)

with col_weekly:
    fig_weekly = create_rrg_chart(weekly_data, benchmark, fx_pairs, fx_names, "週線", 5)
    st.plotly_chart(fig_weekly, use_container_width=True)

# 小時線 RRG 和加權複合 RRG 的新行
col_hourly_rrg, col_weighted_rrg = st.columns(2)

with col_hourly_rrg:
    fig_hourly = create_rrg_chart(hourly_data, benchmark, fx_pairs, fx_names, "小時線", 5)
    st.plotly_chart(fig_hourly, use_container_width=True)

with col_weighted_rrg:
    weighted_rs_dict, weighted_rm_dict = calculate_weighted_rrg(weekly_data, daily_data, hourly_data, weekly_weight, daily_weight, hourly_weight)
    weighted_data = pd.DataFrame()
    for pair in fx_pairs:
        weighted_data[f"{pair}_RS-Ratio"] = weighted_rs_dict[pair]
        weighted_data[f"{pair}_RS-Momentum"] = weighted_rm_dict[pair]
    
    if not weighted_data.empty:
        fig_weighted = create_rrg_chart(weighted_data, benchmark, fx_pairs, fx_names, "加權複合", 10)
        st.plotly_chart(fig_weighted, use_container_width=True)
    else:
        st.warning("無法創建加權複合 RRG 圖表。請確保有足夠的數據。")

# 對於蠟燭圖，確保在創建圖表之前檢查數據是否為空：
if 'selected_pair' in st.session_state:
    pair_hourly_data = get_hourly_data(st.session_state.selected_pair)
    
    if not pair_hourly_data.empty:
        # 將 trigger_level 轉換為浮點數（如果不為空）
        trigger_level_float = None
        if st.session_state.trigger_level:
            try:
                trigger_level_float = float(st.session_state.trigger_level)
            except ValueError:
                st.warning("無效的觸發水平。請輸入有效數字。")
        
        fig_candlestick = create_candlestick_chart(pair_hourly_data, st.session_state.selected_pair, trigger_level_float)
        
        # 蠟燭圖的重置按鈕
        if st.button("重置蠟燭圖"):
            del st.session_state.selected_pair
            st.session_state.trigger_level = ""
            st.rerun()
        
        st.plotly_chart(fig_candlestick, use_container_width=True)
    else:
        st.warning(f"無法獲取 {st.session_state.selected_pair} 的有效數據")
else:
    st.write("選擇一個外匯對以查看蠟燭圖。")

# 如果選中複選框則顯示原始數據
if st.checkbox("顯示原始數據"):
    st.write("日線原始數據:")
    st.write(daily_data)
    st.write("小時線原始數據:")
    st.write(hourly_data)
    st.write("外匯對:")
    st.write(fx_pairs)
    st.write("基準:")
    st.write(benchmark)
