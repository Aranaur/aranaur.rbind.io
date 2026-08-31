import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Energy Forecasting Final Exam")
ALPHA = 0.05

# --- METRIC FUNCTIONS ---
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_mape(y_true, y_pred):
    # Уникнення ділення на нуль
    mask = y_true != 0
    if mask.sum() == 0: return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_winkler(y_true, lower, upper, alpha=0.05):
    score = upper - lower
    mask_low = y_true < lower
    mask_up = y_true > upper
    # Penalties
    score[mask_low] += (2/alpha) * (lower[mask_low] - y_true[mask_low])
    score[mask_up] += (2/alpha) * (y_true[mask_up] - upper[mask_up])
    return np.mean(score)

def parse_datetime(df):
    """Robust datetime parsing expecting Date and Time columns or Datetime"""
    df.columns = [c.strip() for c in df.columns]  # Clean column names
    
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    else:
        st.error("CSV must contain 'Datetime' column or 'Date' and 'Time' columns.")
        return None
    return df.sort_values('Datetime').reset_index(drop=True)

# --- SIDEBAR & DATA LOADING ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("Data Upload")
    
    st.markdown("### 1. System Data")
    ground_truth_file = st.file_uploader("Upload Ground Truth (csv)", type='csv')
    train_file = st.file_uploader("Upload Train Data (csv)", type='csv')
    
    st.markdown("---")
    st.markdown("### 2. Student Submission")
    submission_file = st.file_uploader("Upload Your Forecast (csv)", type='csv')

# --- MAIN APP LOGIC ---
st.title("MATH840: Final Examination Dashboard")
st.markdown("""
Upload your forecast file to see how it performs against the Ground Truth.
The dashboard calculates **RMSE**, **MAPE**, and **Winkler Score**.
""")

if ground_truth_file and train_file and submission_file:
    # 1. Load Data
    try:
        df_true = pd.read_csv(ground_truth_file)
        df_train = pd.read_csv(train_file)
        df_sub = pd.read_csv(submission_file)
        
        df_true = parse_datetime(df_true)
        df_train = parse_datetime(df_train)
        df_sub = parse_datetime(df_sub)
        
        # Standardize Value column names for Ground Truth
        value_col_true = [c for c in df_true.columns if 'Value' in c or 'kWh' in c][0]
        df_true = df_true.rename(columns={value_col_true: 'Value'})

        # --- ВИПРАВЛЕННЯ ПОЧИНАЄТЬСЯ ТУТ ---
        # Standardize Value column names for Train Data
        value_col_train = [c for c in df_train.columns if 'Value' in c or 'kWh' in c][0]
        df_train = df_train.rename(columns={value_col_train: 'Value'})
        # --- ВИПРАВЛЕННЯ ЗАКІНЧУЄТЬСЯ ТУТ ---
        
        # Standardize Submission columns (Forecast, Lower_95, Upper_95)
        if 'Forecast' not in df_sub.columns:
             # Try to find the value column
             cols = [c for c in df_sub.columns if c not in ['Date', 'Time', 'Datetime']]
             if len(cols) > 0:
                 df_sub = df_sub.rename(columns={cols[0]: 'Forecast'})
        
        if 'Lower_95' not in df_sub.columns or 'Upper_95' not in df_sub.columns:
            st.warning("⚠️ Submission does not contain 'Lower_95' and 'Upper_95' columns. Winkler Score cannot be calculated properly (setting intervals to 0 width for calculation).")
            if 'Forecast' in df_sub.columns:
                df_sub['Lower_95'] = df_sub['Forecast']
                df_sub['Upper_95'] = df_sub['Forecast']

    except Exception as e:
        st.error(f"Error processing files: {e}")
        st.stop()

    # 2. Generate SNAIVE Benchmark (Seasonal Naive - 1 week lag)
    periods_per_day = 24 * 4
    lag_week = periods_per_day * 7
    
    # Create baseline dataframe matching ground truth structure
    df_snaive = df_true.copy()
    df_snaive['Forecast'] = df_true['Value'].shift(lag_week)
    
    # --- ВИПРАВЛЕННЯ ТУТ ---
    # Тепер ми беремо std() від колонки 'Value', яка точно є числовою
    std_train = df_train['Value'].std() if not df_train.empty else 10
    
    df_snaive['Lower_95'] = df_snaive['Forecast'] - 1.96 * std_train
    df_snaive['Upper_95'] = df_snaive['Forecast'] + 1.96 * std_train
    df_snaive = df_snaive.dropna()
    
    # 3. Merge and Align
    # Student vs Truth
    merged_sub = pd.merge(df_true[['Datetime', 'Value']], df_sub, on='Datetime', how='inner')
    # SNaive vs Truth (aligned to same index)
    merged_snaive = pd.merge(df_true[['Datetime', 'Value']], df_snaive[['Datetime', 'Forecast', 'Lower_95', 'Upper_95']], on='Datetime', how='inner')
    
    # Ensure we are comparing the same timeframes
    common_index = merged_sub['Datetime']
    merged_snaive = merged_snaive[merged_snaive['Datetime'].isin(common_index)]

    if merged_sub.empty:
        st.error("No overlapping dates between Submission and Ground Truth.")
        st.stop()

    # 4. Calculate Metrics
    def get_metrics_row(name, df_merged):
        y_true = df_merged['Value'].values
        y_pred = df_merged['Forecast'].values
        rmse = calculate_rmse(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)
        winkler = calculate_winkler(y_true, df_merged['Lower_95'].values, df_merged['Upper_95'].values, ALPHA)
        return {'Name': name, 'RMSE': rmse, 'MAPE': mape, 'Winkler': winkler}

    metrics_sub = get_metrics_row("Your Submission", merged_sub)
    metrics_snaive = get_metrics_row("SNAIVE Benchmark", merged_snaive)
    
    results_df = pd.DataFrame([metrics_sub, metrics_snaive])
    
    # Calculate Score for plot sizing
    # Simple normalization for score visualization (lower is better)
    results_df['Score_Proxy'] = results_df['RMSE'] + results_df['Winkler'] 

    # 5. Display KPI Row
    st.markdown("### Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("RMSE (Lower is better)", f"{metrics_sub['RMSE']:.2f}", delta=f"{metrics_sub['RMSE'] - metrics_snaive['RMSE']:.2f}", delta_color="inverse")
    with c2:
        st.metric("MAPE (Lower is better)", f"{metrics_sub['MAPE']:.2f}%", delta=f"{metrics_sub['MAPE'] - metrics_snaive['MAPE']:.2f}%", delta_color="inverse")
    with c3:
        st.metric("Winkler (Lower is better)", f"{metrics_sub['Winkler']:.2f}", delta=f"{metrics_sub['Winkler'] - metrics_snaive['Winkler']:.2f}", delta_color="inverse")
    with c4:
        # Check if partial
        is_partial = len(merged_sub) < len(df_true) * 0.95
        status = "⚠️ Partial Forecast" if is_partial else "✅ Full Forecast"
        st.info(f"Status: {status} ({len(merged_sub)} periods)")

    # 6. Main Plot (Forecast vs Truth)
    st.markdown("### Forecast Visualization")
    
    fig_main = go.Figure()
    
    # Truth
    fig_main.add_trace(go.Scatter(
        x=merged_sub['Datetime'], y=merged_sub['Value'],
        mode='lines', name='Ground Truth', line=dict(color='black', width=3)
    ))
    
    # Submission
    # Confidence Interval
    fig_main.add_trace(go.Scatter(
        x=merged_sub['Datetime'], y=merged_sub['Upper_95'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig_main.add_trace(go.Scatter(
        x=merged_sub['Datetime'], y=merged_sub['Lower_95'],
        mode='lines', line=dict(width=0), fill='tonexty', 
        fillcolor='rgba(0, 100, 80, 0.2)', name='95% Confidence', hoverinfo='skip'
    ))
    # Line
    fig_main.add_trace(go.Scatter(
        x=merged_sub['Datetime'], y=merged_sub['Forecast'],
        mode='lines', name='Your Forecast', line=dict(color='#00CC96', width=2)
    ))
    
    fig_main.update_layout(
        xaxis_title='Datetime', yaxis_title='Energy (kWh)',
        hovermode="x unified", template="plotly_white", height=500
    )
    st.plotly_chart(fig_main, use_container_width=True)
    
    # 7. Efficiency Plot
    st.markdown("### Efficiency Analysis (RMSE vs Uncertainty)")
    
    # Add fake data points just to make the chart look like a leaderboard for context
    # In a real app, you might save previous submissions to a database
    dummy_data = [
        {'Name': 'Student A', 'RMSE': metrics_sub['RMSE']*1.2, 'Winkler': metrics_sub['Winkler']*1.1, 'Score_Proxy': 100},
        {'Name': 'Student B', 'RMSE': metrics_sub['RMSE']*0.9, 'Winkler': metrics_sub['Winkler']*1.3, 'Score_Proxy': 90},
        {'Name': 'SNAIVE', 'RMSE': metrics_snaive['RMSE'], 'Winkler': metrics_snaive['Winkler'], 'Score_Proxy': 150},
        metrics_sub
    ]
    plot_df = pd.DataFrame(dummy_data)
    
    fig_eff = px.scatter(
        plot_df, x='RMSE', y='Winkler',
        color='Name', size=[20]*len(plot_df),
        text='Name',
        title="Accuracy vs Uncertainty (Lower-Left is Better)"
    )
    
    fig_eff.update_traces(textposition='top center')
    
    # Add quadrants
    avg_rmse = plot_df['RMSE'].mean()
    avg_wink = plot_df['Winkler'].mean()
    fig_eff.add_vline(x=avg_rmse, line_dash="dash", line_color="grey")
    fig_eff.add_hline(y=avg_wink, line_dash="dash", line_color="grey")
    
    fig_eff.update_layout(template='plotly_white')
    st.plotly_chart(fig_eff, use_container_width=True)

    # 8. Data Table
    with st.expander("View Detailed Metrics Table"):
        st.dataframe(results_df.style.highlight_min(axis=0, color='lightgreen'))

else:
    st.info("👈 Please upload `ground_truth.csv`, `train.csv`, and your `submission.csv` in the sidebar to begin.")
    
    st.markdown("#### Sample Submission Format")
    example_df = pd.DataFrame({
        'Date': ['11/1/2024', '11/1/2024'],
        'Time': ['0:00:00', '0:15:00'],
        'Forecast': [45.2, 46.1],
        'Lower_95': [40.0, 41.0],
        'Upper_95': [50.0, 51.0]
    })
    st.table(example_df)