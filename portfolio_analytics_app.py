import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Portfolio Analytics", layout="wide")

st.title("📊 Portfolio Analytics Dashboard")
st.markdown("Calculate portfolio metrics: Expected Return, Variance, Standard Deviation, and Sharpe Ratio")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Historical Stock Prices (CSV)",
    type=['csv'],
    help="CSV should have 'Date' column and one column per stock with adjusted prices"
)

# Risk-free rate input
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (annual %)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.1,
    help="Enter annual risk-free rate as a percentage (e.g., 3.0 for 3%)"
) / 100  # Convert to decimal

# Main content
if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display raw data preview
        st.subheader("📁 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Identify date column and stock columns
        date_col = df.columns[0]  # Assume first column is date
        stock_cols = df.columns[1:].tolist()
        
        st.info(f"**Detected {len(stock_cols)} stocks:** {', '.join(stock_cols)}")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Portfolio weights input
        st.sidebar.subheader("Portfolio Weights")
        st.sidebar.markdown("*Enter weights for each stock (must sum to 1.0)*")
        
        weights = {}
        for stock in stock_cols:
            weights[stock] = st.sidebar.number_input(
                f"{stock}",
                min_value=0.0,
                max_value=1.0,
                value=1.0/len(stock_cols),  # Equal weights by default
                step=0.01,
                format="%.4f"
            )
        
        # Convert weights to array
        weights_array = np.array(list(weights.values()))
        
        # Check if weights sum to 1
        weights_sum = weights_array.sum()
        if abs(weights_sum - 1.0) > 0.001:
            st.sidebar.error(f"⚠️ Weights sum to {weights_sum:.4f}, must equal 1.0")
        else:
            st.sidebar.success(f"✅ Weights sum to {weights_sum:.4f}")
        
        # Calculate button
        if st.sidebar.button("Calculate Portfolio Metrics", type="primary"):
            
            # Calculate log returns: ln(P_t / P_t-1)
            returns_df = pd.DataFrame()
            for stock in stock_cols:
                returns_df[stock] = np.log(df[stock] / df[stock].shift(1))
            
            # Remove first row (NaN from shift)
            returns_df = returns_df.dropna()
            
            # Display returns preview
            st.subheader("📈 Calculated Log Returns")
            st.dataframe(returns_df.head(10), use_container_width=True)
            
            # Calculate statistics
            mean_returns = returns_df.mean()  # Mean monthly return for each stock
            cov_matrix = returns_df.cov()  # Covariance matrix
            
            # Portfolio metrics
            # Expected portfolio return (monthly)
            portfolio_return_monthly = np.dot(weights_array, mean_returns)
            
            # Annualized expected return (assuming 12 months)
            portfolio_return_annual = portfolio_return_monthly * 12
            
            # Portfolio variance
            portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
            
            # Portfolio standard deviation (monthly)
            portfolio_std_monthly = np.sqrt(portfolio_variance)
            
            # Annualized standard deviation
            portfolio_std_annual = portfolio_std_monthly * np.sqrt(12)
            
            # Sharpe ratio (using annualized figures)
            sharpe_ratio = (portfolio_return_annual - risk_free_rate) / portfolio_std_annual
            
            # Display results
            st.subheader("🎯 Portfolio Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Return (Annual)",
                    f"{portfolio_return_annual*100:.2f}%",
                    help="Annualized expected portfolio return"
                )
                st.metric(
                    "Expected Return (Monthly)",
                    f"{portfolio_return_monthly*100:.2f}%",
                    help="Monthly expected portfolio return"
                )
            
            with col2:
                st.metric(
                    "Portfolio Variance",
                    f"{portfolio_variance:.6f}",
                    help="Portfolio variance (monthly)"
                )
            
            with col3:
                st.metric(
                    "Std Deviation (Annual)",
                    f"{portfolio_std_annual*100:.2f}%",
                    help="Annualized portfolio standard deviation (volatility)"
                )
                st.metric(
                    "Std Deviation (Monthly)",
                    f"{portfolio_std_monthly*100:.2f}%",
                    help="Monthly portfolio standard deviation"
                )
            
            with col4:
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe_ratio:.4f}",
                    help="Risk-adjusted return metric"
                )
            
            # Detailed breakdown
            st.subheader("📊 Detailed Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Individual Stock Statistics (Monthly)**")
                stats_df = pd.DataFrame({
                    'Stock': stock_cols,
                    'Weight': weights_array,
                    'Mean Return': mean_returns.values,
                    'Std Dev': returns_df.std().values,
                    'Weighted Return': weights_array * mean_returns.values
                })
                st.dataframe(stats_df.style.format({
                    'Weight': '{:.4f}',
                    'Mean Return': '{:.6f}',
                    'Std Dev': '{:.6f}',
                    'Weighted Return': '{:.6f}'
                }), use_container_width=True)
            
            with col2:
                st.markdown("**Covariance Matrix**")
                st.dataframe(cov_matrix.style.format('{:.6f}'), use_container_width=True)
            
            # Summary for download
            st.subheader("📥 Download Results")
            
            summary_text = f"""Portfolio Analytics Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO COMPOSITION:
{chr(10).join([f'{stock}: {weight:.4f}' for stock, weight in weights.items()])}

PORTFOLIO METRICS:
Expected Return (Annual): {portfolio_return_annual*100:.2f}%
Expected Return (Monthly): {portfolio_return_monthly*100:.2f}%
Portfolio Variance: {portfolio_variance:.6f}
Standard Deviation (Annual): {portfolio_std_annual*100:.2f}%
Standard Deviation (Monthly): {portfolio_std_monthly*100:.2f}%
Sharpe Ratio: {sharpe_ratio:.4f}
Risk-Free Rate: {risk_free_rate*100:.2f}%

INDIVIDUAL STOCK STATISTICS (Monthly):
{stats_df.to_string(index=False)}

COVARIANCE MATRIX:
{cov_matrix.to_string()}
"""
            
            st.download_button(
                label="Download Summary (TXT)",
                data=summary_text,
                file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV has a Date column (first column) and stock price columns.")

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file with historical monthly stock prices to begin")
    
    st.markdown("""
    ### 📋 Instructions:
    
    1. **Prepare your CSV file** with the following format:
       - First column: Date (e.g., 2024-01-31, 2024-02-29, etc.)
       - Subsequent columns: Adjusted closing prices for each stock
       
    2. **Example CSV format:**
    ```
    Date,AAPL,MSFT,GOOGL
    2024-01-31,150.25,380.50,140.75
    2024-02-29,155.30,385.20,142.30
    2024-03-31,158.40,390.10,145.60
    ```
    
    3. **Enter portfolio weights** in the sidebar (must sum to 1.0)
    
    4. **Set the risk-free rate** (annual percentage)
    
    5. **Click "Calculate Portfolio Metrics"** to see results
    
    ### 📐 Calculations:
    
    - **Stock Returns**: ln(Price_t / Price_t-1)
    - **Expected Portfolio Return**: Weighted average of individual stock returns
    - **Portfolio Variance**: w^T × Σ × w (where Σ is the covariance matrix)
    - **Portfolio Std Dev**: √(Portfolio Variance)
    - **Sharpe Ratio**: (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
    
    ### 💡 Tips:
    - Use monthly adjusted closing prices for accurate results
    - Ensure data is sorted chronologically
    - The app treats the historical data as a sample for statistical calculations
    """)

# Footer
st.markdown("---")
st.markdown("*Portfolio Analytics App | Built with Streamlit*")
