import pandas as pd
import yfinance as yf
import numpy as np 

start_date = '2007-01-01'  # Adjust start date as needed
end_date = '2024-01-01'    # Adjust end date as needed

# Fetch historical data for WTI Crude Oil Futures (daily frequency)
oil_ticker = 'CL=F'
oil_data = yf.download(oil_ticker, start=start_date, end=end_date, interval='1d')
oil_prices = oil_data['Adj Close']
oil_returns = oil_prices.pct_change()

# Define tickers for the ETFs dataset
etf_tickers = ['AGG', 'DBC', '^VIX', 'VTI']
etf_data = yf.download(etf_tickers, start=start_date, end=end_date, interval='1d')
etf_prices = etf_data['Adj Close']
etf_returns = etf_prices.pct_change()

# Define NASDAQ tickers (example with a few stocks)
nasdaq_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Extend to 500 stocks
nasdaq_data = yf.download(nasdaq_tickers, start=start_date,end=end_date, interval='1d')
nasdaq_prices = nasdaq_data['Adj Close']
nasdaq_returns = nasdaq_prices.pct_change()



#Join Data 
returns = pd.concat([oil_returns,etf_returns,nasdaq_returns], axis=1)
returns = returns.dropna()
returns = returns.reset_index(drop=True)
returns.to_csv('returns.csv')