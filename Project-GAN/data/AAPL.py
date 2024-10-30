# import yfinance as yf

# # Define the time range
# aapl = yf.Ticker("AAPL")

# # Fetch historical data since 2006
# aapl_data = aapl.history(start="2014-10-29", end=None)
# aapl_data = aapl_data[['Open', 'High', 'Low', 'Close', 'Volume']]
# # Show the first few rows
# print(aapl_data.head())

# # Save it to a CSV file
# aapl_data.to_csv("aapl_data.csv")