import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-01-01"
    data = fetch_data(ticker, start_date, end_date)
    data.to_csv(f"{ticker}_data.csv")