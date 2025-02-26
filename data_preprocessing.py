import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

if __name__ == "__main__":
    file_path = "AAPL_data.csv"
    scaled_data, scaler = preprocess_data(file_path)
    pd.DataFrame(scaled_data, columns=['Close']).to_csv("scaled_data.csv", index=False)