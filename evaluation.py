import pandas as pd
import matplotlib.pyplot as plt

def plot_results(original_data, predicted_data):
    plt.figure(figsize=(14, 5))
    plt.plot(original_data, color='blue', label='Original Data')
    plt.plot(predicted_data, color='red', label='Predicted Data')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    original_data = pd.read_csv("scaled_data.csv")['Close'].values
    predicted_data = pd.read_csv("predictions.csv")['Predicted_Close'].values
    plot_results(original_data, predicted_data)