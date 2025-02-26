import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM


def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


def train_model(data, time_step=100):
    X, Y = create_dataset(data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, batch_size=1, epochs=1)

    model.save("lstm_model.h5")
    return model


if __name__ == "__main__":
    data = pd.read_csv("scaled_data.csv").values
    model = train_model(data)