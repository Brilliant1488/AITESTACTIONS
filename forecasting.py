import numpy as np
import pandas as pd
from keras.models import load_model

def forecast(data, model, time_step=100):
    temp_input = list(data[-time_step:])
    lst_output = []
    for i in range(30):  # Predicting for 30 days
        if len(temp_input) > time_step:
            temp_input = temp_input[1:]
        temp_input_reshaped = np.array(temp_input).reshape(1, time_step, 1)
        yhat = model.predict(temp_input_reshaped, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
    return lst_output

if __name__ == "__main__":
    data = pd.read_csv("scaled_data.csv").values
    model = load_model("lstm_model.h5")
    predictions = forecast(data, model)
    pd.DataFrame(predictions, columns=['Predicted_Close']).to_csv("predictions.csv", index=False)