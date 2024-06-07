import streamlit as st
import pandas as pd
import numpy as np  
import math
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

st.title('SARIMA Model ')
# Function to calculate Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load data
data = pd.read_csv('data_set.csv', usecols=['Order Date', 'Total Profit'])
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.set_index('Order Date', inplace=True)
x = data.values

# Split data into train and test sets
len_train = math.ceil(data.size * 0.90)
len_test = math.floor(data.size * 0.10)
train = x[:len_train]
test = x[len_train:]
predictions = []

# Button to train and save ARIMA model
if st.button('TRAIN THE MODEL AND SAVE'):
    
   # Function to train and save ARIMA model
    def train_and_save_model(train_data):
        model_arima = auto_arima( train_data, start_p=0, start_q=0, max_p=3, max_q=0,m=12,
                                 seasonal=True,d=1, D=1 , trace=True,error_action='ignore',
                                   suppress_warnings=True, stepwise=True )
        model_arima_fit = model_arima.fit(train_data)
        joblib.dump(model_arima_fit, 'sarima_model.pkl')
        return model_arima_fit
    model_arima_fit = train_and_save_model(train)
    st.write("MODEL IS TRAINED AND SAVED")

if st.button("ACCURACY CHECK"):
    def load_model_and_forecast(model_file_path, future_periods=len(test)):
        loaded_model_fit = joblib.load(model_file_path)
        future_forecast = loaded_model_fit.predict(n_periods=future_periods)
        return future_forecast
    model_file_path = 'Sarima_model.pkl'  # Path where the model is saved

    predictions = load_model_and_forecast(model_file_path)

    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape = mean_absolute_percentage_error(test, predictions)
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    last_date = data.index[-1]
    plt.figure(figsize=(12,6))
    plt.plot(data.index[-len(test):],test, label='Actual test')
    plt.plot(data.index[-len(predictions):],predictions,label='predicted')
    plt.title("ARIMA Model : predicted values graph")
    plt.xlabel("Time")
    plt.ylabel("Total Profit")
    plt.grid(True)
    plt.legend()
    plt.show()
    st.pyplot(plt)

# Button to load model and forecast future
st.header("PREDICT FUTURE VALUES")

future_periods = st.number_input('Please give number of days',min_value=1,max_value=300)
if st.button('SUBMIT'):
    last_date = data.index[-1]
   # Function to load ARIMA model and make future predictions
    def load_model_and_forecast(model_file_path, future_periods):
        loaded_model_fit = joblib.load(model_file_path)
        future_forecast = loaded_model_fit.predict(n_periods=future_periods)
        return future_forecast
    model_file_path = 'sarima_model.pkl'  # Path where the model is saved
    future_forecast = load_model_and_forecast(model_file_path , future_periods)
    last_date = data.index[-1]

    future_dates = pd.date_range(start=last_date, periods=len(future_forecast)+1)[1:]
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(test):], test, label='Actual', color='blue')
    plt.plot(future_dates, future_forecast, label='Future Predictions', color='red')
    plt.title("Actual vs Forecasted Values")
    plt.xlabel("Time")
    plt.ylabel("Total Profit")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
