import streamlit as st

st.title("LSTM MODEL")

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Load the data

# Set random seed for PyTorch
torch.manual_seed(42)

# Set random seed for NumPy
np.random.seed(42)
data = pd.read_csv("data_set.csv")

# Convert date column to datetime and set it as index
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.set_index('Order Date', inplace=True)

# Extract target variable
target = data['Total Profit'].values.astype(float)

# Normalize the target variable
scaler = MinMaxScaler(feature_range=(-1, 1))
target_normalized = scaler.fit_transform(target.reshape(-1, 1))

# Define sequence length
sequence_length = 100

# Create sequences of data
sequences = []
for i in range(len(target_normalized) - sequence_length):
    sequences.append(target_normalized[i:i+sequence_length+1])

# Convert sequences to numpy array
sequences = np.array(sequences)

# Split into input and output
X = sequences[:, :-1]
y = sequences[:, -1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=42)


# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Convert data to PyTorch tensors and move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Reshape input data
input_size = X_train.shape[-1]  # Number of features
X_train = X_train.view(-1, sequence_length, input_size)
X_test = X_test.view(-1, sequence_length, input_size)

# Extract dates for plotting
dates = data.index[-len(y_test):]

# Define hyperparameters
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 2000
learning_rate = 0.001

if st.button("TRAIN THE MODEL"):

    # Initialize the model and move it to GPU if available
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

    # Inverse transform the actual values (y_test)
    actual_profit = scaler.inverse_transform(y_test.cpu().numpy())
    predicted_profit = scaler.inverse_transform(test_outputs.cpu().numpy())
    torch.save(model.state_dict(), 'lstm_model.pht')
    file_path = "lstm_array_data.csv"
    np.savetxt(file_path, predicted_profit, delimiter=",")
    st.write("MODEL IS TRAINED AND SAVED")



loaded_model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
loaded_model.load_state_dict(torch.load('lstm_model.pht', map_location=torch.device('cpu')))
loaded_model.eval()  # Set the model to evaluation mode


if st.button("ACCURACY CHECK"):


    file_path = "lstm_array_data.csv"
    predicted_profit = np.loadtxt(file_path, delimiter=",")

        # Inverse transform the actual values (y_test)
    actual_profit = scaler.inverse_transform(y_test.cpu().numpy())

        # Inverse transform the future predictions
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape = mean_absolute_percentage_error(actual_profit, predicted_profit)
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Plot actual vs predicted values with dates on the x-axis
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual_profit, label='Actual')
    plt.plot(dates, predicted_profit, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.title('Actual vs Predicted Profit')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.grid(True)
    plt.show()
    st.pyplot(plt)
    
st.header('PREDICT FUTURE VALUES')


# Extend the test data for future prediction

# Forecast using the loaded model
future_days = st.number_input('Please give number of days',min_value=1,max_value=300)

if st.button('SUBMIT'):
    extended_sequence = []
    last_sequence = X_test[-1]
    for _ in range(future_days):
        with torch.no_grad():
            output = loaded_model(last_sequence.unsqueeze(0))
            extended_sequence.append(output.squeeze().item())
            last_sequence = torch.cat((last_sequence[1:], output), axis=0)

    # Inverse transform the actual values (y_test)
    actual_profit = scaler.inverse_transform(y_test.cpu().numpy())

    # Inverse transform the future predictions
    future_predicted_profit = scaler.inverse_transform(np.array(extended_sequence).reshape(-1, 1))

    # Generate dates for the future predictions
    last_date = dates[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

    # Plot actual, predicted (test + future forecast), and train values
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual_profit, label='Actual', color='blue')
    plt.plot(future_dates, future_predicted_profit, label='Forecast', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.title('Actual and Forecasted Profit')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)