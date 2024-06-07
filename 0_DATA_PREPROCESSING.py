import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
df = pd.read_csv('1000000 Sales Records.csv')

st.title('DATA PREPROCESSING')

#preprocessing for models 
if st.button("START PROCESSING"):
    date_column = 'Order Date'
    def parse_date(date_str):
        # Try different formats based on your data
        if '/' in date_str:
            return pd.to_datetime(date_str, format='%m/%d/%Y')
        elif '-' in date_str:
            return pd.to_datetime(date_str, format='%d-%m-%Y')
        else:
        # Handle other formats or errors (e.g., coerce, fill with NaNs)
            return pd.to_datetime(date_str, errors='coerce')  # Or other error handling
    
    df[date_column] = df[date_column].apply(parse_date)
    # convert string into date.
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    #select specific values like (order date and total revenue).
    actual_df = df[['Order Date','Total Profit']]
    #sort the values acconding order using Order Date values.
    actual_df.sort_values(by='Order Date', inplace=True)

    given_date = pd.to_datetime('2017-01-01')  # Example date, replace with your desired date

# Filter the DataFrame
    filtered_df = actual_df[actual_df['Order Date'] < given_date]

    sum_revenue_per_date = filtered_df.groupby('Order Date')['Total Profit'].sum()
    sum_revenue_per_date.to_csv('data_set.csv')
    st.write("DATA PREPROCESSing AND SAVE IT INTO CSV FILE SUCESSFULLY")


if st.button("VIEW DATA"):
    data = pd.read_csv('data_set.csv', usecols=['Order Date', 'Total Profit'])
    st.write("PREPROCESS DATA")
    st.write(data)
    st.write("PREPROCESS DATA GRAPH")
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data.set_index('Order Date', inplace=True)
    plt.figure(figsize=(26,10))
    plt.plot(data , label='Actual data')
    plt.title("Data graph = Time vs Total Profit")
    plt.xlabel("time")
    plt.ylabel("Total Profit")
    plt.grid(True)
    plt.legend()
    plt.show()
    st.pyplot(plt)