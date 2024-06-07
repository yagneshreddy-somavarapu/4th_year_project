import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
df = pd.read_csv('1000000 Sales Records.csv')

st.title('PROFIT FORECASTING USING TIME SERIES FORECASTING TECHNIQUES')

st.sidebar.title('Exploratory Data Analysis')


region = st.sidebar.selectbox(options= df.Region.unique(),label= 'Select Region')
region_df = df[df['Region']==region]

country = st.sidebar.selectbox(options=region_df.Country.unique(),label= 'Select Country')
country_df = region_df[region_df['Country']==country]

Item = st.sidebar.selectbox(options=country_df['Item Type'].unique(),label= 'Select Item')
item_df=country_df[country_df['Item Type']==Item]

#result_df = df[(df['Region']==region) & (df['Country']==country) & (df['Country']==Item)]

def parse_date(date_str):
    # Try different formats based on your data
    if '/' in date_str:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    elif '-' in date_str:
        return pd.to_datetime(date_str, format='%d-%m-%Y')
    else:
        # Handle other formats or errors (e.g., coerce, fill with NaNs)
        return pd.to_datetime(date_str, errors='coerce')  # Or other error handling

item_df['Order Date']= item_df['Order Date'].apply(parse_date)
item_df['Order Date'] = pd.to_datetime(item_df['Order Date']).dt.date
item_df = item_df.sort_values(by='Order Date')


if item_df is  not None:
    st.dataframe(item_df)

st.dataframe(item_df.describe().T)

if st.sidebar.button('Show Missing Data'):
    # Generate a sequence of dates spanning the range of your data
    min_date = item_df['Order Date'].min()
    max_date = item_df['Order Date'].max()
    date_range = pd.date_range(start=min_date, end=max_date)

    # Find missing dates
    missing_dates = date_range[~date_range.isin(item_df['Order Date'])]

    # Count missing dates
    missing_dates_count = len(missing_dates)

    st.write("Number of missing dates:", missing_dates_count)
    st.write("Missing dates:")
    st.write(missing_dates)


feature = st.sidebar.selectbox(options=item_df.columns,label='Select Feature')
st.line_chart(x ='Order Date',y= feature,data=item_df,width = 1000, height=600)
