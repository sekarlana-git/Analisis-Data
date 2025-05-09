import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('bank_transactions_data_2.csv')
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
    df['TransactionMonth'] = df['TransactionDate'].dt.month
    df['TransactionDay'] = df['TransactionDate'].dt.day_name()
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+']
    df['AgeGroup'] = pd.cut(df['CustomerAge'], bins=bins, labels=labels)
    # Convert AgeGroup to string
    df['AgeGroup'] = df['AgeGroup'].astype(str)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header('Filters')
transaction_type = st.sidebar.multiselect(
    'Transaction Type',
    options=df['TransactionType'].unique(),
    default=df['TransactionType'].unique()
)

channel = st.sidebar.multiselect(
    'Channel',
    options=df['Channel'].unique(),
    default=df['Channel'].unique()
)

# Get unique age groups and handle NaN values
age_groups = df['AgeGroup'].dropna().unique()
age_group = st.sidebar.multiselect(
    'Age Group',
    options=age_groups,
    default=age_groups  # Set all age groups as default
)

# Filter data
filtered_df = df[
    (df['TransactionType'].isin(transaction_type)) &
    (df['Channel'].isin(channel)) &
    (df['AgeGroup'].isin(age_group))
]

# Dashboard title
st.title('Bank Transactions Analysis Dashboard')

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(filtered_df))
col2.metric("Average Amount", f"${filtered_df['TransactionAmount'].mean():.2f}")
col3.metric("Unique Customers", filtered_df['AccountID'].nunique())

# Visualization 1
st.subheader('Transaction Distribution by Channel and Type')
fig1, ax1 = plt.subplots(figsize=(10,4))
sns.countplot(data=filtered_df, x='Channel', hue='TransactionType', ax=ax1)
st.pyplot(fig1)

# Visualization 2
st.subheader('Transaction Amount by Age Group and Occupation')
fig2, ax2 = plt.subplots(figsize=(12,6))
sns.barplot(
    data=filtered_df, 
    x='AgeGroup', 
    y='TransactionAmount', 
    hue='CustomerOccupation', 
    estimator=np.mean, 
    ci=None,
    ax=ax2
)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig2)

# RFM Analysis
st.subheader('RFM Customer Segmentation')
current_date = filtered_df['TransactionDate'].max() + pd.Timedelta(days=1)
rfm = filtered_df.groupby('AccountID').agg({
    'TransactionDate': lambda x: (current_date - x.max()).days,
    'TransactionID': 'count',
    'TransactionAmount': 'sum'
}).reset_index()
rfm.columns = ['AccountID', 'Recency', 'Frequency', 'Monetary']

fig3, ax3 = plt.subplots(1, 3, figsize=(15,5))
sns.histplot(rfm['Recency'], bins=20, ax=ax3[0])
ax3[0].set_title('Recency Distribution')
sns.histplot(rfm['Frequency'], bins=20, ax=ax3[1])
ax3[1].set_title('Frequency Distribution')
sns.histplot(rfm['Monetary'], bins=20, ax=ax3[2])
ax3[2].set_title('Monetary Distribution')
st.pyplot(fig3)