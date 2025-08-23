import pandas as pd

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Determine the number of rows for each split
total_rows = len(df)
train_rows = int(total_rows * 0.8)  # 80% for training
test_rows = total_rows - train_rows   # 20% for testing

# Split the dataset into two parts
train_df = df.iloc[:train_rows]  # First 80%
test_df = df.iloc[train_rows:]    # Remaining 20%

# Save the datasets to CSV files
train_df.to_csv('customer_data_train.csv', index=False)
test_df.to_csv('customer_data_test.csv', index=False)