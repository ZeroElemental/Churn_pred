import pandas as pd
import numpy as np
from faker import Faker
import random

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()

# Function to generate synthetic data
def generate_telecom_data(num_records):
    data = {
        'customer_name': [fake.name() for _ in range(num_records)],
        'customer_id': [fake.unique.random_number(digits=8) for _ in range(num_records)],
        'gender': np.random.choice(['Male', 'Female'], num_records),
        'age': np.random.randint(18, 80, num_records),
        'zipcode': [fake.zipcode() for _ in range(num_records)],
        'city': [fake.city() for _ in range(num_records)],
        'Tenure in months': np.random.randint(1, 72, num_records),
        'phone no.': [fake.phone_number() for _ in range(num_records)],
        'avg monthly usage': np.random.randint(50, 2000, num_records),
        'plan type': np.random.choice(['Basic', 'Standard', 'Premium'], num_records),
        'avg monthly download(GB)': np.random.randint(5, 200, num_records),
        'payment method': np.random.choice(['Credit Card', 'Bank Transfer', 'PayPal'], num_records),
        'total charge': np.random.uniform(20, 200, num_records).round(2),
        'total refunds': np.random.uniform(0, 50, num_records).round(2),
        'customer status': np.random.choice(['Stayed', 'Churned'], num_records, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # Add churn category and reason only for churned customers
    churn_categories = ['Attitude', 'Competitor', 'Dissatisfaction', 'Other', 'Price']
    churn_reasons = {
        'Attitude': ['Attitude of support person', 'Attitude of service provider'],
        'Competitor': ['Competitor offered better device', 'Competitor offered more data', 'Competitor had better coverage'],
        'Dissatisfaction': ['Network reliability', 'Service dissatisfaction', 'Product dissatisfaction'],
        'Other': ['Moved', 'Deceased', 'Unknown'],
        'Price': ['Extra data charges', 'Long distance charges', 'Price too high']
    }
    
    df['churn category'] = ''
    df['churn reason'] = ''
    
    churned = df['customer status'] == 'Churned'
    df.loc[churned, 'churn category'] = np.random.choice(churn_categories, churned.sum())
    
    for category in churn_categories:
        mask = (churned) & (df['churn category'] == category)
        df.loc[mask, 'churn reason'] = np.random.choice(churn_reasons[category], mask.sum())
    
    return df

# Generate data
num_records = np.random.randint(1500, 2001)  # Random number between 1500 and 2000
df = generate_telecom_data(num_records)

# Save to CSV
df.to_csv('C:/Users/shreyash/Downloads/Churn/telecom_customer_churn_data.csv', index=False)
print(f"Generated {num_records} records. Data saved to 'telecom_customer_churn_data.csv'")

# Display first few rows and data info
print(df.head())
print(df.info())
