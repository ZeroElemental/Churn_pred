import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of passengers to generate
num_passengers = 800

# Create new DataFrame for the generated dataset
generated_data = pd.DataFrame()

# Generate random data based on original dataset structure
generated_data['PassengerId'] = range(1, num_passengers + 1)
generated_data['Pclass'] = np.random.choice([1, 2, 3], size=num_passengers)
generated_data['Sex'] = np.random.choice(['male', 'female'], size=num_passengers)
generated_data['Age'] = np.random.randint(0, 80, size=num_passengers)  # Random ages between 0 and 79
generated_data['SibSp'] = np.random.randint(0, 5, size=num_passengers)  # Random siblings/spouses aboard
generated_data['Parch'] = np.random.randint(0, 3, size=num_passengers)  # Random parents/children aboard
generated_data['Fare'] = np.round(np.random.uniform(7.25, 512.33, size=num_passengers), 2)  # Random fare between min and max

# Calculate combined score for cabin assignment (higher is better)
generated_data['CombinedScore'] = generated_data['Pclass'] + generated_data['Fare']

# Assign Cabin No. based on CombinedScore (higher score gets lower cabin number)
generated_data.sort_values(by='CombinedScore', ascending=False, inplace=True)
generated_data['CabinNo'] = range(1, num_passengers + 1)

# Assign minimum distance to lifeboat randomly between 0 and 1000
generated_data['MinDistanceToLifeboat'] = np.random.randint(0, 1001, size=num_passengers)

# Age classification for survival calculation
def age_group(row):
    if row['Age'] <= 20:
        return 'child'
    elif row['Age'] <= 50:
        return 'adult'
    else:
        return 'old'

# Apply age group classification
generated_data['AgeGroup'] = generated_data.apply(age_group, axis=1)

# Determine survival based on age group and gender with more realistic logic
def calculate_survival(row):
    age_group = row['AgeGroup']
    sex = row['Sex']
    
    # Survival probabilities based on age group and gender
    if age_group == 'child':  # Children have a survival chance of 100%
        return 1  
    elif age_group == 'adult':  
        if sex == 'male':
            return np.random.choice([0, 1], p=[0.5, 0.5])  # Male adults have ~50% survival chance
        else:
            return np.random.choice([0, 1], p=[0.1, 0.9])   # Female adults have ~90% survival chance
    else:  # Old
        if sex == 'male':
            return np.random.choice([0, 1], p=[0.95, 0.05]) # Male old have ~5% survival chance
        else:
            return np.random.choice([0, 1], p=[0.80, 0.20]) # Female old have ~15% survival chance

# Apply survival calculation
generated_data['Survived'] = generated_data.apply(calculate_survival, axis=1)

# Assign Embarked port randomly (C=Cherbourg, Q=Queenstown, S=Southampton)
generated_data['Embarked'] = np.random.choice(['C', 'Q', 'S'], size=num_passengers)

# Drop the CombinedScore column as it's no longer needed
generated_data.drop(columns=['CombinedScore'], inplace=True)

# Display the first few rows of the generated dataset
print(generated_data.head())

# Save the generated dataset to CSV file
generated_data.to_csv('Titanic_Dataset.csv', index=False)