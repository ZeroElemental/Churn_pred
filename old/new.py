import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the customer data from the uploaded CSV file
data = pd.read_csv('customer_data.csv')

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values: fill numerical with median and categorical with mode
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype in ['int64', 'float64']:
            data[column] = data[column].fillna(data[column].median())
        else:
            data[column] = data[column].fillna(data[column].mode()[0])

# Remove duplicates if any
duplicates = data.duplicated().sum()
if duplicates > 0:
    data.drop_duplicates(inplace=True)

# Encode categorical variables (gender and subscription types)
categorical_columns = ['gender', 'multi_screen', 'mail_subscribed']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Select numerical features for scaling
numerical_features = ['age',
                      'no_of_days_subscribed',
                      'weekly_mins_watched',
                      'minimum_daily_mins',
                      'maximum_daily_mins',
                      'weekly_max_night_mins',
                      'videos_watched',
                      'maximum_days_inactive',
                      'customer_support_calls']

# Standardize numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Prepare features and target variable
target = 'churn'
y = data[target]

# Define features by excluding the target variable and non-feature columns
feature_columns = [col for col in data.columns if col not in [target, 'year', 'customer_id', 'phone_no']]
X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_features": ["auto", "sqrt"],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           scoring="accuracy", cv=5)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("\nBest Parameters from Grid Search:", best_params)
print("Best Cross-Validation Score:", best_score)

optimized_rf_model = RandomForestClassifier(**best_params)
optimized_rf_model.fit(X_train, y_train)
optimized_y_pred = optimized_rf_model.predict(X_test)

print("\nOptimized Confusion Matrix:\n", confusion_matrix(y_test, optimized_y_pred))
print("\nOptimized Classification Report:\n", classification_report(y_test, optimized_y_pred))
print("Optimized Accuracy Score:", accuracy_score(y_test, optimized_y_pred))

# Save the optimized model to a file
joblib.dump(optimized_rf_model, 'optimized_rf_model.pkl')

# Sample Data for Testing Predictions (you can modify this as needed)
sample_data = pd.DataFrame({
    "age": [30],
    "no_of_days_subscribed": [150],
    "weekly_mins_watched": [200],
    "minimum_daily_mins": [10],
    "maximum_daily_mins": [30],
    "weekly_max_night_mins": [60],
    "videos_watched": [5],
    "maximum_days_inactive": [2],
    "customer_support_calls": [1],
    "gender_Male": [0],   # Female
    "multi_screen_yes": [1], # Yes
    "mail_subscribed_yes": [1] # Yes
})

# Preprocess sample data similarly to training data (scaling and encoding)
sample_data[numerical_features] = scaler.transform(sample_data[numerical_features])

# Align input features with model training features
for col in feature_columns:
    if col not in sample_data.columns:
        sample_data[col] = np.zeros(sample_data.shape[0]) # Fill missing columns with zero

prediction = optimized_rf_model.predict(sample_data[feature_columns])
output = "stayed" if prediction[0] == 0 else "churned"
print(f"Sample Prediction: The customer is likely to {output}.")