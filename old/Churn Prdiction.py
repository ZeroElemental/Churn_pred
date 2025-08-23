import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify

data = pd.read_csv('customer_data.csv')
print(data.head())

missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype in ['int64', 'float64']:
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)

duplicates = data.duplicated().sum()
if duplicates > 0:
    data.drop_duplicates(inplace=True)

data['customer_lifetime_value'] = data['total charge'] - data['total refunds']
# # Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
numerical_features = ['age', 'Tenure in months', 'avg monthly usage', 'avg monthly download(GB)', 'total charge', 'total refunds', 'customer_lifetime_value']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
print("Preprocessed Data:\n", data.head())

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='customer status', hue='customer status', legend=False, palette='viridis')
plt.title('Distribution of Customer Status (Stayed vs Churned)')
plt.xlabel('Customer Status')
plt.ylabel('Count')
plt.show()

# Select only numeric columns for correlation
numeric_columns = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_columns].corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='customer status', y='Tenure in months', data=data, palette='Set2')
plt.title('Tenure in Months vs Customer Status')
plt.xlabel('Customer Status')
plt.ylabel('Tenure in Months')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='customer status', y='avg monthly usage', data=data, palette='Set2')
plt.title('Average Monthly Usage vs Customer Status')
plt.xlabel('Customer Status')
plt.ylabel('Average Monthly Usage')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='customer status', y='total charge', data=data, palette='Set2')
plt.title('Total Charge vs Customer Status')
plt.xlabel('Customer Status')
plt.ylabel('Total Charge')
plt.show()

# features = ['Tenure in months', 'avg monthly usage', 'avg monthly download(GB)', 'total charge', 'total refunds'] + list(data.columns[data.columns.str.contains("gender|plan type|payment method")])
# target = 'customer status'
# X = data[features]
# y = data[target].map({'Stayed': 0, 'Churned': 1})
# Check and correct column names
# Select numeric columns
numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

# Select binary categorical columns (assuming they've been one-hot encoded)
binary_features = data.select_dtypes(include=[bool]).columns.tolist()

# Combine features
features = numeric_features + binary_features

# Remove the target variable if it's in the features list
target = 'customer status'
if target in features:
    features.remove(target)

X = data[features]
y = data[target].map({'Stayed': 0, 'Churned': 1})
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

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

joblib.dump(optimized_rf_model, 'optimized_rf_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data, index=[0])
    prediction = optimized_rf_model.predict(input_data)
    output = 'stayed' if prediction[0] == 0 else 'churned'
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)

importances = optimized_rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances from Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

print("Recommendations based on analysis:")
print("- Focus on increasing average monthly usage to reduce churn.")
print("- Consider customer feedback on total charge and refunds to improve satisfaction.")
print("- Tailor retention strategies based on plan types with higher churn rates.")
