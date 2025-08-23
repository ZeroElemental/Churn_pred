import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('Generated_Titanic_Dataset.csv')

# Preprocess data
# Convert categorical variables to numerical values
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Define features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinNo', 'MinDistanceToLifeboat']]
y = data['Survived']

# Train Random Forest model (using 100% of the data)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Make predictions on the same dataset
y_pred = model.predict(X)

# Evaluate the model
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
class_report = classification_report(y, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(class_report)