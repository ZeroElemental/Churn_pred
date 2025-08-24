Customer-Churn-Prediction

Telecom Customer Churn Predictor

Overview

This project implements a machine learning model to predict customer churn for a telecom company. It includes data generation, preprocessing, model training, and evaluation components.

Table of Contents

Project Structure

Installation

Usage

Data Generation

Data Preprocessing

Model Training

Evaluation

Contributing

License

Project Structure
telecom-churn-predictor/
│
├── data/
│   └── telecom_customer_churn_data.csv
│
├── src/
│   ├── generate_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/
│   └── churn_model.joblib
│
├── requirements.txt
├── README.md
└── .gitignore

Installation

Clone this repository:

git clone https://github.com/yourusername/telecom-churn-predictor.git
cd telecom-churn-predictor


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install the required packages:

pip install -r requirements.txt

Usage

To run the entire pipeline:

python src/generate_data.py
python src/preprocess_data.py
python src/train_model.py
python src/evaluate_model.py

Data Generation

The generate_data.py script creates synthetic telecom customer data. It generates 1500–2000 records with features like customer demographics, usage patterns, and churn status.

To generate data:

python src/generate_data.py --output data/telecom_customer_churn_data.csv

Data Preprocessing

The preprocess_data.py script handles:

Missing value imputation

Encoding categorical variables

Feature scaling

To preprocess the data:

python src/preprocess_data.py --input data/telecom_customer_churn_data.csv --output data/processed_data.csv

Model Training

The train_model.py script trains a Logistic Regression model to predict customer churn. It uses the preprocessed data to fit the model and saves it for later use.

To train the model:

python src/train_model.py --input data/processed_data.csv --output models/churn_model.joblib

Evaluation

The evaluate_model.py script assesses the model's performance using metrics such as:

Accuracy

Precision

Recall

F1-score

To evaluate the model:

python src/evaluate_model.py --model models/churn_model.joblib --test-data data/processed_data.csv

Contributing

Contributions to this project are welcome! Please follow these steps:

Fork the repository

Create a new branch (git checkout -b feature/AmazingFeature)

Make your changes

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

License

This project is licensed under the MIT License - see the LICENSE
 file for details.
