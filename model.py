#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Step 1: Download the Iris Dataset
def download_data():
    # Create directory for data if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # URL of the Iris dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # File path to save the dataset
    file_path = "data/iris.data"

    # Download the dataset
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Dataset downloaded and saved at: {file_path}")
    else:
        print("Failed to download the dataset!")

# Step 2: Load the dataset into a pandas DataFrame
def load_data():
    # Define column names as the dataset doesn't contain headers
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # Load dataset into a pandas DataFrame
    iris_data = pd.read_csv("data/iris.data", header=None, names=columns)
    return iris_data

# Step 3: Train a Random Forest Model
def train_model(data):
    # Prepare features (X) and target (y)
    X = data.iloc[:, :-1]  # Features (sepal_length, sepal_width, etc.)
    y = data.iloc[:, -1]   # Target variable (class)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the trained model
    dump(model, "model.joblib")
    print("Model saved as 'model.joblib'")

# Step 4: Load and test the saved model (Optional)
def test_saved_model():
    model = load("model.joblib")
    print("Model loaded successfully!")
    return model

# Main function to run all steps
if __name__ == "__main__":
    download_data()          # Download dataset
    iris_data = load_data()  # Load dataset into DataFrame
    train_model(iris_data)   # Train model and save it
    test_saved_model()       # (Optional) Load and test saved model

