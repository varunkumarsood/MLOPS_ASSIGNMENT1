import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load Iris dataset
file_path = "data/iris_data.csv"
iris= pd.read_csv(file_path, header=None)

# Split features (X) and target (y)
X = iris.iloc[:, :-1]  # All columns except the last one
y = iris.iloc[:, -1]   # Last column as the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVM": SVC(C=1.0, kernel='linear')
}

# Run experiments and log with MLflow
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Log parameters, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

print("Experiments logged successfully.")
