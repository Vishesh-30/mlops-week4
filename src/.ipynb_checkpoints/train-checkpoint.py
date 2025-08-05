import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI to local MLflow server
mlflow.set_tracking_uri("http://34.44.100.201:5000")


def train(data_path):
    print("Working directory:", os.getcwd())

    # Load the dataset
    df = pd.read_csv(data_path)
    print("Columns:", df.columns)

    # Split features and target
    X = df.drop(columns=["species"])
    y = df["species"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", clf.n_estimators)
        mlflow.log_param("random_state", clf.random_state)

        # Log accuracy
        acc = clf.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)

        # Save and log model
        joblib.dump(clf, "model.joblib")
        joblib.dump(le, "label_encoder.joblib")
        joblib.dump((X_test, y_test), "test_data.joblib")

        mlflow.sklearn.log_model(clf, "model")
        mlflow.log_artifact("model.joblib")
        mlflow.log_artifact("label_encoder.joblib")
        mlflow.log_artifact("test_data.joblib")

        print(f"Model trained and logged with accuracy: {acc:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_csv>")
        sys.exit(1)
    train(sys.argv[1])
