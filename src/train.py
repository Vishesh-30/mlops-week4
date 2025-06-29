import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def train():
    # Load data
    print(os.getcwd())
    df = pd.read_csv("iris.csv")

    # Optional: Inspect if target column is named differently
    print("Columns:", df.columns)

    # Split features and target
    X = df.drop(columns=["species"])
    y = df["species"]

    # Encode labels if they're not numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Save model and label encoder
    joblib.dump(clf, "model.joblib")
    joblib.dump(le, "label_encoder.joblib")
    
    # Also save test data for evaluation
    joblib.dump((X_test, y_test), "test_data.joblib")

if __name__ == "__main__":
    train()    
