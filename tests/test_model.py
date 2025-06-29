import os
import joblib
import pytest
import numpy as np

def test_model_file_exists():
    assert os.path.exists("model.joblib"), "model.joblib not found."

def test_model_prediction_shape():
    model = joblib.load("model.joblib")
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # One sample from Iris-like input
    prediction = model.predict(sample)
    assert prediction.shape == (1,), "Prediction shape mismatch."

def test_model_prediction_type():
    model = joblib.load("model.joblib")
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)
    assert isinstance(prediction[0], (int, np.integer)), "Prediction is not an integer class label."
