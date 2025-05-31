import joblib

def test_model_loading():
    model = joblib.load("models/rf_model_2023.joblib")
    assert model is not None
