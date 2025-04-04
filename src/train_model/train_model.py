import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

def load_config(config_path="config.yaml"):
    # Load the configuration file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_model(config_path="config.yaml"):
    # Load configuration parameters
    config = load_config(config_path)
    year = config["data_extraction"]["year"]

    # Define paths for processed data and model storage
    data_path = f"data/processed/prepared_accidents_{year}.csv"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Load the prepared data
    data = pd.read_csv(data_path, low_memory=False)

    # Select relevant features (example: columns that seem relevant for predicting severity)
    # These features can be adjusted based on domain knowledge or feature importance analysis
    features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", "plan", "surf", "situ", "lum", "atm", "col"]
    target = "grav"  # Binary target column (0: grave, 1: not grave)

    # Ensure all selected features exist in the dataset
    available_features = [col for col in features if col in data.columns]
    if not available_features:
        raise ValueError("None of the selected features are available in the dataset.")

    # Prepare features (X) and target (y)
    X = pd.get_dummies(data[available_features], drop_first=True)  # Convert categorical variables to dummy variables
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    model_path = f"{model_dir}/rf_model_{year}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
