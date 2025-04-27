import os
import time
import subprocess
import yaml

def main():
    # Load credentials from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        dagshub_config = config.get("dagshub", {})
        token = dagshub_config.get("token", "")
    
    # Configure DVC credentials directly
    try:
        subprocess.run(["dvc", "remote", "modify", "origin", "--local", "access_key_id", "46fb4962402bf1118d644028129e9c98e5f60ce6"], check=True)
        subprocess.run(["dvc", "remote", "modify", "origin", "--local", "secret_access_key", "46fb4962402bf1118d644028129e9c98e5f60ce6"], check=True)
        print("DVC credentials configured.")
    except subprocess.CalledProcessError as e:
        print(f"Error configuring DVC credentials: {e}")

    time.sleep(10)
    
    files_to_check = [
        "data/raw/accidents_2023.csv",
        "data/processed/prepared_accidents_2023.csv",
        "models/rf_model_2023.joblib"
    ]
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist yet!")
        else:
            print(f"File {file_path} exists!")
            try:
                subprocess.run(["dvc", "commit", "-f", file_path], check=True)
                print(f"DVC commit done for {file_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error during dvc commit for {file_path}: {e}")

    # Push global DVC
    try:
        subprocess.run(["dvc", "push"], check=True)
        print("DVC push done for all files.")
    except subprocess.CalledProcessError as e:
        print(f"Error during dvc push: {e}")

if __name__ == "__main__":
    main() 
