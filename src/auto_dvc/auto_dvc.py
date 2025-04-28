import os
import subprocess
import yaml
from pathlib import Path

# Run a shell command and handle errors
def run_cmd(cmd, desc=""):
    print(f"[CMD] {desc} → {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {desc}:\n{result.stderr.strip()}")
        raise RuntimeError(f"Failed: {desc}")
    else:
        print(result.stdout.strip())
        return result.stdout.strip()

# Check if the file is an output in dvc.yaml
def is_pipeline_output(file_path):
    if not Path("dvc.yaml").exists():
        return False
    with open("dvc.yaml") as f:
        dvc_yaml = yaml.safe_load(f)
    for stage in dvc_yaml.get("stages", {}).values():
        if "outs" in stage and file_path in stage["outs"]:
            return True
    return False

def main():
    # 1. Load configuration file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        dagshub_cfg = config.get("dagshub", {})
        token = dagshub_cfg.get("token", "")

    # 2. Configure DVC remote credentials
    run_cmd(["dvc", "remote", "modify", "origin", "--local", "access_key_id", token], "Set access_key_id")
    run_cmd(["dvc", "remote", "modify", "origin", "--local", "secret_access_key", token], "Set secret_access_key")

    # 3. Reproduce the pipeline to update outputs
    run_cmd(["dvc", "repro"], "Reproduce the pipeline")

    # 4. Check and add/update files with DVC
    data_files = [
        "data/raw/accidents_2023.csv",
        "data/processed/prepared_accidents_2023.csv",
        "models/rf_model_2023.joblib",
        "models/best_model_2023.joblib"
    ]
    for file in data_files:
        dvc_file = f"{file}.dvc"
        if is_pipeline_output(file):
            print(f"{file} is already a pipeline output, skipping dvc add.")
            # DVC commit if needed (to register a manual modification)
            run_cmd(["dvc", "commit", "-f", file], f"DVC commit for {file}")
        else:
            if not Path(dvc_file).exists():
                run_cmd(["dvc", "add", file], f"Add {file} to DVC")
                run_cmd(["git", "add", dvc_file], f"Git add {dvc_file}")
            else:
                run_cmd(["dvc", "commit", "-f", file], f"DVC commit for {file}")

    # 5. Check for modified files
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    changed_files = [line.strip().split()[-1] for line in result.stdout.strip().splitlines()]
    print(f"Modified files: {changed_files}")

    # 6. Git add + commit + push if changes are detected
    relevant_files = ["dvc.lock", "dvc.yaml"]
    if any(f in changed_files for f in relevant_files):
        run_cmd(["git", "add", ".gitignore", "dvc.yaml", "dvc.lock"], "Git add DVC files")
        run_cmd(["git", "commit", "-m", "Auto: pipeline and outputs update"], "Git commit")
        run_cmd(["git", "push"], "Git push")
    else:
        print("No Git commit needed.")

    # 7. Push data to DVC remote
    run_cmd(["dvc", "push"], "Final DVC push")

    print("All pipeline data and models are versioned and synchronized!")

if __name__ == "__main__":
    main()
