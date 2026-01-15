
import subprocess
import yaml
import os
import shutil
from pathlib import Path
import sys

def verify_slurm():
    print("Verifying Slurm Submission (Dry Run)...")
    
    config_data = {
        "run": {
            "name": "test_slurm",
            "output_dir": "tmp_slurm"
        },
        "sector": {"name": "ag"},
        "data": {
            "dataset_dir": "test_data_slurm.csv",
            "columns": {"y": "y", "x": "x", "region": "reg", "year": "yr"}
        },
        "estimation": {
            "functional_form": {"type": "quadratic"},
            "global": {"method": "fixed_effects"},
            "regional": {"aggregation_level": "impact_region"}
        },
        "execution": {
            "mode": "slurm",
            "n_workers": 2,
            "slurm_time": "02:00:00",
            "slurm_mem": "32G",
            "slurm_account": "test_acc"
        }
    }
    
    with open("test_slurm.yaml", "w") as f:
        yaml.dump(config_data, f)
        
    print("Running command...")
    cmd = [
        "python3", "flexdamage-dev/scripts/run_estimation.py",
        "test_slurm.yaml",
        "--mode", "slurm",
        "--dry-run"
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Command output:")
        print(res.stdout)
        
        # Check generated script
        script_path = Path("tmp_slurm/slurm/flexdamage_test_slurm.sh")
        if script_path.exists():
            print(f"\nSUCCESS: Slurm script generated at {script_path}")
            content = script_path.read_text()
            print("\nScript Content:")
            print(content)
            
            if "#SBATCH --time=02:00:00" in content and "--mode local" in content and "test_acc" in content:
                print("\nSUCCESS: Script content matches configuration.")
            else:
                print("\nFAILURE: Script content missing expected flags.")
        else:
            print(f"\nFAILURE: Slurm script NOT found at {script_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"\nFAILURE: Command failed with return code {e.returncode}")
        print(e.stderr)

    # Cleanup
    if os.path.exists("test_slurm.yaml"):
        os.remove("test_slurm.yaml")
    if os.path.exists("tmp_slurm"):
        shutil.rmtree("tmp_slurm")

if __name__ == "__main__":
    verify_slurm()
