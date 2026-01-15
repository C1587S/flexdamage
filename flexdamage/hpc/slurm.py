
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional
from ..config.schema import RunConfig

logger = logging.getLogger(__name__)

class SlurmSubmitter:
    """
    Handles Slurm job submission for FlexDamage pipelines.
    """
    def __init__(self, config: RunConfig, config_path: str):
        self.config = config
        self.config_path = str(Path(config_path).absolute())
        self.output_dir = Path(config.run.output_dir)
        self.slurm_dir = self.output_dir / "slurm"
        self.slurm_dir.mkdir(parents=True, exist_ok=True)
        
    def submit(self, dry_run: bool = False) -> Optional[str]:
        """
        Generate submission script and submit to Slurm.
        Returns job ID if successful.
        """
        exec_conf = self.config.execution
        
        job_name = f"flexdamage_{self.config.run.name}"
        script_path = self.slurm_dir / f"{job_name}.sh"
        log_out = self.slurm_dir / f"{job_name}_%j.out"
        log_err = self.slurm_dir / f"{job_name}_%j.err"
        
        # Command to run the estimation
        # We assume we are in the project root or can access the script
        # Best to use absolute path to script if possible, or python module
        # Let's assume 'scripts/run_estimation.py' relative to CWD 
        # OR better: run as module if installed?
        # For this dev setup, let's look for the script relative to this file
        # This file is in flexdamage/hpc/slurm.py
        # root is ../../
        
        root_dir = Path(__file__).parent.parent.parent
        script_loc = root_dir / "scripts" / "run_estimation.py"
        
        if not script_loc.exists():
            # Fallback to assuming running from CWD
            script_loc = Path("scripts/run_estimation.py")
            
        cmd = f"python {script_loc} {self.config_path}"
        
        # Generate script content
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={log_out}",
            f"#SBATCH --error={log_err}",
            f"#SBATCH --time={exec_conf.slurm_time}",
            f"#SBATCH --mem={exec_conf.slurm_mem}",
            f"#SBATCH --cpus-per-task={exec_conf.slurm_cpus_per_task}",
            f"#SBATCH --partition={exec_conf.slurm_partition}",
        ]
        
        if exec_conf.slurm_account:
            lines.append(f"#SBATCH --account={exec_conf.slurm_account}")
            
        for k, v in exec_conf.slurm_extra_args.items():
            lines.append(f"#SBATCH --{k}={v}")
            
        lines.append("")
        lines.append("echo 'Starting FlexDamage Job'")
        lines.append("date")
        lines.append("echo 'Host: ' $HOSTNAME")
        lines.append("")
        
        # Determine python environment?
        # Typically we just run python and expect the user to have activated the environment before sbatch
        # Or we can inspect sys.executable
        import sys
        python_exe = sys.executable
        lines.append(f"{python_exe} {script_loc} {self.config_path} --mode local") # Force local in the job
        
        lines.append("date")
        lines.append("echo 'Job Complete'")
        
        with open(script_path, 'w') as f:
            f.write("\n".join(lines))
            
        logger.info(f"Generated Slurm script: {script_path}")
        
        if dry_run:
            logger.info("Dry run: skipping submission.")
            return "DRY_RUN"
            
        # Submit
        try:
            res = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
            output = res.stdout.strip()
            # sbatch output format: "Submitted batch job 123456"
            logger.info(f"Submission output: {output}")
            if "Submitted batch job" in output:
                return output.split()[-1]
            return output
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("sbatch command not found. Are you on a Slurm cluster?")
            raise

