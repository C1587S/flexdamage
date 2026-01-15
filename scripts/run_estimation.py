import argparse
import sys
from pathlib import Path
import logging

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from flexdamage.config.loader import load_config
from flexdamage.core.pipeline import EstimationPipeline
from flexdamage.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run FlexDamage Estimation Pipeline")
    parser.add_argument("config", type=str, help="Path to configuration YAML") # Positional is better/easier
    parser.add_argument("--test", action="store_true", help="Override config to run in test mode")
    parser.add_argument("--mode", type=str, choices=["local", "slurm"], help="Override execution mode")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for Slurm submission")
    
    args = parser.parse_args()
    
    # Load Config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
        
    # Override overrides
    if args.test:
        config.execution.test_mode = True
    if args.mode:
        config.execution.mode = args.mode
        
    # Check Mode
    logger = logging.getLogger(__name__)
    
    if config.execution.mode == "slurm":
        # Check if we are already inside a job? 
        # Usually user is running this on login node.
        # We need to SUBMIT.
        
        # Setup pre-logging
        setup_logging(level="INFO") # Console only for submission init
        logger.info("Running in SLURM SUBMISSION mode")
        
        from flexdamage.hpc.slurm import SlurmSubmitter
        submitter = SlurmSubmitter(config, args.config)
        
        try:
            job_id = submitter.submit(dry_run=args.dry_run)
            logger.info(f"Submitting job... Job ID: {job_id}")
        except Exception as e:
            logger.exception("Slurm Submission Failed")
            sys.exit(1)
            
    else:
        # Local Mode (Execution)
        # Setup Logging
        log_file = Path(config.run.output_dir) / "run.log"
        # Ensure dir exists before log setup
        Path(config.run.output_dir).mkdir(parents=True, exist_ok=True)
        
        setup_logging(level="INFO", log_file=log_file)
        
        # Run Pipeline
        try:
            pipeline = EstimationPipeline(config)
            pipeline.run()
        except Exception as e:
            logging.exception("Pipeline Execution Failed")
            sys.exit(1)

if __name__ == "__main__":
    main()
