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
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML")
    parser.add_argument("--test", action="store_true", help="Override config to run in test mode")
    
    args = parser.parse_args()
    
    # Load Config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
        
    # Override test mode if CLI flag set
    if args.test:
        config.execution.test_mode = True
        
    # Setup Logging
    log_file = Path(config.run.output_dir) / "run.log"
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
