import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    name: str = "flexdamage",
    level: str = "INFO",
    log_file: Optional[Path] = None
):
    """
    Configure logging with standard format.
    """
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
        
    logging.getLogger("flexdamage").info(f"Logging initialized (Level: {level})")
