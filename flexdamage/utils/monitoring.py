import psutil
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """
    Tracks memory and CPU usage.
    """
    @staticmethod
    def get_usage() -> Dict[str, float]:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "memory_rss_gb": mem_info.rss / (1024**3),
            "cpu_percent": process.cpu_percent()
        }
        
    @staticmethod
    def log_usage(context: str = ""):
        usage = ResourceMonitor.get_usage()
        logger.info(
            f"Resources [{context}]: "
            f"Memory={usage['memory_rss_gb']:.2f} GB | "
            f"CPU={usage['cpu_percent']:.1f}%"
        )
