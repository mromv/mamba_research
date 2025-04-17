import json
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import asyncio

def load_config(dataset_name: str, configs_dir: str = "configs") -> Dict[str, Any]:
    config_path = f"{configs_dir}/{dataset_name}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def log_error(
    idx: int, 
    sample: Dict, 
    error_message: str, 
    error_log_path: str = "logs/errors.log"
) -> None:
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "idx": idx,
        "input": sample,
        "error": error_message
    }

    async with asyncio.Lock():
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


async def save_sample(
    sample: Dict[str, Any], 
    output_file: str, 
    lock: asyncio.Lock
) -> None:
    async with lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")