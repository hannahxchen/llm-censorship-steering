import json
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
from ..config import DataConfig

DATA_DIR = Path(__file__).resolve().parent
EVAL_DATASETS = [
    "jailbreakbench", "sorrybench", "xstest_safe", "xstest_unsafe", "alpaca_test_sampled", 
    "ccp_sensitive", "ccp_sensitive_sampled", "deccp_censored"
]

def load_dataframe_from_json(filepath):
    data = json.load(open(filepath, "r"))
    return pd.DataFrame.from_records(data)


def load_datasplits(data_cfg: DataConfig, save_dir: Path, use_cache: bool = False) -> Dict[str, pd.DataFrame]:
    datasets = {}        
    for split in ["train", "val"]:
        if use_cache and Path(save_dir / f"{split}.json").exists():
            logging.info(f"Loading cached data from {save_dir}/{split}.json")
            datasets[split] = load_dataframe_from_json(save_dir / f"{split}.json")
        else:
            datasets[split] = load_dataframe_from_json(DATA_DIR / "datasplits" / f"{data_cfg.task}_{split}.json")

            sample_size = getattr(data_cfg, f"n_{split}")
            if sample_size > 0:
                datasets[split] = datasets[split].sample(n=min(len(datasets[split]), sample_size))
    
    return datasets


def load_eval_dataset(dataset_name):
    assert dataset_name in EVAL_DATASETS
    file_path = DATA_DIR / 'processed' / f"{dataset_name}.json"

    with open(file_path, 'r') as f:
        dataset = json.load(f)
 
    return dataset

