import json
import logging
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CENSOR_TYPES = ["refusal", "thought_suppress"]
EVAL_DATASETS = [
    "jailbreakbench", "sorrybench", "alpaca_test_sampled", "xstest_safe", "xstest_unsafe",
    "ccp_sensitive", "ccp_sensitive_sampled", "censorship_test", "deccp_sampled"
]


def load_dataframe_from_json(filepath):
    data = json.load(open(filepath, "r"))
    return pd.DataFrame.from_records(data)


def load_datasplit(censor_type, split="train", sample_size=-1, cached_dir: Path = None):
    assert censor_type in CENSOR_TYPES

    if cached_dir is not None and Path(cached_dir / f"{split}.json").exists():
        logging.info(f"Loading cached data from {cached_dir}/{split}.json")
        data = load_dataframe_from_json(cached_dir / f"{split}.json")

    else:
        data = load_dataframe_from_json(DATA_DIR / "datasplits" / f"{censor_type}_{split}.json")
        if sample_size > 0:
            data = data.sample(n=min(len(data), sample_size))

    return data


def load_eval_dataset(dataset_name):
    assert dataset_name in EVAL_DATASETS
    file_path = DATA_DIR / 'processed' / f"{dataset_name}.json"

    with open(file_path, 'r') as f:
        dataset = json.load(f)
 
    return dataset
