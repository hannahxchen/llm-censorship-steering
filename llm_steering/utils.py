import logging, json, gc
from pathlib import Path
from typing import List, Dict, Iterator
from functools import singledispatch
import numpy as np
import torch
import torch.nn.functional as F


def RMS(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sqrt(np.mean(x ** 2)).item()


def RMSE(proj_ratios: np.ndarray, bias_scores: np.ndarray):
    mask = np.where(np.sign(bias_scores) != np.sign(proj_ratios), 1, 0)
    return RMS(bias_scores * mask)


def pairwise_cosine_similarity(x):
    return F.cosine_similarity(x.unsqueeze(0), x.unsqueeze(1), dim=2)


def ceildiv(a, b):
    return -(a // -b)


def chunks(lst, n) -> Iterator[List]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


def save_to_json_file(results: List[Dict], filepath: Path, silent: bool = False):
    with open(filepath, "w") as f:
        json.dump(results, f, default=to_serializable, indent=4)

    if not silent:
        logging.info(f"Results saved to: {filepath}")


def loop_coeffs(min_coeff=-1, max_coeff=1, increment=0.1):
    coeffs = []
    n = int((max_coeff - min_coeff)/increment) + 1
    coeffs = np.array(range(n)) * increment + min_coeff
    coeffs = np.round(coeffs, 2)

    return coeffs.tolist()

def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()