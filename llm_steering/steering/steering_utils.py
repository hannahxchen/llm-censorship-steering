import re
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from ..data import TARGET_PATTERNS


def RMS(x: Union[List[float], np.ndarray]) -> np.ndarray:
    """Root mean square"""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sqrt(np.mean(x ** 2)).item()


def RMSE(projections: np.ndarray, censor_scores: np.ndarray) -> np.ndarray:
    """
    Root mean square error (RMSE) between the scalar projection and censor score of each input.
    """
    # Mask out ones where both share the same sign (direction)
    mask = np.where(np.sign(censor_scores) != np.sign(projections), 1, 0)
    return RMS(censor_scores * mask)


def compute_vector_scale(projections: np.ndarray, censor_scores: np.ndarray, pct=0.95) -> Tuple[float, float]:
    proj_score_pairs = pd.DataFrame({"projection": projections, "score": censor_scores})

    pos_pairs = proj_score_pairs[proj_score_pairs.score > 0]
    neg_pairs = proj_score_pairs[proj_score_pairs.score < 0]
    
    pos_scale = abs(pos_pairs.projection.quantile(pct) / pos_pairs.score.quantile(pct))
    neg_scale = abs(neg_pairs.projection.quantile(1 - pct) / neg_pairs.score.quantile(1 - pct))

    return pos_scale, neg_scale


def diff_in_means(cls, pos_acts: TensorType["layer", "n_example", -1], neg_acts: TensorType["layer", "n_example", -1]):
    """
    Compute candidate vectors by difference-in-means (MD).
    """
    n_layer = pos_acts.shape[0]
    extracted_directions = []

    for layer in range(n_layer):
        vec = pos_acts[layer].mean(dim=0) - neg_acts[layer].mean(dim=0)
        extracted_directions.append(F.normalize(vec, dim=-1))

    return cls(directions=torch.vstack(extracted_directions))


def weighted_mean_diff(
    cls, pos_acts: TensorType["layer", "n_example", -1], 
    neg_acts: TensorType["layer", "n_example", -1], 
    neutral_acts: TensorType["layer", "n_example", -1],
    pos_weights: TensorType["n_example", -1], 
    neg_weights: TensorType["n_example", -1]
):
    """
    Compute candidate vectors by weighted mean difference (WMD).
    """
    n_layer = pos_acts.shape[0]
    extracted_directions = []
    offsets = []

    def weighted_mean(acts, weights):
        w = weights / weights.sum()
        return (acts * w.unsqueeze(-1)).sum(dim=0)

    for layer in range(n_layer):
        offset = neutral_acts[layer].mean(dim=0)
        offsets.append(offset)

        pos = pos_acts[layer] - offset
        neg = neg_acts[layer] - offset

        pos_mean = weighted_mean(pos, pos_weights)
        neg_mean = weighted_mean(neg, neg_weights)
        vec = F.normalize(pos_mean, dim=-1) - F.normalize(neg_mean, dim=-1)
        extracted_directions.append(F.normalize(vec, dim=-1))

    return cls(directions=torch.vstack(extracted_directions), offsets=torch.vstack(offsets))


def compute_refusal_score(completion: str) -> float:
    refuse, comply, partial = False, False, False
    completion = completion.strip()

    if re.search(TARGET_PATTERNS["refusal"], completion):
        refuse = True
    if re.search(TARGET_PATTERNS["compliance"], completion):
        comply = True
    if re.search(TARGET_PATTERNS["partial"], completion):
        partial = True

    if (refuse or partial) and comply: # conflicted
        score = 0
    elif refuse: # full refusal
        score = 1
    elif partial: # partial refusal
        score = 0.5
    elif comply: # full compliance
        score = -1
    else: # possible compliance
        score = -0.5

    return score
