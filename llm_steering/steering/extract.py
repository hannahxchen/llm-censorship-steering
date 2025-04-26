import os, logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

from . import ModelBase
from ..config import Config
from .steering_utils import get_all_layer_activations


def weighted_mean(acts, weights):
    """Weighted Mean"""
    w = weights / weights.sum()
    return (acts * w.unsqueeze(-1)).sum(dim=0)


def get_activations(model: ModelBase, examples: pd.DataFrame, save_dir: Path, label: str, batch_size: int, use_cache=False):
    activation_path = save_dir / f"{label}.pt"
    if use_cache and activation_path.exists():
        acts = torch.load(activation_path)
    else:
        prompts = model.apply_chat_template(examples["prompt"].tolist())
        acts = get_all_layer_activations(model, prompts, batch_size).to(torch.float64)
        torch.save(acts, activation_path)
    return acts


def extract_candidate_vectors(
    cfg: Config, 
    model: ModelBase, 
    pos_examples: pd.DataFrame, 
    neg_examples: pd.DataFrame, 
    neutral_examples: pd.DataFrame,
):  
    save_dir = cfg.artifact_path() / "activations"
    os.makedirs(save_dir, exist_ok=True)

    pos_acts = get_activations(model, pos_examples, save_dir, label="positive", batch_size=cfg.batch_size, use_cache=cfg.use_cache)
    neg_acts = get_activations(model, neg_examples, save_dir, label="negative", batch_size=cfg.batch_size, use_cache=cfg.use_cache)
    neutral_acts = get_activations(model, neutral_examples, save_dir, label="neutral", batch_size=cfg.batch_size, use_cache=cfg.use_cache)

    pos_weights = torch.Tensor(pos_examples.prob_diff.tolist())
    neg_weights = torch.Tensor(neg_examples.prob_diff.tolist())

    extracted_vectors, offsets = [], []
        
    for layer in range(model.n_layer):
        pos = pos_acts[layer]
        neg = neg_acts[layer]

        offset = neutral_acts[layer].mean(dim=0)
        offsets.append(offset)
        pos -= offset
        neg -= offset

        pos_mean = weighted_mean(pos, pos_weights)
        neg_mean = weighted_mean(neg, neg_weights)
        vec = F.normalize(pos_mean, dim=-1) - F.normalize(neg_mean, dim=-1)

        extracted_vectors.append(vec)

    os.makedirs(cfg.artifact_path() / "activations", exist_ok=True)
    filepath = cfg.artifact_path() / "activations" / "candidate_vectors.pt"

    torch.save(torch.vstack(extracted_vectors), filepath)
    torch.save(torch.vstack(offsets), cfg.artifact_path() / "activations" / "offsets.pt")

    logging.info(f"Candidate vectors saved to: {filepath}")

    