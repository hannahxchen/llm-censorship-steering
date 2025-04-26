import os
import logging
from pathlib import Path
from typing import List, Callable, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from . import ModelBase
from .steering_utils import get_all_layer_activations, scalar_projection
from ..config import Config
from ..utils import RMS, RMSE, save_to_json_file, chunks, loop_coeffs
from ..data.prompt_iterator import PromptIterator
from .intervention import get_intervention_func


def RMSE(proj_ratios: np.ndarray, bias_scores: np.ndarray):
    mask = np.where(np.sign(bias_scores) != np.sign(proj_ratios), 1, 0)
    return RMS(bias_scores * mask)


def evaluate_candidate_vectors(
    model: ModelBase, prompts: List[str],
    candidate_vectors: TensorType["n_layer", "hidden_size"], 
    prob_diffs: np.ndarray, save_dir: Path, 
    offsets = None,
    filter_layer_pct: float = 0.2, batch_size: int = 32
) -> List[Dict]:
    os.makedirs(save_dir, exist_ok=True)

    results, projections = [], []
    prompt_acts = get_all_layer_activations(model, prompts, batch_size)

    for layer in range(model.n_layer):
        vec = candidate_vectors[layer]
        acts = prompt_acts[layer]
        
        if offsets is not None:
            acts -= offsets[layer]

        projs = scalar_projection(acts, vec).numpy()

        r = pearsonr(projs, prob_diffs)
        rmse = RMSE(projs, prob_diffs)

        projections.append(projs.tolist())
        results.append({
            "layer": layer, 
            "corr": r.statistic, 
            "p_val": r.pvalue,
            "RMSE": rmse
        })

    np.save(save_dir / "projections.npy", np.array(projections))
    save_to_json_file(results, save_dir / "proj_correlation.json")
    
    max_layer = round(model.n_layer * (1 - filter_layer_pct)) - 1
    filtered_results = [x for x in results if x["layer"] < max_layer] # Filter layers close to the last layer
    top_layer_results = sorted(filtered_results, key=lambda x: (x["corr"]-x["RMSE"]), reverse=True) # Sort layers by RMSE & correlation

    return top_layer_results


def run_steering_test(
    model: ModelBase, prompts: List[str], layer:int = None, intervene_func: Callable = None, batch_size: int = 32,
    max_new_tokens: int = 50, do_sample: bool = True, num_return_sequences: int = 5, top_p: float = 0.6
):
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    all_completions = []

    for prompt_batch in prompt_iterator:
        prompt_batch = model.apply_chat_template(prompt_batch)
        completions = model.generate(
            prompt_batch, layer, intervene_func=intervene_func, 
            max_new_tokens=max_new_tokens, do_sample=do_sample, 
            num_return_sequences=num_return_sequences, top_p=top_p
        )
        all_completions.extend(completions)

    return chunks(all_completions, num_return_sequences)


def validate(
    cfg: Config, model: ModelBase, val_data: pd.DataFrame, 
    min_coeff=-1.5, max_coeff=1.5, increment=0.5,
    max_new_tokens=50, do_sample=True, num_return_sequences=5, top_p=0.8
):
    save_dir = cfg.artifact_path() / "validation"
    activation_dir = cfg.artifact_path() / "activations"
    candidate_vectors = torch.load(activation_dir / "candidate_vectors.pt")

    if cfg.use_offset:
        offsets = torch.load(activation_dir / "offsets.pt")
    else:
        offsets = None

    prompts = val_data.prompt.tolist()
    formatted_prompts = model.apply_chat_template(prompts)
    bias_baseline = val_data["prob_diff"].to_numpy()

    top_layer_results = evaluate_candidate_vectors(
        model, formatted_prompts, candidate_vectors, bias_baseline, 
        save_dir, offsets=offsets, 
        filter_layer_pct=cfg.filter_layer_pct, 
        batch_size=cfg.batch_size
    )

    logging.info(f"Top layers: {[x['layer'] for x in top_layer_results]}")
    save_to_json_file(top_layer_results, save_dir / "top_layers.json")

    # Run steering test on a few examples
    prompts = val_data.sample(n=cfg.steering_test_size).prompt.tolist()
    formatted_prompts = model.apply_chat_template(prompts)

    layer = top_layer_results[0]["layer"]
    steering_vec = candidate_vectors[layer]
    offset = offsets[layer]
    steering_vec, offset = model.set_dtype(steering_vec, offset)

    for coeff in loop_coeffs(min_coeff, max_coeff, increment):
        logging.info(f"Steering test: coeff={coeff:.1f}")
        intervene_func = get_intervention_func(steering_vec, offset=offset, coeff=coeff)

        outputs = run_steering_test(
            model, formatted_prompts, layer=layer, 
            intervene_func=intervene_func, batch_size=cfg.generation_batch_size,
            max_new_tokens=max_new_tokens, do_sample=do_sample, 
            num_return_sequences=num_return_sequences, top_p=top_p
        )

        results = []
        for p, completions in zip(prompts, outputs):
            results.append({
                "prompt": p,
                "completions": completions
            })

        save_to_json_file(results, save_dir / f"steering_outputs_coeff={coeff}.json")
        