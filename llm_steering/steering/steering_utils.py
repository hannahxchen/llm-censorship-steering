from typing import List, Optional, Dict
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from ..data.prompt_iterator import PromptIterator


def get_all_layer_activations(
    model, prompts: List[str], batch_size: Optional[int] = 32, positions=[-1]
) -> TensorType["n_layer", "n_prompt", "hidden_size"]:
    acts_all = []
    layers = list(range(model.n_layer))
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    if prompt_iterator.pbar is not None:
        prompt_iterator.pbar.set_description("Extracting activations")

    for prompt_batch in prompt_iterator:
        acts = model.get_activations(layers, prompt_batch, positions=positions).squeeze(-2)
        acts_all.append(acts)

    return torch.concat(acts_all, dim=1)


def scalar_projection(acts: TensorType[..., -1], steering_vec: TensorType[-1]):
    cosin_sim = F.cosine_similarity(acts, steering_vec, dim=-1)
    projs = acts.norm(dim=-1) * cosin_sim
    return projs.to(torch.float64)


def compute_projections(model, steering_vec: TensorType, layer: int, prompts: List[str], offset=0, batch_size=32):
    activations = []
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size, desc="Extracting activations for computing projections")
    for prompt_batch in prompt_iterator:
        acts = model.get_activations(layer, prompt_batch, positions=[-1]).squeeze()
        activations.append(acts)
    activations = torch.vstack(activations)
    return scalar_projection(activations - offset, steering_vec)

