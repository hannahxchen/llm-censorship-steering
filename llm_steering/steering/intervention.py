from typing import List, Callable
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from transformers import BatchEncoding
from nnsight import LanguageModel
from nnsight.intervention import Envoy


def apply_intervention(
    model: LanguageModel, 
    inputs: BatchEncoding, 
    layer_block: Envoy,
    intervene_func: Callable,
) -> TensorType["n_prompt", "seq_len", "vocab_size"]:

    with model.trace(inputs) as tracer:
        acts = layer_block.output[0].clone()
        new_acts = intervene_func(acts)
        layer_block.output = (new_acts,) + layer_block.output[1:]
        logits = model.lm_head.output.detach().to("cpu").to(torch.float64).save()

    return logits


def intervene_generation(
    model: LanguageModel, 
    inputs: BatchEncoding, 
    layer_blocks: List[Envoy],
    layer: int,
    intervene_func: Callable, 
    max_new_tokens: int = 10, 
    do_sample: bool = False, **kwargs
) -> TensorType["n_prompt", "seq_len"]:

    with model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
        layer_blocks.all()
        acts = layer_blocks[layer].output[0].clone()
        new_acts = intervene_func(acts)
        layer_blocks[layer].output[0][:] = new_acts
        outputs = model.generator.output.save()
        
    return outputs


def orthogonal_projection(a: TensorType[..., -1], unit_vec: TensorType[-1]) -> TensorType[..., -1]:
    return a @ unit_vec.unsqueeze(-1) * unit_vec


def custom_func(acts, unit_vec, offset, coeff, displacement=0):
    # If coeff > 0, impose censorship; If coeff < 0, remove censorship
    projs = torch.matmul(acts - offset, unit_vec)  # dot product
    return acts - coeff * (projs - displacement).unsqueeze(-1) * unit_vec


def get_intervention_func(steering_vec: TensorType, offset=0, coeff=0, k=0) -> Callable:
    """Get function for model intervention.
    Methods:
    - scaled_proj: Scale steering coefficients by projections.
    - constant: Use a constant steering coefficient.
    """
    unit_vec = F.normalize(steering_vec, dim=-1)
    rescaled_vec = unit_vec * k

    return lambda acts: acts - orthogonal_projection(acts - offset, unit_vec) + coeff * rescaled_vec