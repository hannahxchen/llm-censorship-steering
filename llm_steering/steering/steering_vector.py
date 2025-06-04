import os, json
from pathlib import Path
from dataclasses import dataclass
from typing import Self, List, Dict, Callable
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from ..utils import save_to_json_file
from .steering_utils import *


@dataclass
class SteeringVector:
    directions: TensorType["layer", -1]
    offsets: TensorType["layer", -1] = None
    pos_scales: List[float] = None
    neg_scales: List[float] = None

    def __post_init__(self):
        if self.offsets is None:
            self.offset_func = lambda x, layer: x
        else:
            self.offset_func = lambda x, layer: x - self.offsets[layer]

    @classmethod
    def load(cls, save_dir: Path) -> Self:
        """
        Load a SteeringVector instance from attributes stored in a directory.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        pos_scales = None
        neg_scales = None
        offsets = None

        directions = torch.load(save_dir / "directions.pt", weights_only=True)
        if Path(save_dir / "vector_scales.json").exists():
            scales = json.load(open(save_dir / "vector_scales.json", "r"))
            pos_scales = scales["pos"]
            neg_scales = scales["neg"]

        if Path(save_dir / "offsets.pt").exists():
            offsets = torch.load(save_dir / "offsets.pt", weights_only=True)
            
        return cls(directions=directions, offsets=offsets, 
                   pos_scales=pos_scales, neg_scales=neg_scales)


    def save(self, save_dir: Path):
        """
        Save the instance's attributes to a directory.
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.directions, save_dir / "directions.pt")
        if self.offsets is not None:
            torch.save(self.offsets, save_dir / "offsets.pt")

        if self.pos_scales is not None and self.neg_scales is not None:
            save_to_json_file({"pos": self.pos_scales, "neg": self.neg_scales}, save_dir / "vector_scales.json", silent=True)

    def set_dtype(self, dtype=torch.bfloat16):
        self.directions = self.directions.to(dtype)
        if self.offsets is not None:
            self.offsets = self.offsets.to(dtype)

    @classmethod
    def fit(cls, method: str, pos_acts: TensorType["layer", "n_example", -1], neg_acts: TensorType["layer", "n_example", -1], **kwargs) -> Self:
        """
        Compute candidate vectors from model activations and return a SteeringVector instance.
        """
        if method == "MD":
            return diff_in_means(cls, pos_acts, neg_acts)

        elif method == "WMD":
            return weighted_mean_diff(cls, pos_acts, neg_acts, **kwargs)
        else:
            raise ValueError(f"Unknown method: '{method}'")

    def validate(self, val_acts: TensorType["layer", "n_example", -1], censor_scores: np.ndarray, save_dir: Path = None) -> Dict:
        """
        Validate candidate vectors based on projection correlation and RMSE (linear separability).
        """
        n_layer = val_acts.shape[0]
        results, projections = [], []
        self.pos_scales, self.neg_scales = [], []
        self.set_dtype(torch.float64)

        for layer in range(n_layer):
            acts = val_acts[layer]
            projs = self.scalar_projection(acts, layer).numpy()

            r = pearsonr(projs, censor_scores)
            rmse = RMSE(projs, censor_scores)

            projections.append(projs.tolist())
            results.append({
                "layer": layer, 
                "corr": r.statistic, 
                "p_val": r.pvalue,
                "RMSE": rmse
            })

            pos, neg = compute_vector_scale(projs, censor_scores)
            self.pos_scales.append(pos)
            self.neg_scales.append(neg)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_dir / "val_projections.npy", np.array(projections))
            save_to_json_file(results, save_dir / "val_results.json")
        return results
    
    @staticmethod
    def get_top_layer_id(results: Dict = None, save_dir: Path = None, filter_layer_pct: float = 0.2, save_top_layers: bool = True) -> int:
        if results is None and save_dir is None:
            raise ValueError("Either of the arguments is required: results, save_dir")
        
        if results is None:
            results = json.load(open(Path(save_dir / "val_results.json"), "r"))

        n_layer = len(results)
        max_layer = round(n_layer * (1 - filter_layer_pct)) - 1

        filtered_results = [x for x in results if x["layer"] < max_layer] # Filter layers close to the last layer
        top_layer_results = sorted(filtered_results, key=lambda x: (x["corr"]-x["RMSE"]), reverse=True) # Sort layers by RMSE & correlation

        if save_top_layers and save_dir is not None:
            save_to_json_file(top_layer_results, Path(save_dir / "top_layers.json"))

        return top_layer_results[0]["layer"]
    
    def get_scaled_vec(self, layer: int, coeff: float = 0) -> TensorType[-1]:
        if coeff > 0:
            return self.directions[layer] * self.pos_scales[layer] * coeff
        else:
            return self.directions[layer] * self.neg_scales[layer] * coeff
    
    def orthogonal_projection(self, acts: TensorType[..., -1], layer: int):
        """
        Compute vector projections on the candidate vector at a given layer.
        """
        unit_vec = self.directions[layer]
        return self.offset_func(acts, layer) @ unit_vec.unsqueeze(-1) * unit_vec
    
    def scalar_projection(self, acts: TensorType[..., -1], layer: int):
        """
        Compute scalar projections on the candidate vector at a given layer.
        """
        acts = self.offset_func(acts, layer)
        cosin_sim = F.cosine_similarity(acts, self.directions[layer], dim=-1)
        projs = acts.norm(dim=-1) * cosin_sim
        return projs.to(torch.float64)
    
    def steering_func(self, coeff: float = 0, reposition: bool = True) -> Callable:
        """
        Return an intervention function for steering.
        """
        if reposition:
            return lambda x, layer: x - self.orthogonal_projection(x, layer) + self.get_scaled_vec(layer, coeff)
        else:
            return lambda x, layer: x + self.get_scaled_vec(layer, coeff)
