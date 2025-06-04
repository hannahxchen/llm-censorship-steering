import os
from pathlib import Path
from dataclasses import dataclass
from dataclass_wizard import YAMLWizard
from typing import Self, List, Union


@dataclass
class SteeringConfig:
    layer_ids: Union[int|List[int]] # Layer ids to intervene
    coeff: float # Steering coefficient
    min_coeff: float
    max_coeff: float
    increment: float
    max_new_tokens: int
    num_return_sequences: int
    top_p: float
    do_sample: bool
    temperature: float


@dataclass
class Config(YAMLWizard):
    model_name: str
    censor_type: str # Target censorship type
    n_train: int # Training size; Use all samples if None
    n_val: int # Validation size
    method: str # Vector extraction method
    threshold: float # Threshold score for labeling
    filter_layer_pct: float # Filter last N% layers
    save_dir: str
    seed: int

    def __post_init__(self):
        self.model_alias = os.path.basename(self.model_name)
        if self.save_dir is None:
            self.save_dir = f"runs/{self.censor_type}/{self.model_alias}"
    
    def artifact_path(self) -> Path:
        return Path().absolute() / self.save_dir

    def save(self):
        os.makedirs(self.artifact_path(), exist_ok=True)
        self.to_yaml_file(self.artifact_path() / 'config.yaml')
    
    def load(filepath: str) -> Self:
        try:
            return Config.from_yaml_file(filepath)
        
        except FileNotFoundError:
            return None
