import os
from pathlib import Path
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from typing import Self


@dataclass
class EvalConfig:
    layer: int
    coeff: float
    min_coeff: float
    max_coeff: float
    increment: float
    max_new_tokens: int
    num_return_sequences: int
    top_p: float
    do_sample: bool


@dataclass
class DataConfig:
    task: str # Task name
    n_train: int # Training size; Use all samples if None
    n_val: int # Validation size
    max_new_tokens: int = 15
    num_return_sequences: int = 5 # Per example
    top_p: float = 0.8
    prob_threshold: float = 0.1


@dataclass
class Config(YAMLWizard):
    model_name: str
    data_cfg: DataConfig
    method: str # Vector extraction method
    use_offset: bool # Offset by neutral examples
    filter_layer_pct: float = 0.2 # Filter the last 20% layers
    steering_test_size: int = 300
    save_dir: str = None
    use_cache: bool = True
    batch_size: int = 32
    generation_batch_size: int = 8
    seed: int = 4278

    def __post_init__(self):
        self.model_alias = os.path.basename(self.model_name)
        if self.save_dir is None:
            self.save_dir = f"runs_{self.data_cfg.task}"
    
    def artifact_path(self) -> Path:
        return Path().absolute() / self.save_dir / self.model_alias
    
    def baseline_artifact_path(self) -> Path:
        return Path().absolute() / self.save_dir / self.model_alias

    def save(self):
        os.makedirs(self.artifact_path(), exist_ok=True)
        self.to_yaml_file(self.artifact_path() / 'config.yaml')

    def load(filepath: str) -> Self:
        try:
            return Config.from_yaml_file(filepath)
        
        except FileNotFoundError:
            return None

