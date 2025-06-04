import os, logging
from pathlib import Path
from typing import List, Dict, Iterator, Callable, Union
import torch.nn.functional as F
import numpy as np

from ..steering.model import ModelBase
from ..utils import PromptIterator, save_to_json_file, chunks
from ..config import SteeringConfig
from ..steering import SteeringVector
from .eval_utils import Task


def loop_coeffs(min_coeff=-1, max_coeff=1, increment=0.1) -> List[float]:
    coeffs = []
    n = int((max_coeff - min_coeff)/increment) + 1
    coeffs = np.array(range(n)) * increment + min_coeff
    coeffs = np.round(coeffs, 2)

    return coeffs.tolist()


class Evaluator():
    def __init__(
        self, steering_cfg: SteeringConfig, save_dir: Path, use_cache: bool = False,
        batch_size: int = 32, generation_batch_size: int = 16
    ):
        self.cfg = steering_cfg
        self.batch_size = batch_size
        self.generation_batch_size = generation_batch_size
        self.use_cache = use_cache
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def generate_completions(self, model, prompts: List[str], steering_func: Callable = None) -> Iterator[List[str]]:
        prompt_iterator = PromptIterator(prompts, batch_size=self.generation_batch_size, desc=f"Generating completions")
        all_completions = []

        for prompt_batch in prompt_iterator:
            completions = model.generate(
                prompt_batch, self.cfg.layer_ids, steering_func, 
                max_new_tokens=self.cfg.max_new_tokens, do_sample=self.cfg.do_sample, 
                num_return_sequences=self.cfg.num_return_sequences, 
                top_p=self.cfg.top_p, temperature=self.cfg.temperature
            )
            all_completions.extend(completions)

        return chunks(all_completions, self.cfg.num_return_sequences)
    
    def get_next_token_probs(self, model, prompts: List[str], token_ids: List[int], layer_ids: Union[int, List[int]] = None, steering_func: Callable = None):
        prompt_iterator = PromptIterator(prompts, batch_size=self.batch_size, desc="Getting next token probabilty")
        
        token_probs = None
        for prompt_batch in prompt_iterator:
            inputs = model.tokenize(prompt_batch)
            logits = model.get_last_position_logits(inputs, layer_ids=layer_ids, steering_func=steering_func)
            probs = F.softmax(logits, dim=-1)

            if token_probs is None:
                token_probs = probs[:, token_ids].numpy()
            else:
                token_probs = np.concatenate((token_probs, probs[:, token_ids].numpy()), axis=0)
        
        return token_probs

    def save_token_probs(self, data: List[Dict], outputs, filepath: str):
        results = []
        for x, token_probs in zip(data, outputs):
            results.append({
                "_id": x["_id"],
                "prompt": x["prompt"],
                "token_probs": token_probs.tolist()
            })

        save_to_json_file(results, filepath)


    def run_baseline(self, model: ModelBase, task: Task, parse_reasoning: bool = False):
        save_filepath = self.save_dir / f"{task.task_name}_baseline.json"

        if self.use_cache and Path(save_filepath).exists():
            return
        
        prompts = task.prepare_inputs(model.apply_chat_template)
        outputs = self.generate_completions(model, prompts)
        task.save_completions(outputs, save_filepath, parse_reasoning=parse_reasoning)

      
    def run_steering(self, model: ModelBase, task: Task, steering_vec: SteeringVector, parse_reasoning: bool = False):

        if self.cfg.coeff is None:
            coeff_list = loop_coeffs(self.cfg.min_coeff, self.cfg.max_coeff, self.cfg.increment)
        else:
            coeff_list = [self.cfg.coeff]

        prompts = task.prepare_inputs(model.apply_chat_template)
        steering_vec.set_dtype(model.dtype)

        for coeff in coeff_list:
            save_filepath = self.save_dir / f"{task.task_name}_steering_coeff={coeff}.json"

            if self.use_cache and Path(save_filepath).exists():
                continue

            logging.info(f"Steering coefficient={coeff:.1f}")

            steering_func = steering_vec.steering_func(coeff)
            steering_outputs = self.generate_completions(model, prompts, steering_func=steering_func)
            task.save_completions(steering_outputs, save_filepath, parse_reasoning=parse_reasoning)
