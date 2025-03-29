import os
from pathlib import Path
from typing import List, Dict, Iterator, Callable

import torch
import torch.nn.functional as F
from torchtyping import TensorType
import numpy as np
from ..steering.model import ModelBase
from ..utils import save_to_json_file, chunks
from ..config import EvalConfig
from ..steering.steering_utils import scalar_projection
from ..steering.intervention import get_intervention_func
from ..data.prompt_iterator import PromptIterator
from .task import Task, load_task

deepseek_generation_config = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "max_new_tokens": 2048}


def parse_reasoning_output(completion):
    if "</think>" in completion:
        p = [p for p in completion.partition("</think>") if p != ""]
        reasoning = "".join(p[:-1])
        if len(p) == 1:
            answer = None
        else:
            answer = p[-1]
    else:
        answer = None
        reasoning = completion

    return reasoning, answer



class Evaluator():
    def __init__(
        self, eval_cfg: EvalConfig, 
        model: ModelBase, steering_vec: TensorType, 
        save_dir: Path, 
        batch_size: int = 32, offset=0
    ):
        self.cfg = eval_cfg
        self.model = model
        self.steering_vec = model.set_dtype(steering_vec)
        self.offset = model.set_dtype(offset)
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def load_task(self, task_name):
        return load_task(task_name=task_name)
    
    def generate_completions(self, prompts: List[str], layer: int = None, intervene_func: Callable = None) -> Iterator[List[str]]:
        prompt_iterator = PromptIterator(prompts, batch_size=self.batch_size, desc=f"Generating completions")
        all_completions = []

        for prompt_batch in prompt_iterator:
            completions = self.model.generate(
                prompt_batch, layer, intervene_func=intervene_func, 
                max_new_tokens=self.cfg.max_new_tokens, do_sample=self.cfg.do_sample, 
                num_return_sequences=self.cfg.num_return_sequences, top_p=self.cfg.top_p
            )
            all_completions.extend(completions)

        return chunks(all_completions, self.cfg.num_return_sequences)
    
    def run_reasoning(self, prompts: List[str], layer: int = None, intervene_func: Callable = None) -> List[str]:
        prompt_iterator = PromptIterator(prompts, batch_size=self.batch_size, desc=f"Generating completions")
        all_completions = []

        for prompt_batch in prompt_iterator:
            completions = self.model.generate(
                prompt_batch, layer, intervene_func=intervene_func, 
                max_new_tokens=deepseek_generation_config['max_new_tokens'], do_sample=True, 
                num_return_sequences=1, top_p=deepseek_generation_config['max_new_tokens'],
                temperature=deepseek_generation_config['temperature']
            )
            all_completions.extend(completions)

        return all_completions
    
    def get_next_token_probs(self, prompts: List[str], token_ids: List[int], layer: int = None, intervene_func: Callable = None):
        prompt_iterator = PromptIterator(prompts, batch_size=self.batch_size, desc="Getting answer probabilty")
        
        token_probs = None
        for prompt_batch in prompt_iterator:
            inputs = self.model.tokenize(prompt_batch)
            logits = self.model.get_last_position_logits(inputs, layer=layer, intervene_func=intervene_func)
            probs = F.softmax(logits, dim=-1)

            if token_probs is None:
                token_probs = probs[:, token_ids].numpy()
            else:
                token_probs = np.concatenate((token_probs, probs[:, token_ids].numpy()), axis=0)
        
        return token_probs
    
    def compute_projections(self, task: Task, layer: int, use_cache: bool = False) -> np.ndarray:
        filepath = self.save_dir / f"{task.task_name}_projections.npy"
        if use_cache and Path(filepath).exists():
            return

        data = task.prepare_inputs(self.model.apply_chat_template)
        prompts = [x["formatted_prompt"] for x in data]

        activations = []
        prompt_iterator = PromptIterator(prompts, batch_size=self.batch_size, desc="Extracting activations for computing projections")
        for prompt_batch in prompt_iterator:
            acts = self.model.get_activations(layer, prompt_batch, positions=[-1]).squeeze()
            activations.append(acts)
        activations = torch.vstack(activations)
        projs = scalar_projection(activations - self.offset, self.steering_vec).tolist()
        
        np.save(filepath, np.array(projs))
        print(f"Projections saved to: {filepath}")
    
    def save_completions(self, data: List[Dict], outputs: Iterator[List[str]], filepath: str):
        results = []
        for x, output in zip(data, outputs):
            results.append({
                "_id": x["_id"],
                "prompt": x["prompt"],
                "completions": output
            })

        save_to_json_file(results, filepath)

    def save_outputs(self, data: List[Dict], outputs, filepath: str):
        results = []
        for x, token_probs in zip(data, outputs):
            results.append({
                "_id": x["_id"],
                "prompt": x["prompt"],
                "token_probs": token_probs.tolist()
            })

        save_to_json_file(results, filepath)

    def save_reasoning(self, data: List[Dict], outputs: Iterator[List[str]], filepath: str):
        results = []
        for x, output in zip(data, outputs):
            reasoning, answer = parse_reasoning_output(output)
            results.append({
                "_id": x["_id"],
                "prompt": x["prompt"],
                "reasoning": reasoning,
                "answer": answer
            })

        save_to_json_file(results, filepath)
      
    def run(
        self, task: Task, coeff: float = 0, k: float = 0, baseline: bool = False, use_cache: bool = False, 
        get_next_token_prob: bool = False, is_reasoning: bool = False, 
    ):

        if baseline:
            intervene_func = None
            save_filepath = self.save_dir / f"{task.task_name}_baseline.json"
        else:
            intervene_func = get_intervention_func(self.steering_vec, offset=self.offset, coeff=coeff, k=k)
            save_filepath = self.save_dir / f"{task.task_name}_steering_outputs_coeff={coeff}.json"

        if use_cache and Path(save_filepath).exists():
            return
        
        data = task.prepare_inputs(self.model.apply_chat_template)
        prompts = [x["formatted_prompt"] for x in data]

        if is_reasoning:
            steering_outputs = self.run_reasoning(prompts, layer=self.cfg.layer, intervene_func=intervene_func)
            self.save_reasoning(data, steering_outputs, save_filepath)

        elif get_next_token_prob:
            token_ids = [self.model.tokenizer.encode(token, add_special_tokens=False)[0] for token in ['\n\n', '\n']]
            steering_outputs = self.get_next_token_probs(prompts, token_ids=token_ids, layer=self.cfg.layer, intervene_func=intervene_func)
            self.save_outputs(data, steering_outputs, save_filepath)

        else:
            steering_outputs = self.generate_completions(prompts, layer=self.cfg.layer, intervene_func=intervene_func)
            self.save_completions(data, steering_outputs, save_filepath)

            