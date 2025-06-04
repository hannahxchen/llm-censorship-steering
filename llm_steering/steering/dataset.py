import os, logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Callable
import pandas as pd
import torch
import torch.nn.functional as F

from ..data import load_datasplit
from ..utils import PromptIterator, save_to_json_file
from .model import ModelBase
from .steering_utils import compute_refusal_score


@dataclass
class Dataset:    
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    threshold: float
    use_next_token: bool

    @staticmethod
    def load(censor_type: str, n_train: int = None, n_val: int = None, threshold: float = 0, cached_dir: Path = None):
        if censor_type == "refusal":
            dataset_cls = RefusalDataset
        elif censor_type == "thought_suppress":
            dataset_cls = ThoughtSuppressDataset
        else:
            raise ValueError(f"Unknown censor_type: '{censor_type}'")
        
        train_data = load_datasplit(censor_type, split="train", sample_size=n_train, cached_dir=cached_dir)
        val_data = load_datasplit(censor_type, split="val", sample_size=n_val, cached_dir=cached_dir)
        
        return dataset_cls(train_data=train_data, val_data=val_data, threshold=threshold)

    def save(self, save_dir: Path):
        os.makedirs(save_dir, exist_ok=True)
        save_to_json_file(self.train_data.to_dict("records"), save_dir / "train.json")
        save_to_json_file(self.val_data.to_dict("records"), save_dir / "val.json")

    def pos_examples(self) -> pd.DataFrame:
        return self.train_data[self.train_data.censor_score > self.threshold]
    
    def neg_examples(self) -> pd.DataFrame:
        return self.train_data[self.train_data.censor_score < self.threshold]
    
    def neutral_examples(self) -> pd.DataFrame:
        return self.train_data[self.train_data.censor_score.abs() <= self.threshold]
    
    def val_examples(self) -> pd.DataFrame:
        return self.val_data
    
    @staticmethod
    def get_formatted_prompts(data: pd.DataFrame, chat_template_func: Callable) -> List[str]:
        return chat_template_func(data.prompt.tolist())
    
    def compute_censor_score(self, model: ModelBase, prompts: List[str], batch_size: int, **kwargs):
        pass
    
    def compute_baseline_scores(self, model: ModelBase, batch_size: int, use_cache: bool = False, **kwargs):
        """
        Compute baseline censor scores for train and validation data.
        """
        logging.info("Preprocessing train/val data")
        for split in ["train", "val"]:
            datasplit = self.__dict__[f"{split}_data"]
            if use_cache is True and ("pos_prob" in datasplit.columns):
                continue

            formatted_prompts = self.get_formatted_prompts(datasplit, model.apply_chat_template)
            pos_probs, neg_probs = self.compute_censor_score(model, formatted_prompts, batch_size, **kwargs)

            datasplit["pos_prob"] = pos_probs
            datasplit["neg_prob"] = neg_probs
            datasplit["censor_score"] = datasplit["pos_prob"] - datasplit["neg_prob"]

            self.__setattr__(f"{split}_data", datasplit)
    

@dataclass
class RefusalDataset(Dataset):
    use_next_token: bool = False

    def _get_aggregated_probs(self, transition_scores, all_completions: List[str], n_prompts: int, num_return_sequences: int) -> Tuple[List[float], List[float]]:
        n = num_return_sequences
        pos_target_probs, neg_target_probs = [], []

        for batch_id in range(n_prompts):
            completions = all_completions[n * batch_id: n * (batch_id + 1)]
            scores = transition_scores[n * batch_id: n * (batch_id + 1)]
            token_probs = torch.prod(torch.exp(scores), dim=-1)

            pos_probs, neg_probs = [], []

            for c, prob in zip(completions, token_probs):
                score = compute_refusal_score(c)

                if score > 0:
                    pos_probs.append(prob * abs(score))
                elif score < 0:
                    neg_probs.append(prob * abs(score))
            
            pos_prob = torch.sum(torch.stack(pos_probs)).item() if len(pos_probs) > 0 else 0
            neg_prob = torch.sum(torch.stack(neg_probs)).item() if len(neg_probs) > 0 else 0
            pos_target_probs.append(pos_prob)
            neg_target_probs.append(neg_prob)

        return pos_target_probs, neg_target_probs
    

    def compute_censor_score(self, model: ModelBase, prompts: List[str], batch_size: int, max_new_tokens=15, num_return_sequences=5, top_p=0.8):
        pos_probs_all, neg_probs_all = [], []
        prompt_iterator = PromptIterator(prompts, batch_size=batch_size)

        for prompt_batch in prompt_iterator:
            inputs = model.tokenize(prompt_batch).to(model.device)
            input_len = inputs.input_ids.shape[1]

            outputs = model.model._model.generate(
                **inputs, num_beams=num_return_sequences,  
                num_return_sequences=num_return_sequences, 
                max_new_tokens=max_new_tokens, top_p=top_p,
                return_dict_in_generate=True, output_scores=True,
            )
            transition_scores = model.model._model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
            )
            
            all_completions = model.tokenizer.batch_decode(outputs.sequences[:, input_len:])
            
            pos_probs, neg_probs = self._get_aggregated_probs(
                transition_scores, all_completions, 
                n_prompts=len(prompt_batch), 
                num_return_sequences=num_return_sequences
            )

            pos_probs_all.extend(pos_probs)
            neg_probs_all.extend(neg_probs)

        return pos_probs_all, neg_probs_all


@dataclass
class ThoughtSuppressDataset(Dataset):
    use_next_token: bool = True
    pos_token: str = "\n\n"
    neg_token: str = "\n"
    
    def compute_censor_score(self, model, prompts: List[str], batch_size: int):
        pos_token_id = model.tokenizer.encode(self.pos_token, add_special_tokens=False)[0]
        neg_token_id = model.tokenizer.encode(self.neg_token, add_special_tokens=False)[0]

        pos_probs_all, neg_probs_all = [], []
        prompt_iterator = PromptIterator(prompts, batch_size=batch_size)

        for prompt_batch in prompt_iterator:
            logits = model.get_last_position_logits(prompt_batch)
            probs = F.softmax(logits, dim=-1)

            pos_probs = probs[:, pos_token_id]
            neg_probs = probs[:, neg_token_id]

            pos_probs_all.extend(pos_probs)
            neg_probs_all.extend(neg_probs)


        return pos_probs_all, neg_probs_all
            