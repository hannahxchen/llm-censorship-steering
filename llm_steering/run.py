import os
import random
import argparse
import warnings
import logging
from pathlib import Path

import torch
import numpy as np
from .config import Config, DataConfig
from .utils import save_to_json_file
from .data.load_dataset import load_datasplits
from .data.prompt_iterator import PromptIterator
from .steering.compute_targets import compute_refusal_score, compute_suppression_score
from .steering import load_model, ModelBase, extract_candidate_vectors, validate

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False);
logging.basicConfig(level=logging.INFO)

REASONING_TOKENS = {"positive": "\n\n", "negative": "\n"}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Model name.')
    parser.add_argument('--config_file', type=str, default=None, help='Load configuration from file.')
    parser.add_argument('--task', type=str, required=True, choices=["censorship", "refusal"], help='Target task.')
    parser.add_argument('--n_train', type=int, default=1200, help="Number of training examples.")
    parser.add_argument('--n_val', type=int, default=600, help="Number of validation examples.")
    parser.add_argument('--steering_test_size', type=int, default=10, help="Number of examples for steering test.")
    parser.add_argument('--threshold', type=float, default=0.1, help="Threshold score for positive/negative examples.")
    parser.add_argument('--filter_layer_pct', type=float, default=0.2, help='Filter last N percentage layers.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--generation_batch_size', type=int, default=8, help='Batch size for text generation.')
    parser.add_argument('--max_new_tokens', type=int, default=15, help='Maximum number of generated tokens.')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top p value for sampling.')
    parser.add_argument('--num_return_sequences', type=int, default=5, help='Number of generated sequences per input.')
    parser.add_argument('--seed', type=int, default=4238, help='Random seed.')
    parser.add_argument('--save_dir', type=str, help='Save results to specified directory.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse stored cached results.')
    return parser.parse_args()


def train_and_validate(cfg: Config, data_cfg: DataConfig, model: ModelBase):
    datasplits_dir = cfg.artifact_path() / "datasplits"
    datasets = load_datasplits(data_cfg, datasplits_dir, use_cache=cfg.use_cache)
    os.makedirs(datasplits_dir, exist_ok=True)

    logging.info("Preprocessing train/val data")

    if data_cfg.task == "censorship":
        pos_token_id = model.tokenizer.encode(REASONING_TOKENS["positive"], add_special_tokens=False)[0]
        neg_token_id = model.tokenizer.encode(REASONING_TOKENS["negative"], add_special_tokens=False)[0]

    for split in ["train", "val"]:
        df = datasets[split].copy()

        if cfg.use_cache is True and ("pos_prob" in df.columns):
            continue

        logging.info(f"Getting baseline results for {split} split")

        pos_probs_all, neg_probs_all = [], []
        prompt_iterator = PromptIterator(df["prompt"].tolist(), batch_size=cfg.generation_batch_size)

        for prompt_batch in prompt_iterator:
            formatted_prompts = model.apply_chat_template(prompt_batch)

            if data_cfg.task == "refusal":
                pos_probs, neg_probs = compute_refusal_score(
                    model, formatted_prompts, max_new_tokens=data_cfg.max_new_tokens, 
                    num_return_sequences=data_cfg.num_return_sequences, top_p=data_cfg.top_p
                )
            else:
                pos_probs, neg_probs = compute_suppression_score(model, formatted_prompts, pos_token_id, neg_token_id)
                
            pos_probs_all.extend(pos_probs)
            neg_probs_all.extend(neg_probs)

        df["pos_prob"] = pos_probs_all
        df["neg_prob"] = neg_probs_all
        df["prob_diff"] = df["pos_prob"] - df["neg_prob"]
            
        datasets[split] = df
        save_to_json_file(df.to_dict("records"), datasplits_dir / f"{split}.json")

    if not cfg.use_cache or not Path(cfg.artifact_path() / "activations/candidate_vectors.pt").is_file():
        train_data = datasets["train"]
        pos_examples = train_data[train_data.prob_diff > data_cfg.threshold]
        neg_examples = train_data[train_data.prob_diff < -data_cfg.threshold]

        neutral_examples = train_data[train_data.prob_diff.abs() <= data_cfg.threshold]

        extract_candidate_vectors(cfg, model, pos_examples, neg_examples, neutral_examples)

    validate(cfg, model, datasets["val"])


def main():
    args = parse_arguments()
    
    if args.config_file is not None:
        cfg = Config.load(args.config_file)
        logging.info(f"Loaded config file: {args.config_file}")
    else:
        data_cfg = DataConfig(
            task=args.task,
            n_train=args.n_train, n_val=args.n_val,
            hreshold=args.threshold, 
            max_new_tokens=args.max_new_tokens, top_p=args.top_p,
            num_return_sequences=args.num_return_sequences
        )
        cfg = Config(
            model_name=args.model_name, 
            data_cfg=data_cfg, 
            seed=args.seed, 
            filter_layer_pct=args.filter_layer_pct, 
            steering_test_size=args.steering_test_size,
            save_dir=args.save_dir, batch_size=args.batch_size, 
            generation_batch_size=args.generation_batch_size,
            use_cache=args.use_cache,
        )
        cfg.save()

    print("Model:", cfg.model_name)
    print("Configuration:")
    print(repr(cfg))
    print(repr(cfg.data_cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = load_model(cfg.model_name)
    train_and_validate(cfg, data_cfg, model)


if __name__ == "__main__":
    main()