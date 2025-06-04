import os, argparse, random
import warnings, logging
from pathlib import Path
from typing import List, Optional

import torch
from torchtyping import TensorType
import numpy as np
from .config import Config, SteeringConfig
from .data import EVAL_DATASETS
from .steering import ModelBase, SteeringVector, Dataset
from .eval import Evaluator, Task
from .utils import PromptIterator, clear_torch_cache

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_grad_enabled(False);
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='Load configuration from file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--generation_batch_size', type=int, default=8, help='Batch size for text generation.')
    parser.add_argument('--seed', type=int, default=4238, help='Random seed.')
    parser.add_argument('--save_dir', type=str, help='Save results to specified directory.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse stored cached results.')

    # Train a steering vector
    parser.add_argument('--run_train', action='store_true', help='Run training.')
    parser.add_argument('--model_name', type=str, help='Model name.')
    parser.add_argument('--censor_type', type=str, choices=["refusal", "thought_suppress"], help='Target censorship type.')
    parser.add_argument('--method', type=str, default="WMD", choices=["WMD", "MD"], help='Method for computing candidate vectors.')
    parser.add_argument('--n_train', type=int, default=1000, help="Number of training examples.")
    parser.add_argument('--n_val', type=int, default=500, help="Number of validation examples.")
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--filter_layer_pct', type=float, default=0.2, help='Filter last N percentage layers.')

    # Apply Steering
    parser.add_argument('--run_steering', action='store_true', help='Apply steering.')
    parser.add_argument('--compute_projection', action='store_true', help='Compute scalar projections.')
    parser.add_argument('--datasets', nargs="+", type=str, choices=EVAL_DATASETS, help='Dataset(s) to apply steering to.')
    parser.add_argument('--layer_ids', nargs="+", type=int, default=None, help='Layer id(s) to intervene.')
    parser.add_argument('--min_coeff', type=float, default=-1, help="Minimum steering coefficient.")
    parser.add_argument('--max_coeff', type=float, default=1, help="Maximum steering coefficient.")
    parser.add_argument('--increment', type=float, default=0.2, help="Increment of steering coefficient.")
    parser.add_argument('--coeff', type=float, default=None, help="Steering coefficient.")
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of generated tokens.')
    parser.add_argument('--num_return_sequences', type=int, default=5, help='Number of generated sequences per input.')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top p value for sampling.')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for sampling.')
    return parser.parse_args()


def get_all_layer_activations(
    model: ModelBase, prompts: List[str], batch_size: Optional[int] = 32, positions=[-1]
) -> TensorType["n_layer", "n_prompt", "hidden_size"]:
    """
    Extract activations from all model layers.
    """
    acts_all = []
    layers = list(range(model.n_layer))
    prompt_iterator = PromptIterator(prompts, batch_size=batch_size)
    if prompt_iterator.pbar is not None:
        prompt_iterator.pbar.set_description("Extracting activations")

    for prompt_batch in prompt_iterator:
        acts = model.get_activations(layers, prompt_batch, positions=positions).squeeze(-2)
        acts_all.append(acts)

    return torch.concat(acts_all, dim=1).to(torch.float64)


def train_and_validate(cfg: Config, model: ModelBase, batch_size: int, generation_batch_size: int, use_cache: bool = False):
    """
    Extract candidate vectors and select a steering vector.
    """
    datasplits_dir = cfg.artifact_path() / "datasplits"

    if use_cache:
        cached_dir = datasplits_dir

    dataset = Dataset.load(cfg.censor_type, cfg.n_train, cfg.n_val, cfg.threshold, cached_dir=cached_dir)
    if dataset.use_next_token:
        dataset.compute_baseline_scores(model, batch_size=batch_size, use_cache=use_cache)
    else:
        dataset.compute_baseline_scores(model, batch_size=generation_batch_size, use_cache=use_cache)
    dataset.save(cfg.artifact_path() / "datasplits")

    pos_examples = dataset.pos_examples()
    neg_examples = dataset.neg_examples()

    pos_prompts = dataset.get_formatted_prompts(pos_examples, model.apply_chat_template)
    pos_acts = get_all_layer_activations(model, pos_prompts, batch_size)
    neg_prompts = dataset.get_formatted_prompts(neg_examples, model.apply_chat_template)
    neg_acts = get_all_layer_activations(model, neg_prompts, batch_size)

    kwargs = {}
    if cfg.method == "WMD":
        kwargs["pos_weights"] = torch.Tensor(pos_examples["censor_score"].tolist())
        kwargs["neg_weights"] = torch.Tensor(neg_examples["censor_score"].tolist())

        neutral_prompts = dataset.get_formatted_prompts(dataset.neutral_examples(), model.apply_chat_template)
        kwargs["neutral_acts"] = get_all_layer_activations(model, neutral_prompts, batch_size)

    steering_vec = SteeringVector.fit(cfg.method, pos_acts, neg_acts, **kwargs)

    val_examples = dataset.val_examples()
    val_prompts = dataset.get_formatted_prompts(val_examples, model.apply_chat_template)
    val_acts = get_all_layer_activations(model, val_prompts, batch_size)

    results = steering_vec.validate(val_acts, censor_scores=val_examples["censor_score"].to_numpy(), save_dir=cfg.artifact_path())
    top_layer = steering_vec.get_top_layer_id(results, save_dir=cfg.artifact_path(), filter_layer_pct=cfg.filter_layer_pct)
    steering_vec.save(cfg.artifact_path() / "steering_vec")
    logging.info(f"Steering vector files saved to directory: {cfg.artifact_path() / "steering_vec"}")

    return steering_vec, top_layer


def apply_steering(
    steering_cfg: SteeringConfig, model: ModelBase, steering_vec: SteeringVector, task_list: List[str], 
    artifact_path: Path, batch_size: int, generation_batch_size: int, use_cache: bool = True, parse_reasoning: bool = False
):
    """
    Apply the steering vector.
    """
    save_dir = artifact_path / "evaluation"
    os.makedirs(save_dir, exist_ok=True)

    evaluator = Evaluator(
        steering_cfg, save_dir=artifact_path / "evaluation", use_cache=use_cache,
        batch_size=batch_size, generation_batch_size=generation_batch_size
    )
    
    for task_name in task_list:
        logging.info(f"Running task: {task_name}")
        task = Task.load(task_name=task_name)
        evaluator.run_baseline(model, task, parse_reasoning=parse_reasoning)
        evaluator.run_steering(model, task, steering_vec=steering_vec, parse_reasoning=parse_reasoning)

        clear_torch_cache()


def compute_projections(
    model: ModelBase, steering_vec: SteeringVector, layer_id: int, task_list: List[str], 
    artifact_path: Path, batch_size: int, use_cache: bool = True
):
    """
    Compute scalar projections on the steering vector (based on the input's last token position).
    """
    save_dir = artifact_path / "projections"
    os.makedirs(save_dir, exist_ok=True)

    for task_name in task_list:
        logging.info(f"Running task: {task_name}")
        task = Task.load(task_name=task_name)

        filepath = save_dir / f"{task_name}_projections.npy"
        if use_cache and Path(filepath).exists():
            return
        
        prompts = task.prepare_inputs(model.apply_chat_template)
        prompt_iterator = PromptIterator(prompts, batch_size=batch_size, desc="Extracting activations for computing projections")

        activations = []
        
        for prompt_batch in prompt_iterator:
            acts = model.get_activations(layer_id, prompt_batch, positions=[-1]).squeeze()
            activations.append(acts)

        activations = torch.vstack(activations).to(torch.float64)
        steering_vec.set_dtype(torch.float64)
        projs = steering_vec.scalar_projection(activations, layer_id)
        np.save(filepath, np.array(projs))
        print(f"Projections saved to: {filepath}")


def main():
    args = parse_arguments()
    
    if args.config_file is not None:
        cfg = Config.load(args.config_file)
        logging.info(f"Loaded config file: {args.config_file}")
    else:
        cfg = Config(
            model_name=args.model_name, censor_type=args.censor_type,
            n_train=args.n_train, n_val=args.n_val,
            method=args.method, threshold=args.threshold, 
            filter_layer_pct=args.filter_layer_pct, 
            save_dir=args.save_dir, seed=args.seed
        )
        cfg.save()

    print("Model:", cfg.model_name)
    print("Configuration:")
    print(repr(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = ModelBase.load(cfg.model_name)

    if args.run_train:
        steering_vec, top_layer = train_and_validate(
            cfg, model, batch_size=args.batch_size, 
            generation_batch_size=args.generation_batch_size, use_cache=args.use_cache
        )

        logging.info(f"Top layer: {top_layer}")
        args.layer_ids = top_layer

    if args.run_steering or args.compute_projection:
        steering_vec = SteeringVector.load(cfg.artifact_path() / "steering_vec")

        if args.layer_ids is None:
            layer_ids = steering_vec.get_top_layer_id(save_dir=cfg.artifact_path(), filter_layer_pct=cfg.filter_layer_pct)
        else:
            layer_ids = args.layer_ids
        
        if args.compute_projection:
            compute_projections(
                model, steering_vec, 
                layer_id=layer_ids, 
                task_list=args.datasets, 
                artifact_path=cfg.artifact_path(), 
                batch_size=args.batch_size, 
                use_cache=args.use_cache
            )
        
        if args.run_steering:
            steering_cfg = SteeringConfig(
                layer_ids=layer_ids, 
                coeff=args.coeff,
                min_coeff=args.min_coeff, 
                max_coeff=args.max_coeff, 
                increment=args.increment,
                max_new_tokens=args.max_new_tokens, 
                num_return_sequences=args.num_return_sequences, 
                do_sample=True, 
                temperature=args.temperature,
                top_p=args.top_p
            )

            print("Steering configuration:")
            print(repr(steering_cfg))
            
            if "DeepSeek-R1" in cfg.model_name:
                parse_reasoning = True
            else:
                parse_reasoning = False

            apply_steering(
                steering_cfg, model, 
                steering_vec,
                task_list=args.datasets, 
                artifact_path=cfg.artifact_path(),
                batch_size=args.batch_size, 
                generation_batch_size=args.generation_batch_size, 
                use_cache=args.use_cache, 
                parse_reasoning=parse_reasoning
            )


if __name__ == "__main__":
    main()