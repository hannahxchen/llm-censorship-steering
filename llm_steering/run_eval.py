import os, warnings
import json, argparse
import logging, random
from pathlib import Path
import torch
import numpy as np
from .steering import load_model, ModelBase
from .eval.metrics import WildGuard
from .eval.evaluator import Evaluator
from .utils import save_to_json_file, clear_torch_cache, loop_coeffs
from .config import Config, EvalConfig
from .data.load_dataset import load_dataframe_from_json


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)

deepseek_generation_config = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "max_new_tokens": 2048}


SAFETY_TASKS =["jailbreakbench", "xstest_unsafe", "sorrybench"]
HARMLESS_TASKS = ["alpaca_test_sampled", "xstest_safe"]
SENSITIVE_TASKS = ["ccp_sensitive_sampled"]
CENSORSHIP_TASKS = ["ccp_sensitive", 'jailbreakbench', 'sorrybench', 'alpaca_test_sampled']


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_file', type=str, default=None, help='Load configuration from file.')
    parser.add_argument('--run_wildguard', action='store_true', help='Run WildGuard evaluation.')
    parser.add_argument('--layer', type=int, help="Steering layer.")
    parser.add_argument('--task_type', type=str, choices=['harmful', 'harmless', 'all', 'sensitive', 'censorship'], default='all', help='Task type.')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of generated tokens.')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top p value for sampling.')
    parser.add_argument('--num_return_sequences', type=int, default=5, help='Number of generated sequences per input.')
    parser.add_argument('--min_coeff', type=float, default=-1, help="Minimum steering coefficient.")
    parser.add_argument('--max_coeff', type=float, default=1, help="Maximum steering coefficient.")
    parser.add_argument('--increment', type=float, default=0.2, help="Increment of steering coefficient.")
    parser.add_argument('--coeff', type=float, default=None, help="Steering coefficient.")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse stored cached results.')
    parser.add_argument('--seed', type=int, default=3456, help='Random seed.')

    return parser.parse_args()

def get_projection_percentile(projections, pct=10, decimals=1):
    p = np.percentile(projections, pct)
    return float(np.round(p, decimals=decimals))


def run_eval(
    eval_cfg: EvalConfig, model: ModelBase, artifact_path: Path, batch_size: int, use_cache: bool = True,
    task_list=["jailbreakbench", "sorrybench", "alpaca_test_sampled", "xstest_safe", "xstest_unsafe"], 
    get_next_token_prob: bool = False, compute_projection: bool = False, run_reasoning: bool = False
):
    steering_vec = torch.load(artifact_path / "activations/candidate_vectors.pt", weights_only=True)[eval_cfg.layer]
    offset = torch.load(artifact_path / "activations/offsets.pt", weights_only=True)[eval_cfg.layer]
    
    val_data = load_dataframe_from_json(artifact_path / "datasplits/val.json")
    val_projections = np.load(artifact_path / "validation/projections.npy")

    k_pos = abs(get_projection_percentile(val_projections, 95)/val_data.prob_diff.quantile(0.95))
    k_neg = abs(get_projection_percentile(val_projections, 5)/val_data.prob_diff.quantile(0.05))

    evaluator = Evaluator(
        eval_cfg, model, steering_vec, 
        save_dir=artifact_path / "evaluation", batch_size=batch_size, offset=offset
    )

    if eval_cfg.coeff is None:
        coeff_list = loop_coeffs(eval_cfg.min_coeff, eval_cfg.max_coeff, eval_cfg.increment)
    else:
        coeff_list = [eval_cfg.coeff]
    
    for task_name in task_list:
        logging.info(f"Running task: {task_name}")
        task = evaluator.load_task(task_name)

        if compute_projection:
            evaluator.compute_projections(task, layer=eval_cfg.layer, use_cache=use_cache)

        evaluator.run(task, baseline=True, use_cache=use_cache, get_next_token_prob=get_next_token_prob, is_reasoning=run_reasoning)

        for coeff in coeff_list:
            logging.info(f"Steering coefficient={coeff:.1f}")
            if coeff > 0:
                k = k_pos
            elif coeff < 0:
                k = k_neg
            else:
                k = 0

            evaluator.run(task, coeff=coeff, k=k, baseline=False, use_cache=use_cache, get_next_token_prob=get_next_token_prob, is_reasoning=run_reasoning)

            clear_torch_cache()


def run_wildguard(artifact_dir: Path, batch_size: int = 16, use_cache: bool = True):
    wildguard = WildGuard()

    for filepath in Path(artifact_dir).rglob('*.json'):
        outputs = json.load(open(filepath))

        if use_cache and "harmful_request" in outputs[0]:
            continue

        prompts, responses = [], []
        for x in outputs:
            prompts.extend([x["prompt"]] * len(x["completions"]))
            responses.extend(x["completions"])

        all_scores = wildguard.run(prompts, responses, batch_size=batch_size)
        
        i = 0
        for x in outputs:
            n_completions = len(x["completions"])

            for key in wildguard.response_keys:
                x[key] = all_scores[key][i:i+n_completions].tolist()

            i += n_completions

        save_to_json_file(outputs, filepath)


def run_wildguard_deepseek(artifact_dir: Path, batch_size: int = 16, use_cache: bool = True):
    wildguard = WildGuard()

    for filepath in Path(artifact_dir).rglob('*.json'):
        outputs = json.load(open(filepath))

        if use_cache and "harmful_request" in outputs[0]:
            continue

        prompts, responses = [], []
        for x in outputs:
            if x['answer'] is None:
                continue
            prompts.append(x["prompt"])
            responses.append(x["answer"])

        all_scores = wildguard.run(prompts, responses, batch_size=batch_size)
        
        i = 0
        for x in outputs:
            if x['answer'] is None:
                continue

            x['harmful_request'] = all_scores['harmful_request'][i]
            x['harmful_answer'] = all_scores['harmful_response'][i]
            x['refusal_answer'] = all_scores['refusal_response'][i]

            i += 1

        save_to_json_file(outputs, filepath)

        prompts, responses = [], []
        for x in outputs:
            if x['reasoning'] is None:
                continue
            prompts.append(x["prompt"])
            responses.append(x["reasoning"])

        all_scores = wildguard.run(prompts, responses, batch_size=batch_size)
        
        i = 0
        for x in outputs:
            if x['reasoning'] is None:
                continue

            x['harmful_reasoning'] = all_scores['harmful_response'][i]
            x['refusal_reasoning'] = all_scores['refusal_response'][i]

            i += 1

        save_to_json_file(outputs, filepath)


def main():
    args = parse_arguments()
    cfg = Config.load(args.config_file)
    logging.info(f"Loaded config file: {args.config_file}")

    if args.run_wildguard:
        if "DeepSeek-R1" in cfg.model_name:
            run_wildguard_deepseek(cfg.artifact_path() / "evaluation", args.batch_size, use_cache=args.use_cache)
        else:
            run_wildguard(cfg.artifact_path() / "evaluation", args.batch_size, use_cache=args.use_cache)
    else:
        print("Model:", cfg.model_name)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        model = load_model(cfg.model_name)
        args.layer = json.load(open(cfg.artifact_path() / "validation/top_layers.json", "r"))[0]['layer']

        eval_cfg = EvalConfig(
            layer=args.layer, coeff=args.coeff,
            min_coeff=args.min_coeff, max_coeff=args.max_coeff, 
            increment=args.increment,
            max_new_tokens=args.max_new_tokens, 
            num_return_sequences=args.num_return_sequences, 
            top_p=args.top_p, do_sample=True
        )

        if args.task_type == "harmless":
            task_list = HARMLESS_TASKS
        elif args.task_type == "sensitive":
            task_list = SENSITIVE_TASKS
        elif args.task_type == "all":
            task_list = SAFETY_TASKS + HARMLESS_TASKS
        elif args.task_type == "censorship":
            task_list = CENSORSHIP_TASKS
        else:
            task_list = SAFETY_TASKS
        

        if "DeepSeek-R1" in cfg.model_name:
            get_next_token_prob = True
            # run_reasoning = True
        else:
            get_next_token_prob = False
            # run_reasoning = False

        run_eval(
            eval_cfg, model, cfg.artifact_path(), batch_size=args.batch_size, use_cache=args.use_cache, 
            task_list=task_list, 
            # run_reasoning=run_reasoning,
            get_next_token_prob=get_next_token_prob, 
        )


if __name__ == "__main__":
    main()