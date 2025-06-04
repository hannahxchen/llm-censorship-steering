import os, warnings
import json, argparse
import logging
from typing import List, Dict
from pathlib import Path
from .eval import WildGuard, JudgeLM, EvalPair
from .config import Config
from .utils import save_to_json_file


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Load configuration from file.')
    parser.add_argument('--run_wildguard', action='store_true', help='Run WildGuard.')
    parser.add_argument('--run_judgelm', action='store_true', help='Run JudgeLM.')
    parser.add_argument('--tasks', nargs="+", type=str, default=None, help='List of tasks to evaluate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse stored cached results.')
    parser.add_argument('--seed', type=int, default=3456, help='Random seed.')

    return parser.parse_args()


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


def run_wildguard_deepseek(artifact_dir: Path, task_name: str = None, batch_size: int = 16, use_cache: bool = True):
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



def run_judgelm(
    cfg: Config, task_list: List[str] = ["alpaca_test_sampled", "xstest_safe"], 
    baseline_output_dir: Path = None, batch_size: int = 16, use_cache: bool = True, skip_refusal: bool = True
):
    if baseline_output_dir is None:
        baseline_output_dir = cfg.artifact_path() / "evaluation"

    steering_output_dir = cfg.artifact_path() / "evaluation"
    save_dir = cfg.artifact_path() / "evaluation" / "judgelm_outputs"
    os.makedirs(save_dir, exist_ok=True)

    def filter_refusal_outputs(x: Dict):
        return [x for i, x in enumerate(x["completions"]) if x["refusal_response"][i] < 0.5]

    judgelm = JudgeLM()

    for task_name in task_list:
        baseline_outputs = json.load(open(baseline_output_dir / f"{task_name}_baseline.json", "r"))

        if skip_refusal and "refusal_response" in baseline_outputs[0]:
            for x in baseline_outputs:
                x["completions"] = filter_refusal_outputs(x)

        for filepath in Path(steering_output_dir).rglob(f'{task_name}_steering_coeff=*.json'):
            file_suffix = str(filepath).split("_")[-1]
            judgelm_output_filepath = save_dir / f"{task_name}_{file_suffix}"

            if use_cache and Path(judgelm_output_filepath).exists():
                continue

            steering_outputs = json.load(open(filepath))

            all_eval_inputs = []
            for baseline, steered in zip(baseline_outputs, steering_outputs):
                steered_answers = steered["completions"]

                if skip_refusal:
                    steered_answers = filter_refusal_outputs(steered)
                    if len(baseline["completions"]) == 0 and len(steered_answers) == 0:
                        continue

                all_eval_inputs.append(
                    EvalPair(
                        _id=baseline["_id"], 
                        prompt=baseline["prompt"],
                        baseline_answers=baseline["completions"], 
                        steered_answers=steered_answers
                    )
                )

            graded_eval_inputs = judgelm.run(all_eval_inputs, batch_size=batch_size)

            eval_results = [x.to_dict() for x in graded_eval_inputs]
            save_to_json_file(eval_results, judgelm_output_filepath)


def main():
    args = parse_arguments()
    cfg = Config.load(args.config_file)
    logging.info(f"Loaded config file: {args.config_file}")

    if args.run_wildguard:
        if "DeepSeek-R1" in cfg.model_name:
            run_wildguard_deepseek(cfg.artifact_path() / "evaluation", args.batch_size, use_cache=args.use_cache)
        else:
            run_wildguard(cfg.artifact_path() / "evaluation", args.batch_size, use_cache=args.use_cache)

    if args.run_judgelm:
        run_judgelm(cfg, task_list=args.tasks, batch_size=args.batch_size, use_cache=args.use_cache)


if __name__ == "__main__":
    main()
    