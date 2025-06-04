from dataclasses import dataclass
from typing import Dict, List, Callable, Self, Iterator
import numpy as np
from ..data import load_eval_dataset
from ..utils import save_to_json_file


def loop_coeffs(min_coeff=-1, max_coeff=1, increment=0.1) -> List[float]:
    coeffs = []
    n = int((max_coeff - min_coeff)/increment) + 1
    coeffs = np.array(range(n)) * increment + min_coeff
    coeffs = np.round(coeffs, 2)

    return coeffs.tolist()


def parse_reasoning_output(completion: str):
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


@dataclass
class Task:
    task_name: str
    dataset: List[Dict]

    @classmethod
    def load(cls, task_name: str) -> Self:
        dataset = load_eval_dataset(task_name)
        return cls(task_name=task_name, dataset=dataset)
    
    def prepare_inputs(self, chat_template_func: Callable) -> List[str]:
        return chat_template_func([x["prompt"] for x in self.dataset])
    
    def save_completions(self, outputs: Iterator[List[str]], filepath: str, parse_reasoning: bool = False):
        results = []
        for x, output in zip(self.dataset, outputs):
            if parse_reasoning:
                reasoning, answer = parse_reasoning_output(output[0])
                results.append({
                    "_id": x["_id"],
                    "prompt": x["prompt"],
                    "reasoning": reasoning,
                    "answer": answer
                })
            else:
                results.append({
                    "_id": x["_id"],
                    "prompt": x["prompt"],
                    "completions": output
                })

        save_to_json_file(results, filepath)
