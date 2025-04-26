from typing import Dict, List, Callable
from ..data.load_dataset import load_eval_dataset

class Task():
    def __init__(self, task_name: str, dataset_name: str, is_qa: bool = False):
        self.task_name = task_name
        self.dataset = self.load_dataset(dataset_name)
        self.is_qa = is_qa

    def load_dataset(self, dataset_name):
        return load_eval_dataset(dataset_name)
    
    def prepare_inputs(self, chat_template_func: Callable) -> List[Dict]:
        inputs = []
        for x in self.dataset:
            inputs.append({
                "_id": x["_id"],
                "prompt": x["prompt"],
                "formatted_prompt": chat_template_func(x["prompt"])[0]
            })
            
        return inputs
    
    def process_results(self, results):
        pass
    

def load_task(task_name: str):
    return Task(task_name=task_name, dataset_name=task_name)

