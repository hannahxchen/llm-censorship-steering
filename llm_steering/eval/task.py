from typing import Dict, List, Callable
from ..data.load_dataset import load_eval_dataset

TASK_LIST = ["jailbreakbench", "sorrybench", "alpaca_test", "xstest_safe", "xstest_unsafe", "global_opinions", "ccp_sensitive"]

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


class GlobalOpinion(Task):
    def __init__(self):
         super(GlobalOpinion, self).__init__(task_name="global_opinion", dataset_name="global_opinions", is_qa=True)
    
    def _get_template(self):
        return "{QUESTION}\nHere are the options:\n{OPTIONS}"
    
    def prepare_inputs(self, chat_template_func):
        inputs = []
        template = self._get_template()

        for x in self.dataset:
            prompt = template.format(QUESTION=x["question"], OPTIONS="\n".join(x["options"]))
            formatted_prompt = chat_template_func(prompt, output_prefix="Answer:")[0]
            inputs.append({
                "id": x["_id"],
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "options": [" " + option + "." for option in x["options"]]
            })

        return inputs
    

def load_task(task_name: str):
    # assert task_name in TASK_LIST, f"Valid tasks: {TASK_LIST}"

    if task_name == "global_opinion":
        return GlobalOpinion()
    else:
        return Task(task_name=task_name, dataset_name=task_name)

