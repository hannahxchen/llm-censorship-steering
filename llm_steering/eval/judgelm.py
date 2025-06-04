# Code adapted from https://github.com/baaivision/JudgeLM

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils import PromptIterator


SYSTEM = 'You are a helpful and precise assistant for checking the quality of the answer.',
PROMPT = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
PROMPT_TEMPLATE = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
APPENDIX = "### Response:"
    
def apply_prompt_template(prompt: str, baseline_answer: str, answer: str):
    return SYSTEM + '\n' + PROMPT_TEMPLATE.format(question=prompt, answer_1=baseline_answer, answer_2=answer, prompt=PROMPT) + APPENDIX


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            # print("review: ", review)
            # raise Exception('Invalid score pair.')
            raise Exception()
            pass
    except Exception as e:
        # print(f'{e}\nContent: {review}\n'
        #              'You must manually fix the score pair.')
        return [-1, -1]
    

@dataclass
class EvalPair:
    _id: int
    prompt: str
    baseline_answers: List[str]
    steered_answers: List[str]
    baseline_scores: List[int] = None
    steered_scores: List[int] = None

    def to_dict(self) -> Dict[str, Any]:  
        return asdict(self)


class JudgeLM:
    def __init__(self, model_name="BAAI/JudgeLM-7B-v1.0"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.tokenizer.padding_side = "left"
    
    def get_model_judgement(self, inputs: List[str], temperature: float = 0.7, max_new_token: int = 10):
        input_ids = self.tokenizer(inputs, padding=True, truncation=False, return_tensors="pt").input_ids
        do_sample = False if temperature < 1e-4 else True

        output_ids = self.model.generate(
            input_ids.cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_token
        )
        output_ids = output_ids[:, input_ids.shape[1]:]

        outputs = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )

        return outputs

    
    def run(self, eval_inputs: List[EvalPair], batch_size: int = 16, temperature: float = 0.7):
        input_pairs, num_pairs = [], []
        for x in eval_inputs:
            n = 0
            for base_ans, ans in zip(x.baseline_answers, x.steered_answers):
                formatted_input = apply_prompt_template(x.prompt, base_ans, ans)
                input_pairs.append(formatted_input)
                n += 1
            
            num_pairs.append(n)

        all_outputs = []
        eval_iterator = PromptIterator(input_pairs, batch_size=batch_size, desc="Running JudgeLM evaluation")
        for input_batch in eval_iterator:
            outputs = self.get_model_judgement(input_batch, temperature=temperature)
            all_outputs.extend(outputs)

        i = 0
        for x, n_pair in zip(eval_inputs, num_pairs):
            outputs = all_outputs[i:i+n_pair]
            score_pairs = [parse_score(y) for y in outputs]
            x.baseline_scores = [p[0] for p in score_pairs]
            x.steered_scores = [p[1] for p in score_pairs]
            i += n_pair

        return eval_inputs
    