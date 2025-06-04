import os, warnings
from operator import attrgetter
from typing import List, Optional, Union, Callable, Self

import torch
from torchtyping import TensorType
from transformers import AutoTokenizer, BatchEncoding, AutoConfig
from nnsight import LanguageModel
from nnsight.intervention import Envoy

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda"


class ModelBase:
    def __init__(
        self, model_name: str, tokenizer: AutoTokenizer = None, 
        block_module_attr: Optional[str] = None,
        system_role: Optional[bool] = False, **model_kwargs
    ):
        if tokenizer is None:
            self.tokenizer = self._load_tokenizer(model_name)
        else:
            self.tokenizer = tokenizer

        self.system_role = system_role
        self.model = self._load_model(model_name, self.tokenizer, **model_kwargs)
        self.device = self.model.device
        self.dtype = self.model.dtype
        
        self.n_layer = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        if block_module_attr is None:
            block_module_attr = detect_module_attrs(self.model)
        self.block_modules = self.get_module(block_module_attr)
    
    @staticmethod
    def _load_model(model_name: str, tokenizer: AutoTokenizer, **kwargs) -> LanguageModel:
        return LanguageModel(model_name, tokenizer=tokenizer, dispatch=True, trust_remote_code=True, **kwargs)
    
    @staticmethod
    def _load_tokenizer(model_name, **kwargs) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def get_module(self, attr: str) -> Envoy:
        return attrgetter(attr)(self.model)
    
    @classmethod
    def load(
        cls, model_name: str, tokenizer: AutoTokenizer = None, 
        device_map="auto", torch_dtype=torch.bfloat16, 
        block_module_attr=None, use_default_config: bool = True, **model_kwargs
    ) -> Self:
        if use_default_config:
            tokenizer, block_module_attr, model_kwargs = get_default_configuration(model_name, **model_kwargs)

        return cls(model_name, tokenizer=tokenizer, device_map=device_map, 
                   torch_dtype=torch_dtype, block_module_attr=block_module_attr, **model_kwargs)

    def tokenize(self, prompts: Union[str, List[str], BatchEncoding]) -> BatchEncoding:
        if isinstance(prompts, BatchEncoding):
            return prompts
        else:
            return self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    
    def apply_chat_template(
        self, instructions: Union[str, List[str]], 
        system_message: Optional[str] = None, 
        add_generation_prompt: Optional[bool] = True,
        output_prefix: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        if isinstance(instructions, str):
            instructions = [instructions]

        prompts = []
        for i in range(len(instructions)):
            messages = []
            inputs = instructions[i]

            if system_message:
                if self.system_role:
                    messages.append({"role": "system", "content": system_message})
                else:
                    inputs = system_message + " " + instructions[i]

            messages.append({"role": "user", "content": inputs})
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

            if output_prefix is not None:
                if isinstance(output_prefix, str):
                    inputs += output_prefix
                else:
                    inputs += output_prefix[i]
            prompts.append(inputs)
            
        return prompts
        
    def get_activations(
        self, layers: Union[int, List[int]], 
        prompts: Union[str, List[str], BatchEncoding],
        positions: Optional[List[int]] = [-1]
    ) -> List[TensorType["n_layer", "n_prompt", "n_pos", "hidden_size"]]:
        
        if isinstance(layers, int):
            layers = [layers]

        inputs = self.tokenize(prompts)
        all_acts = []
        
        with self.model.trace(inputs) as tracer:
            for layer in layers:
                if positions is None:
                    acts = self.block_modules[layer].output[0]
                else:
                    acts = self.block_modules[layer].output[0][:, positions, :]

                acts = acts.detach().to("cpu").unsqueeze(0).save()
                all_acts.append(acts)

            self.block_modules[layer].output.stop() # Early stopping
        return torch.vstack(all_acts)
    
    def get_logits(
        self, prompts: Union[str, List[str], BatchEncoding], 
        layer_ids: Union[int, List[int]] = None, 
        steering_func: Callable = None, **kwargs
    ) -> TensorType["n_prompt", "seq_len", "vocab_size"]:
        inputs = self.tokenize(prompts)

        if steering_func is not None:
            if isinstance(layer_ids, int):
                layer_ids = [layer_ids]

            with self.model.trace(inputs) as tracer:
                for layer in layer_ids:
                    acts = self.block_modules[layer].output[0].clone()
                    new_acts = steering_func(acts, layer)
                    self.block_modules[layer].output[0][:] = new_acts
                    logits = self.model.lm_head.output.save()
        else:
            logits = self.model.trace(inputs, trace=False).logits
            
        return logits.detach().to("cpu").to(torch.float64)

    def get_last_position_logits(self, prompts: Union[str, List[str], BatchEncoding], **kwargs) -> TensorType["n_prompt", "vocab_size"]:
        return self.get_logits(prompts, **kwargs)[:, -1, :]
    
    def generate(
        self, prompts: Union[str, List[str], TensorType[int, "n_prompt", "seq_len"]],  
        layer_ids: Union[int, List[int]] = None, steering_func: Callable = None, 
        max_new_tokens: int = 10, do_sample: bool = False, **kwargs
    ) -> List[str]:
        inputs = self.tokenize(prompts)

        if steering_func is not None:
            if isinstance(layer_ids, int):
                layer_ids = [layer_ids]

            with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
                self.block_modules.all()
                
                for layer in layer_ids:
                    acts = self.block_modules[layer].output[0].clone()
                    new_acts = steering_func(acts, layer)
                    self.block_modules[layer].output[0][:] = new_acts

                outputs = self.model.generator.output.save()
        else:
            outputs = self.model._model.generate(**inputs.to(self.device), max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs)
        
        input_len = inputs.input_ids.shape[1]
        completions = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        return completions



def detect_module_attrs(model: LanguageModel) -> str:
    if "model" in model._modules and "layers" in model.model._modules:
        return "model.layers"
    elif "transformers" in model._modules and "h" in model.transformers._modules:
        return "transformers.h"
    else:
        raise Exception("Failed to detect module attributes.")
    

QWEN_CHAT_TEMPLATE = """\
{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>' }}\n\
{% elif message['role'] == 'user' %}\n{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>' }}\n\
{% elif message['role'] == 'assistant' %}\n{{ '<|im_start|>assistant\n'  + message['content'] + '<|im_end|>' }}\n\
{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|im_start|>assistant' }}\n{% endif %}\n{% endfor %}\
""" 

def get_default_configuration(model_name, **model_kwargs):
    tokenizer = None
    block_module_attr = None

    if model_name.startswith("Qwen/Qwen-"):
        tokenizer = ModelBase._load_tokenizer(model_name, use_fast=False)
        # The model never sees or computes the pad token, so you may use any known token.
        # Reference: https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE

        block_module_attr = "transformer.h"

        if model_name == "Qwen/Qwen-1_8B-chat":
            if "config" not in model_kwargs:
                config = AutoConfig.from_pretrained("Qwen/Qwen-1_8B-chat", trust_remote_code=True)
                config.use_flash_attn = False
                model_kwargs["config"] = config

    elif model_name.startswith('google/gemma-2'):
        model_kwargs["attn_implementation"] = "eager"

    elif "DeepSeek-R1-Distill-Qwen" in model_name:
        tokenizer = ModelBase._load_tokenizer(model_name)
        tokenizer.chat_template = tokenizer.chat_template.replace("<｜Assistant｜><think>\\n", "<｜Assistant｜><think>")

    return tokenizer, block_module_attr, model_kwargs
