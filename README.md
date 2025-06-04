

# LLM Censorship Steering

<div align="center">
    <img src="assets/cover-photo.png" width="80%">
</div>

This repository contains code implementation for [*Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control*](https://arxiv.org/abs/2504.17130) by Hannah Cyberey and David Evans.

We introduce a method that finds "steering vectors" from LLM internals for detecting and controlling the level of censorship in model outputs. Check out this [blogpost]() for a brief overview of our work.

Try out our demos:
- 🐳 [Steering *Thought Suppression*]( https://mightbeevil.com/censorship) with DeepSeek-R1-Distill-Qwen-7B
- 🦙 [Steering *Refusal—Compliance*](https://hannahcyberey-refusal-censorship-steering.hf.space/) with Llama-3.1-8B-Instruct 

> **NOTE:** The second demo requires a Huggingface account. It's hosted with Huggingface's [ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu), which is free to all users with limited daily usage quota.

## Installation

Download the repository:

```bash
git clone https://github.com/hannahxchen/llm-censorship-steering.git
cd llm-censorship-steering
```

Create a virtual environment with Python 3.11+ and activate it:

```bash
conda create -y -n censorship-steering python=3.11
conda activate censorship-steering
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**Finding a Steering Vector**

To find a censorship steering vector for an instruction model, run:

```bash
python -m llm_steering.run \
    --run_train \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --censor_type refusal \
    --n_train 1000 --n_val 500 \
    --threshold 0.1 \
    --filter_layer_pct 0.2
```

A configuration file will be saved to the specified directory. Alternatively, you can use ```python -m llm_steering.run --config_file CONFIG_FILE``` by passing a YAML configuration file, following the format defined in ```llm_steering/config.py```.

For reasoning models, we use the following configuraiton:

```bash
python -m llm_steering.run \
    --run_train \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --censor_type thought_suppress \
    --n_train -1 --n_val 1000 \
    --threshold 0.1 \
    --filter_layer_pct 0.05 \
    --save_dir SAVE_DIR
```

**Applying the Steering Vector**

Use the following command to apply the steering vector:

```bash
python -m llm_steering.run
    --run_steering \
    --config_file SAVE_DIR/config.yaml \
    --generation_batch_size 8 \
    --coeff -1 \
    --datasets jailbreakbench ccp_sensitive
```
You can either set a single coefficient with ```--coeff``` or set a range of coefficient using ```--min_coeff```, ```--max_coeff```, and ```--increment```. By default, it applies values from -1 to 1 with an increment of 0.2. All model outputs will be saved to ```SAVE_DIR/evaluation/```.

All arguments of ```llm_steering/run.py```:

(For training and validation)
- ```model_name```: Use the name of the model repository on Huggingface.
- ```censor_type```: Use "refusal" for instruction models and "thought_suppress" for reasoning models.
- ```method```: Method for computing candidate vectors. Available options: ```WMD``` (weighted mean difference), ```MD``` (difference-in-means). Default method is ```WMD```.
- ```n_train```, ```n_valid```: Number of training and validation examples. If -1, all examples are used.
- ```threshold```: Threshold score for labeling censored/non-censored examples.
- ```filter_layer_pct```: Filter last N percentage layers.
- ```save_dir```: Directory path for saving the results.

(For applying steering vectors)
- ```run_steering```: Apply the steering vector found.
- ```compute_projection```: Compute scalar projections 
- ```datasets```: Dataset(s) to apply steering to. (See datasets available below)
- ```layer_ids```: Layer id(s) to intervene. Default uses only the top layer identified during vector validation.
- ```coeff```: Run a single coefficient value.
- ```min_coeff```: Minimum coefficient.
- ```max_coeff```: Maximum coefficient
- ```increment```: Increment of the coefficient.
- ```max_new_tokens```: Maximum number of generated tokens.
- ```num_return_sequences```: Number of generated sequences per input.
- ```top_p```: Top p value for sampling.
- ```temperature```: Temperature for sampling.

(Common arguments)
- ```config_file```: Path to the YAML configuration file.
- ```use_cache```: Reuse stored cached results. Useful if you need to resume the process but do not want to re-run the whole thing. The script will reuse/skip the saved artifacts (e.g., preprocessed train/valid data, outputs generated with a coefficient).
- ```batch_size```: Batch size for extracting activations.
- ```generation_batch_size```: Batch size for running generation.
- ``seed``: Random seed.


Datasets available:
- ```jailbreakbench```: Harmful split from [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors).
- ```sorrybench```: Full set of prompts from [SorryBench](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503).
- ```alpaca_test_sampled```: 300 prompts sampled from [Alpaca-Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned).
- ```xstest_safe```, ```xstest_unsafe```: Full set of prompts from [XSTest](https://huggingface.co/datasets/walledai/XSTest).
- ```ccp_sensitive```: [CCP Sensitive](https://huggingface.co/datasets/promptfoo/CCP-sensitive-prompts) prompts covering 68 different sensitive subjects. Each subject has 20 prompts.
- ```ccp_sensitive_sampled```: A smaller set of CCP Sensitive, which contains 5 prompts per subject.
- ```deccp_censored```: Censored split from [deccp](https://huggingface.co/datasets/augmxnt/deccp).

## Evaluation
To evaluate the generated outputs with *WildGuard*, run:

```bash
python -m llm_steering.run_eval \
    --config_file CONFIG_FILE_PATH \
    --batch_size BATCH_SIZE \
    --run_wildguard
```
The script will process all files of model outputs saved under ```SAVE_DIR/evaluation/```. Add ```--use_cache``` to skip ones that have already been processed.

[WildGuard](https://huggingface.co/allenai/wildguard) provides three types of detection and produces outputs in the following format:

```
Harmful request: yes
Response refusal: yes
Harmful response: no
```

We extract the probability of the "yes" or "no" token for each type of detection. The results will be added to the same file as the generated outputs.

## Citation
If you find this work useful, please consider citing our paper:

```
@article{cyberey2025steering,
    title={Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control}, 
    author={Hannah Cyberey and David Evans},
    year={2025},
    eprint={2504.17130},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2504.17130}, 
}
```