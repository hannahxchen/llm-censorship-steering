import gc, json
import logging
from tqdm import tqdm
from pathlib import Path
from functools import singledispatch
from typing import List, Dict, Iterator, Tuple, Iterable, Union
import numpy as np
import torch


def ceildiv(a, b):
    return -(a // -b)


def chunks(lst, n) -> Iterator[List]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


def save_to_json_file(results: List[Dict], filepath: Path, silent: bool = False):
    with open(filepath, "w") as f:
        json.dump(results, f, default=to_serializable, indent=4)

    if not silent:
        logging.info(f"Results saved to: {filepath}")


def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


class PromptIterator:
    def __init__(self, prompts: Union[str, List[str]], batch_size=32, show_progress_bar=True, desc=None):
        self.batch_size = batch_size
        self.prompts = prompts

        total = ceildiv(len(self.prompts), self.batch_size)
        if total >= 5 and show_progress_bar:
            self.pbar = tqdm(total=total)

            if desc is not None:
                self.pbar.set_description(desc)
        else:
            self.pbar = None

    def _update(self, n):
        if self.pbar is not None:
            self.pbar.update(n)
    
    def _done(self):
        if self.pbar is not None:
            self.pbar.close()

    def _slice_prompts(self) -> Iterator[Tuple[List[str], float]]:
        for prompt_batch in chunks(self.prompts, self.batch_size):
            yield prompt_batch
            self._update(1)
        self._done()

    def __iter__(self) -> Iterable:
        return self._slice_prompts()
