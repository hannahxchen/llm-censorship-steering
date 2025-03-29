from tqdm import tqdm
from typing import Union, List, Iterable, Iterator, Tuple
from ..utils import ceildiv, chunks


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

