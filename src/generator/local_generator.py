
from llama_cpp import Llama
from typing import List, Union, Generator
from src.generator.abstract_generator import AbstractGenerator

_LLM_CACHE = {}

class LocalGenerator(AbstractGenerator):
    def __init__(self, model_path: str, n_ctx: int = 4096):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._model = self._get_llama_model()

    def _get_llama_model(self):
        if self.model_path not in _LLM_CACHE:
            try:
                _LLM_CACHE[self.model_path] = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    verbose=False,
                    n_gpu_layers=-1
                )
            except Exception as e:
                print(f"Error loading LLaMA model from {self.model_path} on GPU: {e}")
                _LLM_CACHE[self.model_path] = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    verbose=False
                )
        return _LLM_CACHE[self.model_path]

    def stream(self, query: str, chunks: List[Union[str, tuple]], 
               max_tokens: int = 300, 
               system_prompt_mode: str = "tutor", 
               temperature: float = 0.2) -> Generator[str, None, None]:
        prompt = self.format_prompt(chunks, query, system_prompt_mode=system_prompt_mode)
        
        for ev in self._model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[self.ANSWER_END],
            stream=True,
        ):
            delta = ev["choices"][0]["text"]
            yield delta

    def raw_completion(self, prompt: str, max_tokens: int, temperature: float, stop: List[str] = None):
        """
        Used for backward compatibility with query_enhancement.py
        """
        return self._model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [self.ANSWER_END]
        )
