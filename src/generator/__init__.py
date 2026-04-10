
from typing import List, Union, Generator
from src.generator.abstract_generator import AbstractGenerator

# Re-exporting constants and static methods for backward compatibility
ANSWER_START = AbstractGenerator.ANSWER_START
ANSWER_END = AbstractGenerator.ANSWER_END
text_cleaning = AbstractGenerator.text_cleaning
get_system_prompt = AbstractGenerator.get_system_prompt
format_prompt = AbstractGenerator.format_prompt
dedupe_generated_text = AbstractGenerator.dedupe_generated_text

# Lazy imports/exports to avoid hard dependency on openai
try:
    from src.generator.local_generator import LocalGenerator
except ImportError:
    LocalGenerator = None

try:
    from src.generator.cloud_generator import CloudGenerator
except ImportError:
    CloudGenerator = None

def get_generator(model_path: str) -> AbstractGenerator:
    """
    Factory function to get the appropriate generator based on model path.
    """
    if model_path.startswith("gpt-") or "openai" in model_path.lower() or model_path == "gemini":
        if CloudGenerator is None:
            raise ImportError("CloudGenerator requires 'openai' package. Please install it.")
        return CloudGenerator(model_name=model_path)
    else:
        if LocalGenerator is None:
             raise ImportError("LocalGenerator requires 'llama-cpp-python' package. Please install it.")
        return LocalGenerator(model_path=model_path)

def answer(query: str, chunks, model_path: str, max_tokens: int = 300, system_prompt_mode: str = "tutor", temperature: float = 0.2):
    gen = get_generator(model_path)
    return gen.stream(query, chunks, max_tokens, system_prompt_mode, temperature)

def stream_llama_cpp(prompt: str, model_path: str, max_tokens: int, temperature: float):
    # This is kept for backward compatibility if needed.
    # We can only support it for LocalGenerator as it takes a raw prompt.
    if LocalGenerator is None:
         raise ImportError("LocalGenerator requires 'llama-cpp-python' package. Please install it.")
    gen = LocalGenerator(model_path=model_path)
    for ev in gen._model.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[ANSWER_END],
        stream=True,
    ):
        yield ev["choices"][0]["text"]

def run_llama_cpp(prompt: str, model_path: str, max_tokens: int, temperature: float, **kwargs):
    gen = get_generator(model_path)
    if hasattr(gen, 'raw_completion'):
        return gen.raw_completion(prompt, max_tokens, temperature)
    else:
        # Fallback for generators that don't have raw_completion
        # Note: query and chunks are empty because we only have a raw prompt
        return {"choices": [{"text": "".join(gen.stream(query="", chunks=[], max_tokens=max_tokens, temperature=temperature))}]}

def double_answer(query: str, chunks, model_path: str,
                  max_tokens: int = 300,
                  system_prompt_mode: str = "tutor",
                  temperature: float = 0.2):
    gen = get_generator(model_path)
    
    # ---- Pass 1 ----
    initial_response = gen.generate(query, chunks, max_tokens, system_prompt_mode, temperature)
    initial_response = AbstractGenerator.dedupe_generated_text(initial_response)

    # ---- Pass 2 (repeat SAME question) ----
    base_prompt = AbstractGenerator.format_prompt(
        chunks,
        query,
        system_prompt_mode=system_prompt_mode
    )

    repeated_prompt = (
        base_prompt
        + initial_response
        + f"\n{ANSWER_END}\n"
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + f"Question: {query}"
        + "\n<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + ANSWER_START
    )

    if LocalGenerator is not None and isinstance(gen, LocalGenerator):
         for ev in gen._model.create_completion(
            repeated_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[ANSWER_END],
            stream=True,
        ):
            yield ev["choices"][0]["text"]
    elif CloudGenerator is not None and isinstance(gen, CloudGenerator):
        # CloudGenerator can use its client with raw prompt
        response = gen.client.chat.completions.create(
            model=gen.model_name,
            messages=[{"role": "user", "content": repeated_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[ANSWER_END],
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        # Default fallback
        yield from gen.stream(query, chunks, max_tokens, system_prompt_mode, temperature)
