
import os
from openai import OpenAI
from typing import List, Union, Generator
from src.generator.abstract_generator import AbstractGenerator

class CloudGenerator(AbstractGenerator):
    def __init__(self, model_name: str = "gemini", api_key: str = None, base_url: str = None):
        self.model_name = model_name
        
        final_api_key = os.environ.get("GEMINI_API_KEY")

        self.client = OpenAI(
            api_key=final_api_key,
            base_url=base_url
        )

    def stream(self, query: str, chunks: List[Union[str, tuple]], 
               max_tokens: int = 300, 
               system_prompt_mode: str = "tutor", 
               temperature: float = 0.2) -> Generator[str, None, None]:
        
        system_prompt = self.get_system_prompt(system_prompt_mode)
        
        # Prepare context from chunks
        if chunks and len(chunks) > 0:
            if isinstance(chunks[0], tuple):
                processed_chunks = [c[0] for c in chunks]
            else:
                processed_chunks = chunks
            context = "\n\n".join(processed_chunks)
            context = self.text_cleaning(context)
            user_content = f"Textbook Excerpts:\n{context}\n\n\nQuestion: {query}\n{self.ANSWER_START}"
        else:
            user_content = f"Question: {query}\n{self.ANSWER_START}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[self.ANSWER_END],
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def raw_completion(self, prompt: str, max_tokens: int, temperature: float, stop: List[str] = None):
        """
        Mimic OpenAI-style response for compatibility
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [self.ANSWER_END],
            stream=False
        )
        # Convert to dict for compatibility with existing code that uses dictionary indexing
        return {
            "choices": [
                {
                    "text": response.choices[0].message.content
                }
            ]
        }
