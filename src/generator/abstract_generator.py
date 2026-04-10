
import re
import textwrap
from abc import ABC, abstractmethod
from typing import List, Union, Generator

class AbstractGenerator(ABC):
    """
    Abstract class for generating LLM responses
    """

    ANSWER_START = "<<<ANSWER>>>"
    ANSWER_END   = "<<<END>>>"

    @staticmethod
    def text_cleaning(prompt: str) -> str:
        _CONTROL_CHARS_RE = re.compile(r'[\u0000-\u001F\u007F-\u009F]')
        _DANGEROUS_PATTERNS = [
            r'ignore\s+(all\s+)?previous\s+instructions?',
            r'you\s+are\s+now\s+(in\s+)?developer\s+mode',
            r'system\s+override',
            r'reveal\s+prompt',
        ]
        text = _CONTROL_CHARS_RE.sub('', prompt)
        text = re.sub(r'\s+', ' ', text).strip()
        for pat in _DANGEROUS_PATTERNS:
            text = re.sub(pat, '[FILTERED]', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def get_system_prompt(mode: str = "tutor") -> str:
        """
        Get system prompt based on mode.
        """
        prompts = {
            "baseline": "",
            
            "tutor": textwrap.dedent(f"""
                You are a tutor. Follow these rules:
                1. Analyze the user question and identify all parts that need answering.
                2. Refer ONLY to the provided textbook excerpts to find answers to all parts.
                3. Answer the question completely and concisely, as if teaching a student.
                End your reply with {AbstractGenerator.ANSWER_END}.
            """).strip(),
            
            "concise": textwrap.dedent(f"""
                You are a concise assistant. Answer questions briefly and directly using the provided textbook excerpts.
                - Keep answers short and to the point
                - Focus on key concepts only
                - Use bullet points when appropriate
                End your reply with {AbstractGenerator.ANSWER_END}.
            """).strip(),
            
            "detailed": textwrap.dedent(f"""
                You are a comprehensive educational assistant. Provide thorough, detailed explanations using the provided textbook excerpts.
                - Explain concepts in depth with context
                - Include relevant examples and analogies
                - Break down complex ideas into understandable parts
                - Use proper formatting (markdown, bullets, etc.)
                - Connect concepts to broader topics when relevant
                End your reply with {AbstractGenerator.ANSWER_END}.
            """).strip(),
        }
        
        return prompts.get(mode, "")

    @staticmethod
    def format_prompt(chunks: List[Union[str, tuple]], query: str, system_prompt_mode: str = "tutor") -> str:
        """
        Format prompt for LLM with chunks and query.
        """
        # Get system prompt
        system_prompt = AbstractGenerator.get_system_prompt(system_prompt_mode)
        system_section = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n" if system_prompt else ""
        
        # Build prompt based on whether chunks are provided
        if chunks and len(chunks) > 0:
            if isinstance(chunks[0], tuple):
                processed_chunks = [c[0] for c in chunks]
            else:
                processed_chunks = chunks
            context = "\n\n".join(processed_chunks)
            context = AbstractGenerator.text_cleaning(context)
            
            # Build prompt with chunks
            context_section = f"Textbook Excerpts:\n{context}\n\n\n"
            
            final_prompt = textwrap.dedent(f"""\
                {system_section}<|im_start|>user
                {context_section}Question: {query}
                <|im_end|>
                <|im_start|>assistant
                {AbstractGenerator.ANSWER_START}
            """)
            return final_prompt

        else:
            # Build prompt without chunks
            question_label = "Question: " if system_prompt else ""
            
            return textwrap.dedent(f"""\
                {system_section}<|im_start|>user
                {question_label}{query}
                <|im_end|>
                <|im_start|>assistant
                {AbstractGenerator.ANSWER_START}
            """)

    @abstractmethod
    def stream(self, query: str, chunks: List[Union[str, tuple]], 
               max_tokens: int = 300, 
               system_prompt_mode: str = "tutor", 
               temperature: float = 0.2) -> Generator[str, None, None]:
        """
        Stream response from LLM.
        """
        pass

    def generate(self, query: str, chunks: List[Union[str, tuple]], 
                 max_tokens: int = 300, 
                 system_prompt_mode: str = "tutor", 
                 temperature: float = 0.2) -> str:
        """
        Generate full response from LLM.
        """
        return "".join(self.stream(query, chunks, max_tokens, system_prompt_mode, temperature))

    @staticmethod
    def dedupe_generated_text(text: str) -> str:
        """
        Removes immediate consecutive duplicate sentences or lines from LLM output.
        """
        lines = text.split("\n")
        cleaned = []
        prev = None
        for line in lines:
            normalized = line.strip().lower()
            # Skip if this line is a repeat of the previous one
            if normalized == prev and normalized != "":
                continue
            cleaned.append(line)
            prev = normalized
        return "\n".join(cleaned)
