import regex as re
from collections import Counter


class PreTokenizer:
    # GPT4 Pattern solves the long number token problem
    GPT4_PATTERN_STR = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    SIMPLE_PATTERN_STR = r"""\w+|[^\w\s]+|\s+"""

    def __init__(self, pattern: str = None):
        self.pattern = re.compile(pattern or self.GPT4_PATTERN_STR)

    def pre_tokenize(self, text: str) -> list[str]:
        return self.pattern.findall(text)

    def build_word_freq(self, text: str) -> dict[str, int]:
        chunks = self.pre_tokenize(text)
        return dict(Counter(chunks))