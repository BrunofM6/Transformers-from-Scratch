import pytest
from tokenizer.pre_tokenize import PreTokenizer

# GPT-4 pattern
def test_gpt4_splits_contractions():
    tok = PreTokenizer()
    chunks = tok.pre_tokenize("I've got it")
    assert "'ve" in chunks

def test_gpt4_contractions_case_insensitive():
    tok = PreTokenizer()
    chunks_lower = tok.pre_tokenize("don't")
    chunks_upper = tok.pre_tokenize("DON'T")
    assert "'t" in chunks_lower
    assert "'T" in chunks_upper

def test_gpt4_splits_long_numbers_into_groups_of_three():
    tok = PreTokenizer()
    chunks = tok.pre_tokenize("123456789")
    number_chunks = [c for c in chunks if c.strip().isdigit()]
    assert all(len(c.strip()) <= 3 for c in number_chunks)
    assert "".join(number_chunks) == "123456789"

def test_gpt4_short_number_not_split():
    tok = PreTokenizer()
    chunks = tok.pre_tokenize("42")
    assert "42" in chunks

def test_gpt4_leading_space_attached_to_word():
    tok = PreTokenizer()
    chunks = tok.pre_tokenize("hello world")
    assert "hello" in chunks
    assert " world" in chunks

def test_gpt4_punctuation_separated_from_words():
    tok = PreTokenizer()
    chunks = tok.pre_tokenize("hello, world!")
    assert "hello" in chunks
    assert "," in chunks or " world" in chunks

# roundtrip
@pytest.mark.parametrize("text", [
    "Hello, world! Don't stop.",
    "I've got $1,234.56 in my account.",
    "revenue was 123456789 dollars",
    "",
    "   ",
    "a",
    "multiple   spaces    here",
    "line1\nline2\nline3",
])
def test_pretokenize_roundtrip_gpt4(text):
    tok = PreTokenizer()
    chunks = tok.pre_tokenize(text)
    assert "".join(chunks) == text

@pytest.mark.parametrize("text", [
    "Hello, world!",
    "simple test 123",
    "",
])
def test_pretokenize_roundtrip_simple(text):
    tok = PreTokenizer(PreTokenizer.SIMPLE_PATTERN_STR)
    chunks = tok.pre_tokenize(text)
    assert "".join(chunks) == text

# build_word_freq
def test_build_word_freq_counts_correctly():
    tok = PreTokenizer()
    text = "the cat the dog the cat"
    freqs = tok.build_word_freq(text)
    the_chunks = {k: v for k, v in freqs.items() if k.strip() == "the"}
    assert sum(the_chunks.values()) == 3

def test_build_word_freq_empty_text():
    tok = PreTokenizer()
    assert tok.build_word_freq("") == {}

def test_build_word_freq_sum_matches_chunk_count():
    tok = PreTokenizer()
    text = "one two three two one"
    chunks = tok.pre_tokenize(text)
    freqs = tok.build_word_freq(text)
    assert sum(freqs.values()) == len(chunks)

# unicode / multi-idiom
def test_gpt4_handles_unicode_letters():
    tok = PreTokenizer()
    chunks = tok.pre_tokenize("café über")
    assert "".join(chunks) == "café über"
    assert any("café" in c for c in chunks)

def test_gpt4_handles_emoji():
    tok = PreTokenizer()
    text = "hello 🎉 world"
    chunks = tok.pre_tokenize(text)
    assert "".join(chunks) == text