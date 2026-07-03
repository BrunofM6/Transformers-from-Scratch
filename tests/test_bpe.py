import os
import pytest
from tokenizer.bpe import BPTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "tiny-shakespeare.txt")

# --- Unit Tests ---

# get_stats
def test_get_stats_basic():
    ids = [1, 2, 3, 1, 2]
    stats = BPTokenizer._get_stats(ids)
    assert stats == {(1, 2): 2, (2, 3): 1, (3, 1): 1}

def test_get_stats_empty():
    assert BPTokenizer._get_stats([]) == {}

def test_get_stats_single_element():
    assert BPTokenizer._get_stats([5]) == {}

# merge
def test_merge_basic():
    ids = [1, 2, 3, 1, 2]
    result = BPTokenizer._merge(ids, (1, 2), 99)
    assert result == [99, 3, 99]

def test_merge_no_match():
    ids = [1, 2, 3]
    result = BPTokenizer._merge(ids, (5, 6), 99)
    assert result == ids


def test_merge_overlapping_left_to_right():
    ids = [1, 1, 1]
    result = BPTokenizer._merge(ids, (1, 1), 99)
    assert result == [99, 1]

# karpathy example
def test_karpathy_classic_example():
    # "Let's build the GPT Tokenizer"
    text = "aaabdaaabac"
    ids = list(text.encode("utf-8"))
    # 'a'=97, 'b'=98, 'd'=100, 'c'=99
    stats = BPTokenizer._get_stats(ids)
    # most freq. (97, 97) = "aa", 4 times
    most_frequent = max(stats, key=stats.get)
    assert most_frequent == (97, 97)
    assert stats[most_frequent] == 4

# roundtrip test
def test_roundtrip_small_corpus():
    corpus = "hello world, hello there. how are you doing today?"
    tok = BPTokenizer()
    tok.train(corpus, vocab_size=270)  # 256 + 14 merges
    encoded = tok.encode(corpus)
    decoded = tok.decode(encoded)
    assert decoded == corpus

def test_roundtrip_unseen_text():
    train_text = "the quick brown fox jumps over the lazy dog"
    tok = BPTokenizer()
    tok.train(train_text, vocab_size=270)
    unseen_text = "xyz completely different émoji 🎉 text"
    encoded = tok.encode(unseen_text)
    decoded = tok.decode(encoded)
    assert decoded == unseen_text

def test_compression_ratio_increases_with_vocab_size():
    corpus = "the quick brown fox jumps over the lazy dog. " * 50
    tok_small = BPTokenizer()
    tok_small.train(corpus, vocab_size=260)  # 4 merges
    tokens_small = tok_small.encode(corpus)
    tok_large = BPTokenizer()
    tok_large.train(corpus, vocab_size=310)  # 54 merges
    tokens_large = tok_large.encode(corpus)
    assert len(tokens_large) < len(tokens_small)

# real-time integration
@pytest.mark.skipif(
    not os.path.exists(SHAKESPEARE_PATH),
    reason=f"Corpus not found at {SHAKESPEARE_PATH}",
)
def test_vocab_completeness_after_training():
    with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
        text = f.read()[:5000]
    tok = BPTokenizer()
    tok.train(text, vocab_size=400)
    assert len(tok.vocab) <= 400
    assert len(tok.vocab) >= 256
    for i in range(256):
        assert i in tok.vocab
        assert tok.vocab[i] == bytes([i])

# vocab_size
def test_train_with_vocab_size_equal_256_no_merges():
    tok = BPTokenizer()
    tok.train("hello world", vocab_size=256)  # num_merges = 0
    assert tok.merges == {}
    assert len(tok.vocab) == 256

def test_train_with_vocab_size_below_256():
    tok = BPTokenizer()
    tok.train("hello world", vocab_size=100)  # num_merges neg
    assert tok.merges == {}

# encode
def test_encode_empty_string():
    tok = BPTokenizer()
    tok.train("hello world", vocab_size=270)
    assert tok.encode("") == []

def test_decode_empty_list():
    tok = BPTokenizer()
    tok.train("hello world", vocab_size=270)
    assert tok.decode([]) == ""

# determinism
def test_train_is_deterministic():
    corpus = "the quick brown fox jumps over the lazy dog"
    tok1 = BPTokenizer()
    tok1.train(corpus, vocab_size=280)
    tok2 = BPTokenizer()
    tok2.train(corpus, vocab_size=280)
    assert tok1.merges == tok2.merges