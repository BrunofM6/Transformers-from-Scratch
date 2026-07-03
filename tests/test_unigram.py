import math
import pytest
from tokenizer.unigram import UnigramTokenizer
from tokenizer.pre_tokenize import PreTokenizer

# viterbi_segment
def test_viterbi_segment_simple_vocab():
    tok = UnigramTokenizer(max_piece_len=8)
    vocab = {
        "un": math.log(0.3),
        "happy": math.log(0.2),
        "u": math.log(0.05),
        "n": math.log(0.05),
        "h": math.log(0.01),
        "a": math.log(0.05),
        "p": math.log(0.05),
        "y": math.log(0.05),
    }
    pieces, score = tok._viterbi_segment("unhappy", vocab)
    assert pieces == ["un", "happy"]
    assert "".join(pieces) == "unhappy"

def test_viterbi_segment_falls_back_to_single_chars():
    tok = UnigramTokenizer(max_piece_len=8)
    vocab = {ch: math.log(0.1) for ch in "abc"}
    pieces, score = tok._viterbi_segment("abc", vocab)
    assert pieces == ["a", "b", "c"]

def test_viterbi_segment_prefers_higher_probability_path():
    tok = UnigramTokenizer(max_piece_len=8)
    vocab = {
        "ab": math.log(0.9),
        "a": math.log(0.05),
        "b": math.log(0.05),
    }
    pieces, score = tok._viterbi_segment("ab", vocab)
    assert pieces == ["ab"]

def test_viterbi_segment_respects_max_piece_len():
    tok = UnigramTokenizer(max_piece_len=2)
    vocab = {
        "abc": math.log(0.9),
        "ab": math.log(0.3),
        "c": math.log(0.1),
        "a": math.log(0.05),
        "b": math.log(0.05),
    }
    pieces, score = tok._viterbi_segment("abc", vocab)
    assert pieces != ["abc"]
    assert "".join(pieces) == "abc"

def test_viterbi_segment_empty_word():
    tok = UnigramTokenizer()
    pieces, score = tok._viterbi_segment("", {"a": math.log(0.5)})
    assert pieces == []
    assert score == 0.0

# seed_vocab_all_substrings
def test_seed_vocab_all_substrings_contains_all_chars():
    tok = UnigramTokenizer(max_piece_len=4)
    word_freqs = {"cat": 5, "dog": 3}
    candidates = tok._seed_vocab_all_substrings(word_freqs)
    for ch in "catdog":
        assert ch in candidates

def test_seed_vocab_all_substrings_respects_max_piece_len():
    tok = UnigramTokenizer(max_piece_len=3)
    word_freqs = {"abcdef": 1}
    candidates = tok._seed_vocab_all_substrings(word_freqs)
    assert all(len(piece) <= 3 for piece in candidates)

def test_seed_vocab_all_substrings_counts_weighted_by_freq():
    tok = UnigramTokenizer(max_piece_len=4)
    word_freqs = {"aa": 10}
    candidates = tok._seed_vocab_all_substrings(word_freqs)
    assert candidates["a"] == 20

# seed_vocab_from_bpe
def test_seed_vocab_from_bpe_contains_fallback_chars():
    tok = UnigramTokenizer(max_piece_len=8)
    word_freqs = {"hello": 5, "world": 3}
    candidates = tok._seed_vocab_from_bpe(word_freqs, bpe_vocab_size=270)
    for ch in "helloworld":
        assert ch in candidates

# train()
@pytest.mark.parametrize("seed_method", ["all_substrings", "bpe"])
def test_train_reaches_target_vocab_size(seed_method):
    word_freqs = {
        "the": 50, "quick": 10, "brown": 8, "fox": 6,
        "jumps": 5, "over": 20, "lazy": 4, "dog": 12,
    }
    tok = UnigramTokenizer(max_piece_len=6)
    tok.train(word_freqs, vocab_size=30, seed_method=seed_method, bpe_seed_multiplier=4)
    assert len(tok.vocab) <= 30

def test_train_invalid_seed_method_raises():
    tok = UnigramTokenizer()
    with pytest.raises(ValueError):
        tok.train({"a": 1}, vocab_size=10, seed_method="nonexistent")

def test_train_keeps_all_fallback_chars():
    word_freqs = {"xyz": 5, "abc": 5}
    tok = UnigramTokenizer(max_piece_len=4)
    tok.train(word_freqs, vocab_size=8, seed_method="all_substrings")
    for ch in "xyzabc":
        assert ch in tok.vocab

def test_train_does_not_loop_forever_with_small_vocab_size():
    word_freqs = {"abcdefghij": 1}
    tok = UnigramTokenizer(max_piece_len=4)
    tok.train(word_freqs, vocab_size=3, seed_method="all_substrings")
    assert len(tok.vocab) >= 10 


# roundtrip / encode
def test_encode_covers_full_word():
    word_freqs = {"testing": 10, "test": 5, "tester": 3}
    tok = UnigramTokenizer(max_piece_len=6)
    tok.train(word_freqs, vocab_size=15, seed_method="all_substrings")
    pieces = tok.encode("testing")
    assert "".join(pieces) == "testing"

def test_encode_unseen_word_still_covered_by_fallback():
    word_freqs = {"hello": 10, "world": 8}
    tok = UnigramTokenizer(max_piece_len=6)
    tok.train(word_freqs, vocab_size=15, seed_method="all_substrings")
    pieces = tok.encode("wold")  # w,o,l,d seen, but "wold" no
    assert "".join(pieces) == "wold"

# real corpus
def test_train_on_real_corpus_sample():
    text = "the quick brown fox jumps over the lazy dog. " * 20
    pretok = PreTokenizer()
    word_freqs = pretok.build_word_freq(text)

    tok = UnigramTokenizer(max_piece_len=6)
    tok.train(word_freqs, vocab_size=50, seed_method="all_substrings")

    for word in word_freqs:
        pieces = tok.encode(word)
        assert "".join(pieces) == word