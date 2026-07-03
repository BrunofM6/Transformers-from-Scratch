from collections import defaultdict
from tokenizer.bpe import BPTokenizer
from tokenizer.pre_tokenize import PreTokenizer

def _seed_vocab_from_bpe_with_flag(word_freqs: dict[str, int], bpe_vocab_size: int, pretokenize: bool) -> dict[str, int]:
    corpus_text = " ".join(word for word, freq in word_freqs.items() for _ in range(freq))
    bpe = BPTokenizer(pretokenize=pretokenize)
    bpe.train(corpus_text, vocab_size=bpe_vocab_size)
    candidates = defaultdict(int)
    for word, freq in word_freqs.items():
        ids = bpe.encode(word)
        for id_ in ids:
            piece_str = bpe.vocab[id_].decode("utf-8", errors="ignore")
            if piece_str:
                candidates[piece_str] += freq
    return candidates, bpe

def test_pretokenize_seed_ablation():
    word_freqs = {
        "the": 200,
        "cat": 20,
        "sat": 15,
        "mat": 10,
    }
    bpe_vocab_size = 260
    print("\n" + "=" * 70)
    print("ABLATION: pretokenize=True vs False no BPE usado como seed do Unigram")
    print("=" * 70)
    print(f"word_freqs: {word_freqs}")
    print(f"bpe_vocab_size: {bpe_vocab_size} (apenas {bpe_vocab_size - 256} merges disponíveis)\n")
    candidates_no_pretok, bpe_no_pretok = _seed_vocab_from_bpe_with_flag(
        word_freqs, bpe_vocab_size, pretokenize=False
    )
    print("--- pretokenize=False (BPE vê o corpus_text como um blob contínuo) ---")
    print("Merges aprendidos (ordem):")
    for pair, new_id in bpe_no_pretok.merges.items():
        piece = bpe_no_pretok.vocab[new_id]
        print(f"  merge #{new_id - 256}: {pair} -> {piece!r}")
    print(f"Candidatos extraídos: {dict(candidates_no_pretok)}\n")
    candidates_pretok, bpe_pretok = _seed_vocab_from_bpe_with_flag(
        word_freqs, bpe_vocab_size, pretokenize=True
    )
    print("--- pretokenize=True (BPE respeita fronteiras de chunk) ---")
    print("Merges aprendidos (ordem):")
    for pair, new_id in bpe_pretok.merges.items():
        piece = bpe_pretok.vocab[new_id]
        print(f"  merge #{new_id - 256}: {pair} -> {piece!r}")
    print(f"Candidatos extraídos: {dict(candidates_pretok)}\n")