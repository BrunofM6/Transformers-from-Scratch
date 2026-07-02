import math
from collections import defaultdict

from bpe import BPTokenizer

class UnigramTokenizer:
    def __init__(self, max_piece_len: int = 8):
        self.vocab: dict[str, float] = {}
        self.max_piece_len = max_piece_len

    def _seed_vocab_all_substrings(self, word_freqs: dict[str, int]) -> dict[str, int]:
        candidates = defaultdict(int)
        for word, freq in word_freqs.items():
            n = len(word)
            for i in range(n):
                for j in range(i + 1, min(i + self.max_piece_len, n) + 1):
                    candidates[word[i:j]] += freq
        for word in word_freqs:
            for ch in word:
                candidates[ch] += 0
        return candidates
    
    def _seed_vocab_from_bpe(self, word_freqs: dict[str, int], bpe_vocab_size: int) -> dict[str, int]:
        corpus_text = " ".join(word for word, freq in word_freqs.items() for _ in range(freq))
        bpe = BPTokenizer()
        bpe.train(corpus_text, vocab_size=bpe_vocab_size)
        candidates = defaultdict(int)
        for word, freq in word_freqs.items():
            ids = bpe.encode(word)
            for id_ in ids:
                piece_str = bpe.vocab[id_].decode("utf-8", errors="ignore")
                if piece_str:
                    candidates[piece_str] += freq
        for word in word_freqs:
            for ch in word:
                candidates[ch] += 0
        return candidates
    
    def _viterbi_segment(self, word: str, vocab: dict[str, float]) -> tuple[list[str], float]:
        n = len(word)
        best_score = [-math.inf] * (n + 1)
        best_score[0] = 0.0
        backpointer = [None] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_piece_len), i):
                piece = word[j:i]
                if piece in vocab and best_score[j] + vocab[piece] > best_score[i]:
                    best_score[i] = best_score[j] + vocab[piece]
                    backpointer[i] = j
        pieces = []
        i = n
        while i > 0:
            j = backpointer[i]
            pieces.append(word[j:i])
            i = j
        pieces.reverse()
        return pieces, best_score[n]
    
    def train(self, word_freqs: dict[str, int], vocab_size: int, prune_frac: float = 0.2,
              seed_method: str = "bpe",  bpe_seed_multiplier: int = 4):
        if seed_method == "bpe":
            candidates = self._seed_vocab_from_bpe(word_freqs, bpe_vocab_size=vocab_size * bpe_seed_multiplier)
        elif seed_method == "all_substrings":
            candidates = self._seed_vocab_all_substrings(word_freqs)
        else:
            raise ValueError(f"Unknown seed_method: {seed_method}")
        total = sum(candidates.values())
        vocab = {p: math.log(c / total) for p, c in candidates.items() if c > 0}
        while len(vocab) > vocab_size:
            piece_counts = defaultdict(float)
            for word, freq in word_freqs.items():
                pieces, _ = self._viterbi_segment(word, vocab)
                for p in pieces:
                    piece_counts[p] += freq
            total = sum(piece_counts.values())
            vocab = {p: math.log(c / total) for p, c in piece_counts.items() if c > 0}
            removable = sorted(
                (p for p in vocab if len(p) > 1),
                key=lambda p: piece_counts[p]
            )
            if not removable:
                break
            excess = len(vocab) - vocab_size
            n_remove = min(max(1, int(len(vocab) * prune_frac)), excess, len(removable))
            for p in removable[:n_remove]:
                del vocab[p]
        self.vocab = vocab

    def encode(self, word: str) -> list[str]:
        pieces, _ = self._viterbi_segment(word, self.vocab)
        return pieces