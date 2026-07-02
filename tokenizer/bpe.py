# Corpus -> Tokens

class BPTokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    @staticmethod
    def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        # Count adjacent tokens
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        # Replace all 'pair'occurences for 'new_id'
        new_ids = []
        i, l = 0, len(ids)
        while i < l:
            if i < l - 1 and pair[0] == ids[i] and pair[1] == ids[i + 1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train(self, text: str, vocab_size: int) -> dict[tuple[int, int], int]:
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        self.merges = {}
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
        self._build_vocab()

    def _build_vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            ids = self._merge(ids, pair, self.merges[pair])
        return ids
        
    def decode(self, ids: list[int]) -> list[str]:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")