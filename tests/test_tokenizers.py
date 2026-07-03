import os
import time
import numpy as np

from tokenizer.bpe import BPTokenizer
from tokenizer.unigram import UnigramTokenizer, UnigramWithIds

def load_and_split(text_path: str, val_fraction: float = 0.1):
    with open(text_path,  "r", encoding="utf-8") as f:
        text = f.read()
    split_idx = int(len(text) * (1 - val_fraction))
    return text[:split_idx], text[split_idx:]

def train_and_save_bpe(train_text: str, val_text: str, vocab_size: int, out_dir: str) -> BPTokenizer:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = BPTokenizer()
    t0 = time.time()
    tokenizer.train(train_text, vocab_size)
    train_time = time.time() - t0
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "val.bin"))
    print(f"[BPE]     train_time={train_time:.2f}s  "
          f"vocab_size={len(tokenizer.vocab)}  "
          f"train_tokens={len(train_ids)}  val_tokens={len(val_ids)}  "
          f"compression={len(train_text.encode('utf-8')) / len(train_ids):.3f} bytes/token")
    return tokenizer 

def train_and_save_unigram(train_text: str, val_text: str, vocab_size: int, out_dir: str,
                            seed_method: str = "bpe") -> UnigramWithIds:
    os.makedirs(out_dir, exist_ok=True)
    unigram = UnigramTokenizer(max_piece_len=8)
    t0 = time.time()
    unigram.train_from_text(train_text, vocab_size=vocab_size, seed_method=seed_method)
    train_time = time.time() - t0
    wrapped = UnigramWithIds(unigram)
    train_ids = wrapped.encode(train_text)
    val_ids = wrapped.encode(val_text)
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(out_dir, "val.bin"))
    print(f"[Unigram] train_time={train_time:.2f}s  "
          f"vocab_size={len(unigram.vocab)}  "
          f"train_tokens={len(train_ids)}  val_tokens={len(val_ids)}  "
          f"compression={len(train_text.encode('utf-8')) / len(train_ids):.3f} bytes/token")
    return wrapped

def compare(text_path: str, vocab_size: int = 512, val_fraction: float = 0.1,
            out_dir: str = "../data/compare"):
    train_text, val_text = load_and_split(text_path, val_fraction)
    print(f"Corpus: {text_path}")
    print(f"train chars={len(train_text)}  val chars={len(val_text)}  vocab_size target={vocab_size}\n")
    bpe = train_and_save_bpe(train_text, val_text, vocab_size, os.path.join(out_dir, "bpe"))
    unigram = train_and_save_unigram(train_text, val_text, vocab_size, os.path.join(out_dir, "unigram"))
    sample = val_text[:500]
    bpe_ids = bpe.encode(sample)
    bpe_decoded = bpe.decode(bpe_ids)
    uni_ids = unigram.encode(sample)
    uni_decoded = unigram.decode(uni_ids)
    print("\n--- Sample (first 500 chars from val) ---")
    print(f"Original:      {sample!r}")
    print(f"BPE tokens:     {len(bpe_ids)}   roundtrip_ok={bpe_decoded == sample}")
    print(f"Unigram tokens: {len(uni_ids)}   roundtrip_ok={uni_decoded == sample}")


if __name__ == "__main__":
    compare(
        text_path="../data/tiny-shakespeare.txt",
        vocab_size=512,
        val_fraction=0.1,
        out_dir="../data/compare",
    )