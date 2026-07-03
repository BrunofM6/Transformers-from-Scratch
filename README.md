# Transformers from Scratch
The following project describes a transformers setup from scratch, including the various tools associated, tested. Developed by Bruno Moreira.

It includes ([Implemented]):
- Pre-Tokenizer [X]
- BPE [X] -> all 256 utf-8 chars guaranteed encoding/decoding
- Unigram [X] -> only decodes strings when train and test have the same SEEN chars

    $P(x) = ∏ P(x_i)$

Their comparison:

```
Corpus: data/tiny-shakespeare.txt
train chars=1003853  val chars=111540  vocab_size target=512

[BPE]     train_time=18.87s  vocab_size=512  train_tokens=511068  val_tokens=57518  compression=1.964 bytes/token
[Unigram] train_time=251.29s  vocab_size=512  train_tokens=458619  val_tokens=53228  compression=2.189 bytes/token

--- Sample (first 500 chars from val) ---
BPE tokens:     280   roundtrip_ok=True
Unigram tokens: 268   roundtrip_ok=True
```

- Transformer Architecture []