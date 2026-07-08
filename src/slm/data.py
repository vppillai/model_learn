import torch
from tokenizers import Tokenizer

EOT_ID = 0


def tokenize_texts(tok: Tokenizer, texts: list[str]) -> list[int]:
    stream: list[int] = []
    for t in texts:
        stream.extend(tok.encode(t).ids)
        stream.append(EOT_ID)
    return stream


def get_batch(data: list[int], batch_size: int, context_len: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(data, dtype=torch.long)
    max_start = len(t) - context_len - 1
    assert max_start > 0, "not enough tokens for one context window"
    starts = torch.randint(0, max_start, (batch_size,), generator=g)
    x = torch.stack([t[s : s + context_len] for s in starts])
    y = torch.stack([t[s + 1 : s + 1 + context_len] for s in starts])
    return x, y


def load_tinystories(split: str = "train", limit: int | None = None) -> list[str]:
    """Real dataset loader for training runs (not used in unit tests).

    When `limit` is set (e.g. the local toy run), stream the dataset and take
    only the first `limit` stories, so we avoid downloading the full ~1.9GB
    corpus just to use a few thousand. When `limit` is None (the Colab `small`
    run), download the whole split.
    """
    from datasets import load_dataset
    if limit is not None:
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        out = []
        for row in ds:
            out.append(row["text"])
            if len(out) >= limit:
                break
        return out
    ds = load_dataset("roneneldan/TinyStories", split=split)
    return [row["text"] for row in ds]
