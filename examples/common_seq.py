"""
Sequence models for notebooks 06 (tiny character LM) and 07 (TIMIT phoneme CTC decoder).

The transformer block is written with plain nn.Linear layers (q, k, v, o, ff1, ff2), so every matrix is
quantizable by VVTK_QModel(module_types=[nn.Linear]). nn.MultiheadAttention would hide its input
projection in a raw Parameter.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("VVTK_DATA", os.path.join(HERE, "data"))
OUT = os.path.join(HERE, "out")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------------------
# Transformer pieces
# ----------------------------------------------------------------------------------------
def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, T, Dh)."""
    T, d = x.shape[-2], x.shape[-1]
    f = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, ff_mult=4, dropout=0.1, causal=False):
        super().__init__()
        self.h, self.causal = heads, causal
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ff1, self.ff2 = nn.Linear(d, ff_mult * d), nn.Linear(ff_mult * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        B, T, D = x.shape
        h = self.n1(x)
        q, k, v = (f(h).view(B, T, self.h, D // self.h).transpose(1, 2) for f in (self.q, self.k, self.v))
        q, k = rope(q), rope(k)
        mask = None
        if pad_mask is not None:                                   # True = padding
            mask = ~pad_mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.causal and mask is None,
                                           dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.n2(x)))))


# ----------------------------------------------------------------------------------------
# Tiny character LM (notebook 06)
# ----------------------------------------------------------------------------------------
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def tiny_shakespeare():
    """Returns (train ids, val ids, itos). ~1.1 M characters, 65 symbols, 90/10 split."""
    path = os.path.join(DATA, "tinyshakespeare.txt")
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text = open(path, encoding="utf-8").read()
    itos = sorted(set(text))
    stoi = {c: i for i, c in enumerate(itos)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], itos


class CharLM(nn.Module):
    def __init__(self, vocab, d=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, dropout=dropout, causal=True) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def lm_batch(ids, batch, ctx, gen=None):
    i = torch.randint(0, len(ids) - ctx - 1, (batch,), generator=gen)
    x = torch.stack([ids[j:j + ctx] for j in i.tolist()])
    y = torch.stack([ids[j + 1:j + ctx + 1] for j in i.tolist()])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def lm_eval(model, ids, ctx=256, batches=40, batch=64):
    """Mean validation cross-entropy (nats/char) on fixed random windows (same windows for every model)."""
    model.eval()
    g = torch.Generator().manual_seed(1234)
    tot = 0.0
    for _ in range(batches):
        x, y = lm_batch(ids, batch, ctx, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return tot / batches


@torch.no_grad()
def lm_sample(model, itos, prompt="ROMEO:", n=200, temp=0.8, seed=0):
    model.eval()
    stoi = {c: i for i, c in enumerate(itos)}
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.tensor([[stoi[c] for c in prompt]], device=DEVICE)
    for _ in range(n):
        p = F.softmax(model(x[:, -256:])[:, -1] / temp, -1)
        x = torch.cat([x, torch.multinomial(p, 1, generator=g)], 1)
    return "".join(itos[i] for i in x[0].tolist())


# ----------------------------------------------------------------------------------------
# TIMIT phoneme recognizer (notebook 07)
# ----------------------------------------------------------------------------------------
TIMIT_ROOT = os.environ.get("TIMIT_ROOT", "/extra/cadrete/database/data/asr_en")


def timit_phones():
    """ph.list: SIL NOISE MUSIC AA AE ... ZH (42 symbols). CTC vocab: blank + the 39 phones."""
    ph = open(os.path.join(TIMIT_ROOT, "lst", "ph.list")).read().split()
    phones = ph[3:]
    return ph, ["<b>"] + phones


def timit_features(n_mels=40, cache=None):
    """
    Log-mel features (25 ms / 10 ms, 40 mel) and CTC phone sequences for all TIMIT utterances.
    Frame labels (10 ms) are collapsed to phone sequences; SIL/NOISE/MUSIC are dropped.
    Cached in data/ after the first run (reads each wav once, sequentially).
    """
    import soundfile as sf
    import torchaudio
    cache = cache or os.path.join(DATA, f"timit_mel{n_mels}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    names = [l.split()[0].split("/")[-1] for l in open(os.path.join(TIMIT_ROOT, "lst", "full-timit.file.list"))]
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, win_length=400, hop_length=160, n_mels=n_mels)
    feats, labels, keep = [], [], []
    for i, n in enumerate(names):
        wav, sr = sf.read(os.path.join(TIMIT_ROOT, "wav", "full-timit", n + ".wav"), dtype="float32")
        f = torch.log(mel(torch.from_numpy(wav)) + 1e-6).T                     # (frames, n_mels)
        lab = np.load(os.path.join(TIMIT_ROOT, "ilabel", "full-timit", n + ".npy")).astype(np.int64)
        seq = [int(p) for j, p in enumerate(lab) if (j == 0 or p != lab[j - 1]) and p >= 3]
        feats.append(f.half()); labels.append(torch.tensor([p - 2 for p in seq]))   # 3..41 -> 1..39 (0 = blank)
        keep.append(n)
        if (i + 1) % 1000 == 0:
            print(f"  features: {i + 1}/{len(names)}")
    d = dict(names=keep, feats=feats, labels=labels)
    torch.save(d, cache)
    return d


def timit_split(d, test_frac=0.1, seed=0):
    """Speaker-disjoint split (not the official core test set); SA sentences removed, as is customary."""
    spk = sorted({n.split("-")[0] for n in d["names"]})
    rng = np.random.default_rng(seed)
    test_spk = set(rng.choice(spk, int(len(spk) * test_frac), replace=False))
    tr, te = [], []
    for i, n in enumerate(d["names"]):
        if n.split("-")[1].startswith("sa"):
            continue
        (te if n.split("-")[0] in test_spk else tr).append(i)
    return tr, te


class PhoneCTC(nn.Module):
    """Conv1d front end (x2 subsampling) + transformer encoder + linear CTC head."""

    def __init__(self, n_in=40, n_out=40, d=192, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_in, d, 5, padding=2)
        self.conv2 = nn.Conv1d(d, d, 5, stride=2, padding=2)
        self.blocks = nn.ModuleList(Block(d, heads, dropout=dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

    def forward(self, x, lens):
        x = F.gelu(self.conv2(F.gelu(self.conv1(x.transpose(1, 2))))).transpose(1, 2)
        lens = (lens + 1) // 2
        pad = torch.arange(x.shape[1], device=x.device)[None] >= lens[:, None]
        for b in self.blocks:
            x = b(x, pad)
        return self.head(self.norm(x)), lens


def ctc_batches(d, idx, batch=32, shuffle=True, seed=0, mean=None, std=None):
    """Length-sorted buckets of utterances -> padded (x, lens, y, ylens) on DEVICE."""
    idx = sorted(idx, key=lambda i: len(d["feats"][i]))
    chunks = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    if shuffle:
        np.random.default_rng(seed).shuffle(chunks)
    for c in chunks:
        fs = [d["feats"][i].to(DEVICE).float() for i in c]
        if mean is not None:
            fs = [(f - mean) / std for f in fs]
        lens = torch.tensor([len(f) for f in fs])
        x = nn.utils.rnn.pad_sequence(fs, batch_first=True)
        ys = [d["labels"][i] for i in c]
        yl = torch.tensor([len(y) for y in ys])
        yield x.to(DEVICE), lens.to(DEVICE), torch.cat(ys).to(DEVICE), yl.to(DEVICE)


def edit_distance(a, b):
    """Levenshtein distance between two int sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


@torch.no_grad()
def ctc_per(model, d, idx, mean, std):
    """Phone error rate (%) with greedy CTC decoding."""
    model.eval()
    err = tot = 0
    for x, lens, y, yl in ctc_batches(d, idx, batch=64, shuffle=False, mean=mean, std=std):
        logits, ol = model(x, lens)
        best = logits.argmax(-1).cpu()
        ys = torch.split(y.cpu(), yl.cpu().tolist())
        for k in range(len(ys)):
            p = best[k, :ol[k]].tolist()
            hyp = [t for j, t in enumerate(p) if t != 0 and (j == 0 or t != p[j - 1])]
            err += edit_distance(hyp, ys[k].tolist()); tot += len(ys[k])
    return 100.0 * err / tot


def ctc_train_epoch(model, d, idx, opt, mean, std, sched=None, step_hook=None, seed=0, batch=32):
    model.train()
    tot = n = 0
    for x, lens, y, yl in ctc_batches(d, idx, batch=batch, seed=seed, mean=mean, std=std):
        if step_hook is not None:
            step_hook()
        logits, ol = model(x, lens)
        lp = F.log_softmax(logits.float(), -1).transpose(0, 1)
        loss = F.ctc_loss(lp, y, ol, yl, blank=0, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item(); n += 1
    return tot / n


def cosine(opt, total, warmup=0):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
