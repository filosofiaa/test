"""CIFAR-10 helpers shared by notebooks 03-05: data, the small ResNet of the v0.x notebooks, train and evaluate."""
import os
import time

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = os.environ.get("VVTK_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def _cifar_arrays(split):
    """CIFAR-10 as uint8 (N, 32, 32, 3) + int64 labels, from the Hugging Face parquet mirror (cached as .npz).
    The original Toronto server is often very slow; the images and labels are the same."""
    import numpy as np
    cache = os.path.join(DATA, f"cifar10_{split}.npz")
    if not os.path.exists(cache):
        import io, urllib.request
        import pandas as pd
        from PIL import Image
        pq = os.path.join(DATA, "hf_cifar10", f"{split}.parquet")
        if not os.path.exists(pq):
            os.makedirs(os.path.dirname(pq), exist_ok=True)
            urllib.request.urlretrieve(HF_URL.format(split=split), pq)
        df = pd.read_parquet(pq)
        x = np.stack([np.array(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in df["img"]])
        np.savez(cache, x=x, y=df["label"].to_numpy().astype(np.int64))
    d = np.load(cache)
    return d["x"], d["y"]


class CIFAR10(torch.utils.data.Dataset):
    """Tensor-based CIFAR-10: random crop (pad 4) + horizontal flip for training, normalisation always."""

    def __init__(self, split, augment):
        x, y = _cifar_arrays(split)
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()     # uint8 (N, 3, 32, 32)
        self.y = torch.from_numpy(y)
        self.augment = augment
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].float() / 255.0
        if self.augment:
            img = nn.functional.pad(img, (4, 4, 4, 4))
            dx, dy = torch.randint(0, 9, (2,)).tolist()
            img = img[:, dy:dy + 32, dx:dx + 32]
            if torch.rand(()) < 0.5:
                img = img.flip(-1)
        return (img - self.mean) / self.std, self.y[i]


def loaders(batch_size=128, workers=4):
    tr, te = CIFAR10("train", augment=True), CIFAR10("test", augment=False)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (torch.utils.data.DataLoader(tr, batch_size, shuffle=True, drop_last=True, **kw),
            torch.utils.data.DataLoader(te, 512, shuffle=False, **kw))


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(out)) + self.shortcut(x)


class SmallResNet(nn.Module):
    """The model of the v0.x notebooks (4.8 M parameters)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = ResidualBlock(64, 128)
        self.layer2 = ResidualBlock(128, 256)
        self.layer3 = ResidualBlock(256, 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(out, 1), 1))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_epoch(model, loader, opt, sched=None, step_hook=None):
    """One epoch of cross-entropy training. step_hook(step) is called before each step (e.g. to anneal T)."""
    model.train()
    crit = nn.CrossEntropyLoss()
    t0, n, tot = time.time(), 0, 0.0
    for x, y in loader:
        if step_hook is not None:
            step_hook()
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item(); n += 1
    return tot / n, time.time() - t0


def pretrained_fp32(train_loader, test_loader, epochs=5, path=None, seed=0):
    """Train (or load) the float baseline: Adam 1e-3, `epochs` epochs, as in the v0.x notebooks."""
    path = path or os.path.join(OUT, f"cifar_fp32_e{epochs}_s{seed}.pth")
    torch.manual_seed(seed)
    model = SmallResNet().to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"loaded {path}")
    else:
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        for ep in range(epochs):
            loss, dt = train_epoch(model, train_loader, opt)
            print(f"fp32 epoch {ep + 1}/{epochs}  loss {loss:.3f}  test acc {evaluate(model, test_loader):.2f}%  ({dt:.0f}s)")
        torch.save(model.state_dict(), path)
    return model
