from tqdm import tqdm
import torch
from torch import nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import glob

torch.set_float32_matmul_precision('high')

# Each record: 44 uint8 features + 1 uint8 result (0=black wins, 1=draw, 2=white wins)
NB_FEATURES = 44
RECORD_SIZE = NB_FEATURES + 1  # 45 bytes


class FeaturesDataset(Dataset):
    def __init__(self, data_dir, max_records=500000000):
        files = sorted(glob.glob(data_dir + "/*.features.txt"))
        if not files:
            raise FileNotFoundError(f"No .features.txt files found in {data_dir}")

        self.mmaps = []
        self.cumulative = []  # cumulative[i] = total records before file i
        total = 0
        for path in files:
            size = os.path.getsize(path)
            nb_records = size // RECORD_SIZE
            print(f"{path}: {nb_records} records", flush=True)
            mm = np.memmap(path, dtype=np.uint8, mode='r', shape=(nb_records, RECORD_SIZE))
            self.cumulative.append(total)
            self.mmaps.append(mm)
            total += nb_records
            if total > max_records: break

        self.size = total
        print(f"Total: {total} records across {len(files)} files", flush=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Binary search to find which file contains this index
        lo, hi = 0, len(self.cumulative)
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.cumulative[mid] <= idx:
                lo = mid
            else:
                hi = mid
        local_idx = idx - self.cumulative[lo]
        record = self.mmaps[lo][local_idx]
        features = torch.tensor(record[:NB_FEATURES], dtype=torch.float32) / 255.0
        label = int(record[NB_FEATURES])
        return features, torch.tensor(label, dtype=torch.long)


class Net(nn.Module):
    """
    Direct-from-bytes classifier: takes features as uint8/255 and produces
    (black, draw, white) logits. No BatchNorm — the C++ first layer in ffnn.h
    reads raw byte/255 and any pre-fc1 normalization would need to be folded
    into fc1's int8 weights, where low-variance features blow the ±127/64
    clipping budget. Training directly on byte/255 keeps Net.clip() honest:
    the clamped weights are exactly what gets quantized.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NB_FEATURES, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc2(x), min=0.0, max=1.0)
        return self.fc3(x) #nn.functional.softmax(self.fc3(x), dim=1)

    def clip(self):
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.weight.data.clamp_(-127/64, 127/64)
            fc.bias.data.clamp_(-127/64, 127/64)


def save_quantized(net, filename, scale=64):
    """
    Serializes a trained Net into the integer text format consumed by
    ffnn.h's FFNN<I, H1, H2, O> class (see ffnn_detail::read_matrix/read_bias).

    Quantization scheme (must match ffnn.h):
      - weights  : int8, W_q  = round(scale * W)               clamped to [-127, 127]
      - fc1 bias : int16, b_q = round(scale * b)               (first-layer path
                   adds bias after the float /255, so scale is just `scale`)
      - fc2/fc3  : int16, b_q = round(scale^2 * b)             (integer dot product
         bias     accumulates at scale^2; bias must live at the same scale before >>6)

    File layout (whitespace-separated, one section per layer in fc1→fc2→fc3 order):
        W
        <m> <n>
        <m*n values>
        B
        <n>
        <n values>

    The file stores the *unpadded* input dimension (e.g. 44 for fc1); padding to
    I_PADDED is done at load time by the C++ FFNN constructor.

    Warns on saturation: if a weight exceeds ±127 or a bias exceeds int16 range,
    the model was not clipped tightly enough during training (see Net.clip()).
    """
    net.train(False)

    w_int8_max = 127
    b_int16_max = 32767

    def _quantize_weight(weight, s, layer_name):
        q = torch.round(s * weight.detach().to(torch.float64))
        max_abs = int(q.abs().max().item()) if q.numel() > 0 else 0
        if max_abs > w_int8_max:
            print(
                f"warning: {layer_name} weights saturate int8 "
                f"(max |W*{s}| = {max_abs} > {w_int8_max}); "
                f"consider tightening Net.clip() or retraining",
                flush=True,
            )
        return q.clamp_(-w_int8_max, w_int8_max).to(torch.int64)

    def _quantize_bias(bias, s, layer_name):
        q = torch.round(s * bias.detach().to(torch.float64))
        max_abs = int(q.abs().max().item()) if q.numel() > 0 else 0
        if max_abs > b_int16_max:
            print(
                f"warning: {layer_name} bias saturates int16 "
                f"(max |b*{s}| = {max_abs} > {b_int16_max})",
                flush=True,
            )
        return q.clamp_(-b_int16_max - 1, b_int16_max).to(torch.int64)

    def _write_matrix(f, W_q):
        m, n = W_q.shape
        f.write(f"W\n{m} {n}\n")
        for i in range(m):
            f.write(" ".join(str(int(v)) for v in W_q[i].tolist()))
            f.write("\n")

    def _write_bias(f, b_q):
        n = b_q.shape[0]
        f.write(f"B\n{n}\n")
        f.write(" ".join(str(int(v)) for v in b_q.tolist()))
        f.write("\n")

    fc1_w = _quantize_weight(net.fc1.weight, scale,         "fc1")
    fc1_b = _quantize_bias  (net.fc1.bias,   scale,         "fc1")  # scale^1
    fc2_w = _quantize_weight(net.fc2.weight, scale,         "fc2")
    fc2_b = _quantize_bias  (net.fc2.bias,   scale * scale, "fc2")  # scale^2
    fc3_w = _quantize_weight(net.fc3.weight, scale,         "fc3")
    fc3_b = _quantize_bias  (net.fc3.bias,   scale * scale, "fc3")  # scale^2

    with open(filename, "w") as f:
        _write_matrix(f, fc1_w)
        _write_bias  (f, fc1_b)
        _write_matrix(f, fc2_w)
        _write_bias  (f, fc2_b)
        _write_matrix(f, fc3_w)
        _write_bias  (f, fc3_b)

    print(f"quantized model written to {filename} (scale={scale})", flush=True)


NB_EPOCHS = 100
MODEL_PATH = "/mnt/"
MODEL_NAME = "features_128x64x3"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
DATA_DIR = "/mnt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dataloader(trainset, valset, batch_size):
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True#, prefetch_factor=2
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    return train_loader, val_loader


class Trainer:
    def __init__(self, model, train_loader, val_loader, save_every=1):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_every = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.model = torch.compile(self.model)
            self.stream = torch.cuda.Stream()

    def _save_checkpoint(self, epoch):
        state_dict = self.model._orig_mod.state_dict() if hasattr(self.model, '_orig_mod') else self.model.state_dict()
        torch.save(state_dict, f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _to_device(self, X, y):
        if self.device == "cuda":
            with torch.cuda.stream(self.stream):
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.stream)
        else:
            X, y = X.to(self.device), y.to(self.device)
        return X, y

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in tqdm(self.train_loader):
            n += len(X)
            X, y = self._to_device(X, y)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy += sum(torch.argmax(logits, dim=1) == y).item()
            inner = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            inner.clip()
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        print('epoch {} train loss: {} train accuracy: {} lr: {:.6f}'.format(
            epoch + 1, running_loss / n, accuracy / n, current_lr), flush=True)

    def _validate(self, epoch):
        self.model.eval()
        n = 0
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for (X, y) in tqdm(self.val_loader):
                n += len(X)
                X, y = self._to_device(X, y)
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                val_loss += loss.item()
                val_accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        print('epoch {} val loss: {} val accuracy: {}'.format(
            epoch + 1, val_loss / n, val_accuracy / n), flush=True)
        self.model.train()

    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self._run_epoch(epoch)
            self._validate(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    dataset = FeaturesDataset(DATA_DIR)
    print(len(dataset), flush=True)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])
    print(f'Train size: {train_size}, Val size: {val_size}', flush=True)

    train_loader, val_loader = dataloader(trainset, valset, batch_size=512 * 2)

    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL, map_location="cpu"))
    print(net, flush=True)

    # save_quantized(net, f"{MODEL_PATH}{MODEL_NAME}.quantized.txt")

    trainer = Trainer(net, train_loader, val_loader)
    trainer.train(NB_EPOCHS)
