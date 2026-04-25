from tqdm import tqdm
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import glob

torch.set_float32_matmul_precision('high')

# Each record: 57 uint8 features + 1 uint8 result (0=black wins, 1=draw, 2=white wins)
NB_FEATURES = 57
RECORD_SIZE = NB_FEATURES + 1  # 45 bytes


class FeaturesDataset(Dataset):
    def __init__(self, data_dir, max_records=50000000):
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


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bn = nn.BatchNorm1d(NB_FEATURES)
#         self.fc1 = nn.Linear(NB_FEATURES, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 3)

#     def forward(self, x):
#         x = self.bn(x)
#         x = torch.clamp(self.fc1(x), min=0.0, max=1.0)
#         x = torch.clamp(self.fc2(x), min=0.0, max=1.0)
#         return self.fc3(x)

#     def clip(self):
#         for fc in [self.fc1, self.fc2, self.fc3]:
#             fc.weight.data.clamp_(-127/64, 127/64)
#             fc.bias.data.clamp_(-127/64, 127/64)


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
        return self.fc3(x)#nn.functional.softmax(self.fc3(x), dim=1)

    def clip(self):
        for fc in [self.fc1, self.fc2, self.fc3]:
            fc.weight.data.clamp_(-127/64, 127/64)
            fc.bias.data.clamp_(-127/64, 127/64)


class NetNoBN(nn.Module):
    """Same architecture as Net but without BatchNorm — for C++ inference."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NB_FEATURES, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), min=0.0, max=1.0)
        x = torch.clamp(self.fc2(x), min=0.0, max=1.0)
        return self.fc3(x)


def fold_batchnorm(bn, fc):
    """
    Folds a BatchNorm1d layer that precedes a Linear layer into that Linear
    layer in-place, using the BN's frozen running statistics.

    After calling this, fc subsumes the BN transform:
      fc(bn(x)) == fc_new(x)

    Must be called with the model in eval mode (model.train(False)).
    """
    scale = bn.weight / (bn.running_var + bn.eps).sqrt()  # γ / √(σ²+ε)
    shift = bn.bias - bn.running_mean * scale              # β - μ·scale
    # Bias update must use the original weights, before scaling
    fc.bias.data  += fc.weight.data @ shift
    fc.weight.data *= scale.unsqueeze(0)


def export_net(net):
    """
    Returns a NetNoBN with BN folded into fc1, ready for C++ export.
    Does not modify the original net.
    """
    net.train(False)
    exported = NetNoBN()
    for attr in ['fc1', 'fc2', 'fc3']:
        src = getattr(net, attr)
        dst = getattr(exported, attr)
        dst.weight.data = src.weight.data.clone()
        dst.bias.data   = src.bias.data.clone()
    fold_batchnorm(net.bn, exported.fc1)
    return exported


def save_quantized(net, filename, scale=64):
    """
    Saves the weights of a NetNoBN (BN already folded) in quantized integer form,
    mirroring NNUE::save_quantized in nnue.cpp.

    Format for each layer:
      W\nM\nN\n<M*N values, one per line>
      B\nN\n<N values, one per line>

    Weight scale : scale (64 by default).
    Bias scale   : scale for fc1, scale^2 for fc2 and fc3
                   (deeper biases must match the accumulated scale after W_q * x_q).

    fc1 weights are written in row-major order of (out, in), which corresponds to
    the TRANSPOSE=true layout read by the C++ load function.
    """
    net.train(False)

    def write_matrix(f, weight, s):
        m, n = weight.shape
        f.write(f"W\n{m}\n{n}\n")
        for i in range(m):
            for j in range(n):
                f.write(f"{int(round(s * weight[i, j].item()))}\n")

    def write_bias(f, bias, s):
        n = bias.shape[0]
        f.write(f"B\n{n}\n")
        for i in range(n):
            f.write(f"{int(round(s * bias[i].item()))}\n")

    with open(filename, "w") as f:
        write_matrix(f, net.fc1.weight, scale)
        write_bias(f, net.fc1.bias, scale)
        write_matrix(f, net.fc2.weight, scale)
        write_bias(f, net.fc2.bias, scale * scale)
        write_matrix(f, net.fc3.weight, scale)
        write_bias(f, net.fc3.bias, scale * scale)


NB_EPOCHS = 200
MODEL_PATH = "/mnt/"
MODEL_NAME = "features_57x128x64x3"
LAST_MODEL = f"{MODEL_PATH}{MODEL_NAME}.pt"
DATA_DIR = "/mnt"


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65433"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val = DistributedSampler(valset, shuffle=False)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train,
        num_workers=0, pin_memory=True#, prefetch_factor=2
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, sampler=sampler_val,
        num_workers=0, pin_memory=True
    )
    return train_loader, sampler_train, val_loader, sampler_val


class TrainerDDP:
    def __init__(self, gpu_id, model, train_loader, sampler_train, val_loader, sampler_val, save_every=1):
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.sampler_train = sampler_train
        self.val_loader = val_loader
        self.sampler_val = sampler_val
        self.save_every = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        self.loss_fn = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.model = torch.compile(self.model)
        self.stream = torch.cuda.Stream(device=gpu_id)

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in tqdm(self.train_loader):
            n += len(X)
            with torch.cuda.stream(self.stream):
                X = X.to(self.gpu_id, non_blocking=True)
                y = y.to(self.gpu_id, non_blocking=True)
            torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy += sum(torch.argmax(logits, dim=1) == y).item()
            self.model.module.clip()
        self.scheduler.step()
        if self.gpu_id == 0:
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
                with torch.cuda.stream(self.stream):
                    X = X.to(self.gpu_id, non_blocking=True)
                    y = y.to(self.gpu_id, non_blocking=True)
                torch.cuda.current_stream(self.gpu_id).wait_stream(self.stream)
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                val_loss += loss.item()
                val_accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        if self.gpu_id == 0:
            print('epoch {} val loss: {} val accuracy: {}'.format(
                epoch + 1, val_loss / n, val_accuracy / n), flush=True)
        self.model.train()

    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self.sampler_train.set_epoch(epoch)
            self._run_epoch(epoch)
            self._validate(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)


def main(rank, world_size, batch_size, dataset):
    ddp_setup(rank, world_size)
    if rank == 0:
        print(len(dataset), flush=True)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    if rank == 0:
        print(f'Train size: {train_size}, Val size: {val_size}', flush=True)

    train_loader, sampler_train, val_loader, sampler_val = dataloader_ddp(trainset, valset, batch_size)
    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        print(net, flush=True)
    trainer = TrainerDDP(rank, net, train_loader, sampler_train, val_loader, sampler_val)
    trainer.train(NB_EPOCHS)
    destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    dataset = FeaturesDataset(DATA_DIR)
    mp.spawn(main, args=(world_size, 256, dataset), nprocs=world_size)
