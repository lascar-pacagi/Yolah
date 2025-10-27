import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu, softmax
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import sys
sys.path.append("../server")
from yolah import Yolah, Move, Square
import itertools

def bit_not(n, numbits=64):
    return (1 << numbits) - 1 - n

def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0]*(64 - len(b)) + b

class GameDataset(Dataset):
    def __init__(self, games_dir):
        self.inputs = []
        self.outputs = []
        self.size = 0
        for filename in glob.glob(games_dir + "/games*"):
            print(filename)            
            with open(filename, 'rb') as f:
                data = f.read()
                print(len(data))
                counter = 1
                while data:
                    if counter % 1000 == 0: 
                        print(counter)
                    counter += 1
                    #if counter == 3000: break
                    #print(len(data))
                    nb_moves = int(data[0])
                    d = data[:nb_moves * 2 + 4]
                    nb_random_moves = int(d[1])
                    #print(nb_moves, ' ', nb_random_moves)
                    if nb_random_moves >= nb_moves: continue
                    black_score = int(d[-2])
                    white_score = int(d[-1])
                    #print(black_score, ' ', white_score)
                    res = 0
                    if black_score == white_score: res = 1
                    if white_score > black_score: res = 2
                    res = torch.tensor(res, dtype=torch.uint8)
                    yolah = Yolah()
                    for sq1, sq2 in zip(d[2:2+2*nb_random_moves:2], data[3:2+2*nb_random_moves:2]):
                        m = Move(Square(int(sq1)), Square(int(sq2)))
                        yolah.play(m)
                    # print(yolah)
                    # print(self.encode_yolah(yolah))
                    # print(res)
                    # input()
                    self.inputs.append(self.encode_yolah(yolah))
                    self.outputs.append(res)
                    self.size += 1
                    for sq1, sq2 in zip(d[2+2*nb_random_moves:2+2*nb_moves:2], d[3+2*nb_random_moves:2+2*nb_moves:2]):
                        m = Move(Square(int(sq1)), Square(int(sq2)))
                        yolah.play(m)
                        self.inputs.append(self.encode_yolah(yolah))
                        self.outputs.append(res)
                        self.size += 1
                        # print(yolah)
                        # print(self.encode_yolah(yolah))
                        # print(res)
                        # input()
                    data = data[2+2*nb_moves+2:]
                print(sys.getsizeof(self.inputs) / (1024 * 1024))

    @staticmethod
    def encode_yolah(yolah):
        black_list = bitboard64_to_list(yolah.black)
        white_list = bitboard64_to_list(yolah.white)
        one_hot = np.array(list(itertools.chain.from_iterable([
                        black_list, 
                        white_list, 
                        bitboard64_to_list(yolah.empty),                    
                        [Yolah.WHITE_PLAYER if yolah.nb_plies() & 1 else Yolah.BLACK_PLAYER]])))
        indices = np.nonzero(one_hot)[0]
        return torch.tensor(indices, dtype=torch.uint8)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.inputs[idx], self.outputs[idx])

# black positions + white positions + empty positions + turn 
INPUT_SIZE = 64 + 64 + 64 + 1

def custom_collate(batch, vocab_size=INPUT_SIZE):
    """
    Convert batch of sparse indices to dense one-hot tensors.
    Each sample is a tuple of (sparse_indices, target).
    """
    inputs, targets = zip(*batch)
    batch_size = len(inputs)
    device = inputs[0].device

    # Create dense tensor for the batch
    dense = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device=device)

    # Fill in the active features for each sample in the batch
    for i, sparse_indices in enumerate(inputs):
        # Convert uint8 to long for indexing
        indices = sparse_indices.long()
        dense[i, indices] = 1.0

    return dense, torch.stack(targets)

class Net(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, l1_size=2048, l2_size=128, l3_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc2(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc3(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return self.fc4(x)#softmax(self.fc4(x), dim=1)#

NB_EPOCHS=5
#MODEL_PATH="/mnt/"
MODEL_PATH="./"
MODEL_NAME="nnue_2048x128x64x3_2"
LAST_MODEL=f"{MODEL_PATH}{MODEL_NAME}.pt"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def dataloader_ddp(trainset, valset, batch_size):
    sampler_train = DistributedSampler(trainset)
    sampler_val = DistributedSampler(valset, shuffle=False)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train, num_workers=0, pin_memory=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, sampler=sampler_val, num_workers=0, pin_memory=True, collate_fn=custom_collate
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
        self.loss_fn = torch.nn.CrossEntropyLoss()
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.model = torch.compile(self.model, mode="max-autotune")

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in self.train_loader:
            n += len(X)
            X = X.to(self.gpu_id)
            self.optimizer.zero_grad()
            logits = self.model(X)
            y = y.to(self.gpu_id, dtype=torch.long)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        if self.gpu_id == 0:
            print('epoch {} train loss: {} train accuracy: {}'.format(epoch + 1, running_loss / n, accuracy / n), flush=True)

    def _validate(self, epoch):
        self.model.eval()
        n = 0
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for (X, y) in self.val_loader:
                n += len(X)
                X = X.to(self.gpu_id)
                logits = self.model(X)
                y = y.to(self.gpu_id, dtype=torch.long)
                loss = self.loss_fn(logits, y)
                val_loss += loss.item()
                val_accuracy += sum(torch.argmax(logits, dim=1) == y).item()
        if self.gpu_id == 0:
            print('epoch {} val loss: {} val accuracy: {}'.format(epoch + 1, val_loss / n, val_accuracy / n), flush=True)
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

def main(rank, world_size, batch_size):
    ddp_setup(rank, world_size)
    dataset = GameDataset("data")
    print(rank)
    if rank == 0:
        print(len(dataset), flush=True)

    # Split dataset into train and validation (90/10 split)
    train_size = int(0.9 * len(dataset))
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
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    mp.spawn(main, args=(world_size, 2048), nprocs=world_size)
    #data = GameDataset("data")
