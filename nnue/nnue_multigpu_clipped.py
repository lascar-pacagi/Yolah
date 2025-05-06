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
from yolah import Yolah, Move
import itertools

def bit_not(n, numbits=64):
    return (1 << numbits) - 1 - n

def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0]*(64 - len(b)) + b

class GameDataset(Dataset):
    GAME_RE  = re.compile(r"""(\w\d:\w\d)""")
    SCORE_RE = re.compile(r"""(\d+)/(\d+)""")
    
    def __init__(self, games_dir):
        self.inputs = []
        self.outputs = []
        self.infos = []
        nb_positions = 0
        random_moves_re = re.compile(r""".*(\d)r.*""")
        for filename in glob.glob(games_dir + "/games*"):
            l = random_moves_re.findall(filename)
            if l != []:
                r = int(l[0])
                with open(filename) as f:
                    for line in f:
                        moves = GameDataset.GAME_RE.findall(line)
                        nb_positions += len(moves[r:]) + 1
                        self.inputs.append((nb_positions, moves))
                        black_score, white_score = GameDataset.SCORE_RE.findall(line)[0]                              
                        black_score = int(black_score)
                        white_score = int(white_score)
                        res = 1
                        if black_score > white_score: 
                            res = 0
                        elif white_score > black_score: 
                            res = 2                       
                        #self.outputs.append(res)
                        self.outputs.append(torch.tensor(res, dtype=torch.long))
                self.infos.append((filename, r, nb_positions))
        self.size = nb_positions

    # @staticmethod
    # def encode_yolah(yolah):
    #     res = list(itertools.chain.from_iterable([
    #                 bitboard64_to_list(yolah.black), 
    #                 bitboard64_to_list(yolah.white), 
    #                 bitboard64_to_list(yolah.empty),
    #                 bitboard64_to_list(yolah.black | yolah.white | yolah.empty),
    #                 bitboard64_to_list(bit_not(yolah.black | yolah.white | yolah.empty)),
    #                 [yolah.nb_plies() & 1]*64]))
    #     return torch.tensor(res, dtype=torch.float32)

    # @staticmethod
    # def encode_yolah(yolah):
    #     if yolah.current_player() == Yolah.BLACK_PLAYER:
    #         res = list(itertools.chain.from_iterable([
    #                     bitboard64_to_list(yolah.black), 
    #                     bitboard64_to_list(yolah.white), 
    #                     bitboard64_to_list(yolah.empty)]))
    #     else:
    #         res = list(itertools.chain.from_iterable([
    #                     bitboard64_to_list(yolah.white), 
    #                     bitboard64_to_list(yolah.black), 
    #                     bitboard64_to_list(yolah.empty)]))
    #     return torch.tensor(res, dtype=torch.float32)

    @staticmethod
    def encode_yolah(yolah):
        res = list(itertools.chain.from_iterable([
                    bitboard64_to_list(yolah.black), 
                    bitboard64_to_list(yolah.white), 
                    bitboard64_to_list(yolah.empty),                    
                    [Yolah.WHITE_PLAYER if yolah.nb_plies() & 1 else Yolah.BLACK_PLAYER]]))
        return torch.tensor(res, dtype=torch.float32)

    @staticmethod
    def encode(moves):
        yolah = Yolah()
        for m in map(lambda m: Move.from_str(m), moves):
            yolah.play(m)
        return GameDataset.encode_yolah(yolah)

    # @staticmethod
    # def encode(moves, res):
    #     yolah = Yolah()
    #     for m in map(lambda m: Move.from_str(m), moves):
    #         yolah.play(m)
    #     output = 1
    #     if res == 0:
    #         if yolah.current_player() == Yolah.BLACK_PLAYER: output = 0
    #         else: output = 2
    #     elif res == 2:
    #         if yolah.current_player() == Yolah.WHITE_PLAYER: output = 0
    #         else: output = 2
    #     return GameDataset.encode_yolah(yolah), torch.tensor(output, dtype=torch.long)

    @staticmethod
    def encode(moves):
        yolah = Yolah()
        for m in map(lambda m: Move.from_str(m), moves):
            yolah.play(m)        
        return GameDataset.encode_yolah(yolah)

    def get_infos(self):
        return self.infos
    
    def get_nb_random_moves(self, idx):
        for (_, r, n) in self.infos:
            if idx < n: return r
        raise RuntimeError("get_nb_random_moves")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        r = self.get_nb_random_moves(idx)
        lo = 0
        hi = len(self.inputs)
        while lo < hi:
            m = lo + (hi - lo) // 2
            n, _ = self.inputs[m]
            if n <= idx: 
                lo = m + 1
            else: 
                hi = m
        n = self.inputs[lo - 1][0] if lo > 0 else 0
        moves = self.inputs[lo][1]
        #print(n, moves, lo, r)
        return GameDataset.encode(moves[: r + idx - n]), self.outputs[lo]

# black positions + white positions + empty positions + occupied positions + free positions + turn 
#INPUT_SIZE = 64 + 64 + 64 + 64 + 64 + 64

# black positions + white positions + empty positions + turn 
INPUT_SIZE = 64 + 64 + 64 + 1

class Net(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, l1_size=1024, l2_size=64, l3_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 3)

    def forward(self, x):
        x = self.fc1(x)
        #x = relu(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc2(x)
        #x = relu(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        x = self.fc3(x)
        #x = relu(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return softmax(self.fc4(x), dim=1)#self.fc4(x)#
    
    def clip(self):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4]:
            fc.weight.data.clamp_(-127/64, 127/64)
            fc.bias.data.clamp_(-127/64, 127/64)
            
NB_EPOCHS=1000
MODEL_PATH="./"#"/mnt/"
MODEL_NAME="nnue_1024x64x32x3.0"
LAST_MODEL=f"{MODEL_PATH}{MODEL_NAME}.pt"

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def dataloader_ddp(trainset, batch_size):
    sampler_train = DistributedSampler(trainset)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, sampler=sampler_train, pin_memory=True, num_workers=0
    )
    return train_loader, sampler_train

class TrainerDDP:
    def __init__(self, gpu_id, model, train_loader, sampler_train, lr_step_size=15, save_every=5):
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.sampler_train = sampler_train
        self.save_every = save_every
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_step_size)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _save_checkpoint(self, epoch):
        torch.save(self.model.module.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in self.train_loader:            
            n += len(X)
            X = X.to(self.gpu_id)
            y = y.to(self.gpu_id)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            accuracy += sum(torch.argmax(logits, dim=1) == y).item()
            self.model.clip()
        if self.gpu_id == 0:
            print('epoch {} loss: {} accuracy: {}'.format(epoch + 1, running_loss / n, accuracy / n), flush=True)
        self.lr_scheduler.step()

    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self.sampler_train.set_epoch(epoch)
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)

def main(rank, world_size, batch_size):
    ddp_setup(rank, world_size)
    dataset = GameDataset("../data/games")
    print(rank)
    if rank == 0:
        print(len(dataset), flush=True)
    train_loader, sampler_train = dataloader_ddp(dataset, batch_size) 
    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    if rank == 0:
        print(net, flush=True)
    trainer = TrainerDDP(rank, net, train_loader, sampler_train)
    trainer.train(NB_EPOCHS)
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    mp.spawn(main, args=(world_size, 1024), nprocs=world_size)
