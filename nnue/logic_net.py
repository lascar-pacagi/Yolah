from difflogic import LogicLayer, GroupSum
import torch
import os
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader
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
                        self.outputs.append(torch.tensor(res, dtype=torch.long))
                self.infos.append((filename, r, nb_positions))
        self.size = nb_positions

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

# black positions + white positions + empty positions + turn 
INPUT_SIZE = 64 + 64 + 64 + 1

class Net(torch.nn.Module):
    def __init__(self, input_size=INPUT_SIZE, l1_size=16_000, l2_size=16_000, l3_size=16_000):
        super().__init__()
        self.l1 = LogicLayer(input_size, l1_size)
        self.l2 = LogicLayer(l1_size, l2_size)
        self.l3 = LogicLayer(l2_size, l3_size)
        self.l4 = GroupSum(k=3, tau=100)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.l4(x)
        #return torch.softmax(self.l4(x), dim=1)
            
MODEL_PATH="./"#"/mnt/"
MODEL_NAME="logic"
LAST_MODEL=f"{MODEL_PATH}{MODEL_NAME}.pt"

class Trainer:
    def __init__(self, gpu_id, model, loader, save_every=5):
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.loader = loader
        self.save_every = save_every
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), f"{MODEL_PATH}{MODEL_NAME}.{epoch}.pt")

    def _run_epoch(self, epoch):
        n = 0
        running_loss = 0
        accuracy = 0
        for (X, y) in self.loader:
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
        print('epoch {} loss: {} accuracy: {}'.format(epoch + 1, running_loss / n, accuracy / n), flush=True)
        
    def train(self, nb_epochs):
        self.model.train()
        for epoch in range(nb_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint(nb_epochs - 1)

def main(gpu_id, nb_epochs, batch_size):
    dataset = GameDataset("data")
    net = Net()
    if os.path.isfile(LAST_MODEL):
        net.load_state_dict(torch.load(LAST_MODEL))
    print(net, flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    trainer = Trainer(gpu_id, net, loader)
    trainer.train(nb_epochs)
    
if __name__ == "__main__":
    main(0, 100, 128)
