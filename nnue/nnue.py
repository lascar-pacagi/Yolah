import os
import torch
from torch import nn
from tqdm import tqdm
import re
from torch.utils.data import Dataset
import glob
import sys
sys.path.append("../server")
from yolah import Yolah, Move
import itertools

# NN_SIZE_L1 = 256
# NN_SIZE_L2 = 32
# NN_SIZE_L3 = 32
# NN_SIZE_L4 = 2

# # Q and -Q (+/- 1,98) are the minimum and maximum values allowed for weights and biases
# Q = 127 / 64

# # black positions + white positions + empty positions + occupied positions + turn 
# INPUT_SIZE = 64 + 64 + 64 + 64 + 1

def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0]*(64 - len(b)) + b

class GameDataset(Dataset):
    GAME_RE  = re.compile(r"""(\w\d:\w\d)""")
    SCORE_RE = re.compile(r"""(\d+)/(\d+)""")

    @staticmethod
    def encode(moves):
        yolah = Yolah()
        for m in map(lambda m: Move.from_str(m), moves):
            yolah.play(m)
        res = list(itertools.chain.from_iterable([
                    bitboard64_to_list(yolah.black), 
                    bitboard64_to_list(yolah.white), 
                    bitboard64_to_list(yolah.empty),
                    bitboard64_to_list(yolah.black | yolah.white | yolah.empty),
                    [yolah.nb_plies() & 1]]))
        return torch.tensor(res, dtype=torch.float32)

    def __init__(self, games_dir):
        self.inputs = []
        self.outputs = []        
        self.infos = []
        nb_positions = 0
        random_moves_re = re.compile(r""".*(\d)r.*""")
        for filename in tqdm(glob.glob(games_dir + "/games*")):
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
                        res = 0
                        if black_score > white_score: res = 1
                        if white_score > black_score: res = -1
                        self.outputs.append(torch.tensor([res, black_score - white_score], dtype=torch.float32))
                self.infos.append((filename, r, nb_positions))
        self.size = nb_positions

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
        print(n, moves, lo, r)
        return GameDataset.encode(moves[: r + idx - n]), self.outputs[lo]

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
