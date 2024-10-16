import os
import torch
from torch import nn
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import sys
sys.path.append("../server")
from yolah import Yolah, Move
import itertools

def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0]*(64 - len(b)) + b

CLAMP_SCORE=10

class GameDataset(Dataset):
    GAME_RE  = re.compile(r"""(\w\d:\w\d)""")
    SCORE_RE = re.compile(r"""(\d+)/(\d+)""")
    
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
                        score = min(CLAMP_SCORE, max(-CLAMP_SCORE, black_score - white_score)) / CLAMP_SCORE
                        self.outputs.append(torch.tensor([res, score], dtype=torch.float32))
                self.infos.append((filename, r, nb_positions))
        self.size = nb_positions

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

# # Q and -Q (+/- 1,98) are the minimum and maximum values allowed for weights and biases
# # this opens the possibility to use a quantized version of the network post-training
# Q = 127 / 64

# black positions + white positions + empty positions + occupied positions + turn 
INPUT_SIZE = 64 + 64 + 64 + 64 + 1

class Net(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, l1_size=1024, l2_size=512, l3_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        res = torch.tanh(x[:,0])
        score = torch.clamp(x[:,1] / CLAMP_SCORE, -1, 1)
        return res, score

NB_EPOCHS=1000
MODEL_PATH="./nnue.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = GameDataset("../data")
    print(len(dataset))
    train_set, test_set = random_split(dataset, [0.8, 0.2])
    print(len(train_set), len(test_set))
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=True, num_workers=0)
    net = torch.load(MODEL_PATH, weights_only=False).to(device) if os.path.isfile(MODEL_PATH) else Net().to(device)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    res_loss_fn = nn.CrossEntropyLoss()
    score_loss_fn = nn.MSELoss()
    running_loss = 0    
    for epoch in range(NB_EPOCHS):
        net.train()
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            res, score = net(X)
            loss = 10 * res_loss_fn(res, y[:,0]) + score_loss_fn(score, y[:,1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                print('  batch {} loss: {}'.format(i + 1, running_loss / len(X) / 1000))
                running_loss = 0
        net.eval()
        torch.save(net, MODEL_PATH)
        if epoch % 10 == 9:
            accuracy_res = 0
            accuracy_score = 0
            n = 0
            res_values = torch.tensor([-1, 0, 1]).to(device)
            for (X, y) in tqdm(test_loader):
                n += len(X)
                X = X.to(device)
                y = y.to(device)            
                res, score = net(X)
                diff = abs(res.unsqueeze(1) - res_values)
                nearest_indices = torch.argmin(diff, dim=1)
                nearest_elements = res_values[nearest_indices]
                accuracy_res += sum(nearest_elements == y[:,0])
                accuracy_score += sum(abs(score - y[:,1])).item()
            print('  Accuracy res: {}\n  Accuracy score: {}'.format(accuracy_res / n, 1.0 - accuracy_score / n))

if __name__ == "__main__":
    main()