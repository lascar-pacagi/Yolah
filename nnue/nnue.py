import os
import torch
from torch import nn
from torch.nn.functional import relu, softmax
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
                        res = 1 
                        if black_score > white_score: res = 0
                        if white_score > black_score: res = 2                                               
                        self.outputs.append(torch.tensor(res, dtype=torch.long))
                self.infos.append((filename, r, nb_positions))
        self.size = nb_positions

    @staticmethod
    def encode_yolah(yolah):
        res = list(itertools.chain.from_iterable([
                    bitboard64_to_list(yolah.black), 
                    bitboard64_to_list(yolah.white), 
                    bitboard64_to_list(yolah.empty),
                    bitboard64_to_list(yolah.black | yolah.white | yolah.empty),
                    [yolah.nb_plies() & 1]*64]))
        return torch.tensor(res, dtype=torch.float32)

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

# black positions + white positions + empty positions + occupied positions + turn 
INPUT_SIZE = 64 + 64 + 64 + 64 + 64

# class Net(nn.Module):
#     def __init__(self, input_size=INPUT_SIZE, l1_size=1024, l2_size=512, l3_size=128, l4_size=32):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, l1_size)
#         self.fc2 = nn.Linear(l1_size, l2_size)
#         self.fc3 = nn.Linear(l2_size, l3_size)
#         self.fc4 = nn.Linear(l3_size, l4_size)
#         self.fc5 = nn.Linear(l4_size, 3)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = relu(x)
#         x = self.fc2(x)
#         x = relu(x)
#         x = self.fc3(x)
#         x = relu(x)
#         x = self.fc4(x)
#         x = relu(x)
#         return softmax(self.fc5(x), dim=1)

class Net(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, l1_size=1024, l2_size=512, l3_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, l3_size)
        self.fc4 = nn.Linear(l3_size, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        x = relu(x)
        return softmax(self.fc4(x), dim=1)

# class Net(nn.Module):
#     def __init__(self, input_size=INPUT_SIZE, l1_size=2048, l2_size=512, l3_size=128):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, l1_size)
#         self.fc2 = nn.Linear(l1_size, l2_size)
#         self.fc3 = nn.Linear(l2_size, l3_size)
#         self.fc4 = nn.Linear(l3_size, 3)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = relu(x)
#         x = self.fc2(x)
#         x = relu(x)
#         x = self.fc3(x)
#         x = relu(x)
#         return softmax(self.fc4(x), dim=1)

NB_EPOCHS=1000
MODEL_PATH="/mnt/nnue3.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    dataset = GameDataset("../data")
    print(len(dataset))
    train_set, test_set = random_split(dataset, [0.8, 0.2])
    print(len(train_set), len(test_set))
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=8)
    net = Net()
    if os.path.isfile(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH))
    print(net)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(NB_EPOCHS):
        net.train()
        n = 0
        running_loss = 0    
        for _, (X, y) in enumerate(train_loader):
            n += len(X)
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = net(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()            
        print('epoch {} loss: {}'.format(epoch + 1, running_loss / n))
        net.eval()
        torch.save(net.state_dict(), MODEL_PATH)
        if epoch % 20 == 19:
            with torch.no_grad():
                accuracy = 0
                n = 0
                for (X, y) in tqdm(test_loader):
                    n += len(X)
                    X = X.to(device)
                    y = y.to(device)            
                    logits = net(X)                
                    accuracy += sum(torch.argmax(logits, dim=1) == y).item()
            print('epoch {} accuracy: {}'.format(epoch + 1, accuracy / n))

if __name__ == "__main__":
    main()
