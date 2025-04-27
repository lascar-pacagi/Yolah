import sys
from nnue_multigpu_clipped import Net, GameDataset, LAST_MODEL
import torch
sys.path.append("../server")
from yolah import Yolah
sys.path.append("../data")
from replayer import GameHistory

device = "cpu"
net = Net()
net.load_state_dict(torch.load(LAST_MODEL, map_location=torch.device('cpu')))
net.to(device)
net.eval()
history = GameHistory()

def get_nnue_prediction(yolah):
    board = GameDataset.encode_yolah(yolah).unsqueeze(0)
    board = board.to(device)
    logits = net(board)[0]
    black, draw, white = logits[0].item(), logits[1].item(), logits[2].item()
    return black, draw, white

def main():
    history.read(sys.argv[1])
    for i in range(len(history)):
        history.set_current_game(i)
        yolah = history.get_current_game()
        while not yolah.game_over():
            black_proba, draw_proba, white_proba = get_nnue_prediction(yolah)
            print(f'{black_proba:.17}\n{draw_proba:.17}\n{white_proba:.17}')      
            history.play()

if __name__ == "__main__":
    main()