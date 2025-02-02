import sys
from nnue_multigpu import Net, GameDataset, LAST_MODEL
import torch
sys.path.append("../server")
from yolah import Yolah
sys.path.append("../python_binding/build")
from yolah_bind import heuristic
sys.path.append("../data")
from replayer import GameHistory
from tqdm import tqdm

device = "cpu"
net = Net()
net.load_state_dict(torch.load(LAST_MODEL, map_location=torch.device(device)))
net.to(device)
net.eval()
history = GameHistory()

def get_nnue_prediction(yolah):
    board = GameDataset.encode_yolah(yolah).unsqueeze(0)
    board = board.to(device)
    logits = net(board)[0]
    black, draw, white = logits[0].item(), logits[1].item(), logits[2].item()
    return black, draw, white

def get_heuristic_prediction(yolah):
    (black, white, empty, black_score, white_score, ply) = yolah.get_state()
    black_value = heuristic(Yolah.BLACK_PLAYER, black, white, empty, black_score, white_score, ply)
    white_value = heuristic(Yolah.WHITE_PLAYER, black, white, empty, black_score, white_score, ply)
    return black_value, white_value

def main():
    history.read(sys.argv[1])
    n = 0
    nnue_accuracy = 0
    heuristic_accuracy = 0
    for i in tqdm(range(len(history))):
        history.set_current_game(i)
        yolah = history.get_current_game()
        black_score, white_score = history.get_scores()
        while not yolah.game_over():
            black_proba, draw_proba, white_proba = get_nnue_prediction(yolah)
            black_value, white_value = get_heuristic_prediction(yolah)
            if black_score == white_score:
                #nnue_accuracy += black_proba == white_proba or draw_proba > 0.95
                nnue_accuracy += draw_proba == max([black_proba, draw_proba, white_proba])
                heuristic_accuracy += black_value == white_value
            elif black_score > white_score:
                #nnue_accuracy += black_proba > white_proba
                nnue_accuracy += black_proba == max([black_proba, draw_proba, white_proba])
                heuristic_accuracy += black_value > white_value            
            else:
                #nnue_accuracy += white_proba > black_proba
                nnue_accuracy += white_proba == max([black_proba, draw_proba, white_proba])
                heuristic_accuracy += white_value > black_value
            n += 1
            history.play()
    print(f'nnue accuracy     : {nnue_accuracy / n}')
    print(f'heuristic accuracy: {heuristic_accuracy / n}')

if __name__ == "__main__":
    main()