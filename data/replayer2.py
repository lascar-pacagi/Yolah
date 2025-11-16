from tkinter import Tk, Canvas, Button, Menu, Entry, Label, N
from tkinter import StringVar, END, FALSE, messagebox
from tkinter.filedialog import askopenfilename
import re
import sys
sys.path.append("../server")
from yolah import Yolah, Move, Cell
sys.path.append("../python_binding/build")
from yolah_bind import heuristic
sys.path.append("../nnue")
import nnue_multigpu5 as nnue
from pathlib import Path
import torch

class GameHistory:
    GAME_RE  = re.compile(r"""(\w\d:\w\d)""")
    SCORE_RE = re.compile(r"""(\d+),(\d+)""")
    def __init__(self):
        self.games = []
        self.yolah = Yolah()
        self.moves = []
        self.scores = (0, 0)
        self.move_index = -1
        self.game_index = -1

    def read(self, filename):
        with open(filename) as f:
            self.games = []
            for line in f:
                self.games.append(line.strip())
            self.set_current_game(0)

    def valid(self):
        return self.games != []

    def set_current_game(self, index):
        self.game_index = index
        self.yolah.reset()
        line = self.games[index]
        #print(line)
        self.moves = GameHistory.GAME_RE.findall(line)
        self.move_index = 0
        self.scores = GameHistory.SCORE_RE.findall(line)[0]

    def get_current_game(self):
        return self.yolah

    def get_scores(self):
        return self.scores

    def previous_move(self):
        if self.move_index <= 0: return 
        return Move.from_str(self.moves[self.move_index - 1])

    def play(self):
        if self.yolah.game_over(): return
        self.yolah.play(Move.from_str(self.moves[self.move_index]))
        self.move_index += 1

    def undo(self):
        if self.move_index <= 0: return
        self.move_index -= 1
        self.yolah.undo(Move.from_str(self.moves[self.move_index]))

    def begin(self):
        while self.move_index > 0:
            self.undo()

    def end(self):
        while not self.yolah.game_over():
            self.play()

    def get_move_index(self):
        return self.move_index

    def __getitem__(self, i):
        if i < 0 or i >= len(self): raise IndexError
        return self.games[i]

    def __len__(self):
        return len(self.games)

history = GameHistory()

class Model:
    def __init__(self):
        self.net = None
        self.prediction_fn = None

    def is_valid(self):
        return self.net != None and self.prediction_fn != None 
    
    def set_net(self, net):
        self.net = net

    def set_fn(self, fn):
        self.prediction_fn = fn

    def get_prediction(self, yolah):
        return self.prediction_fn(yolah)

model = Model()
    
NB_GAMES_TXT = "# games: "
GAME_INFOS_TXT = "Turn: {}\n\nBlack score: {}\nWhite score: {}\n\nModel prediction\n * Black victory: {}\n * Draw         : {}\n * White victory: {}\n\nHeuristic prediction\n * Black value: {}\n * White value: {}"

CANVAS_WIDTH  = 700
CANVAS_HEIGHT = 700

def update_game_infos(game_infos_var):
    if not history.valid(): return
    yolah = history.get_current_game()
    if model.is_valid(): 
        b, d, w = model.get_prediction(yolah)
        black_victory_p = "{:.3f}".format(b)
        draw_p = "{:.3f}".format(d)
        white_victory_p = "{:.3f}".format(w)
    else:
        black_victory_p, draw_p, white_victory_p = "", "", ""
    (black, white, empty, black_score, white_score, ply) = yolah.get_state()
    black_value = heuristic(Yolah.BLACK_PLAYER, black, white, empty, black_score, white_score, ply)
    white_value = heuristic(Yolah.WHITE_PLAYER, black, white, empty, black_score, white_score, ply)
    game_infos_var.set(GAME_INFOS_TXT.format("WHITE" if yolah.nb_plies() & 1 else "BLACK", 
                        black_score, white_score, 
                        black_victory_p, draw_p, white_victory_p,
                        black_value, white_value))
        
def read_file(entry, nb_games_var, canvas, game_infos_var):
    filename = askopenfilename()
    history.read(filename)
    entry.delete(0, END)
    entry.insert(0, "1")
    nb_games_var.set(NB_GAMES_TXT + str(len(history)))
    history.set_current_game(0)
    draw_game(canvas, game_infos_var)
    
def entry_update(entry, canvas, game_infos_var):
    if not history.valid(): return
    history.set_current_game(int(entry.get()) - 1)
    draw_game(canvas, game_infos_var)

def play_game(play_var, canvas, game_infos_var):
    if not history.valid(): return
    if play_var.get() == "Play":
        play_var.set("Pause")
        history.play()
        draw_game(canvas, game_infos_var)
        canvas.after(1500, lambda: continue_game(play_var, canvas, game_infos_var))
    else:
        play_var.set("Play")

def continue_game(play_var, canvas, game_infos_var):
    if play_var.get() != "Play":
        history.play()
        draw_game(canvas, game_infos_var)
        canvas.after(1500, lambda: continue_game(play_var, canvas, game_infos_var))

def draw_game(canvas, game_infos_var):
    if not history.valid(): return
    update_game_infos(game_infos_var)    
    yolah = history.get_current_game()
    canvas.create_rectangle((0, 0), (CANVAS_WIDTH, CANVAS_HEIGHT), fill='black')
    w = CANVAS_WIDTH // Yolah.DIM
    h = CANVAS_HEIGHT // Yolah.DIM
    dx, dy = w // 8, h // 8
    grid = yolah.grid()
    for i in range(Yolah.DIM):
        for j in range(Yolah.DIM):
            y, x = i * w, j * h            
            canvas.create_rectangle((x, y), (x + w, y + h), fill=['grey', 'maroon'][(i + j) % 2], outline='black')
            match grid[Yolah.DIM - 1 - i][j]:
                case Cell.BLACK:
                    canvas.create_oval((x + dx, y + dy), (x + w - dx, y + h - dy), fill='black', outline='white')
                case Cell.WHITE:
                    canvas.create_oval((x + dx, y + dy), (x + w - dx, y + h - dy), fill='white', outline='black')
                case Cell.EMPTY:
                    canvas.create_rectangle((x, y), (x + w, y + h), fill='black', outline='black')
                case Cell.FREE:
                    ()
    if yolah.nb_plies() != 0:
        m = history.previous_move()
        i1, j1 = m.from_sq.to_coordinates()
        y1, x1 = (Yolah.DIM - 1 - i1) * w + w // 2, j1 * h + h // 2
        i2, j2 = m.to_sq.to_coordinates()
        y2, x2 = (Yolah.DIM - 1 - i2) * w + w // 2, j2 * h + h // 2
        canvas.create_line((x1, y1), (x2, y2), width=w // 5, arrow='last', arrowshape=(w // 3, w // 2, w // 5), fill='orange')

def next_move(entry, canvas, game_infos_var):
    if not history.valid(): return
    if history.get_current_game().game_over():
        n = int(entry.get())
        if n < len(history):
            entry.delete(0, END)
            entry.insert(0, n + 1)
            history.set_current_game(n)
    else:
        history.play()
    draw_game(canvas, game_infos_var)

def previous_move(entry, canvas, game_infos_var):
    if not history.valid(): return
    if history.get_current_game().nb_plies() == 0:
        n = int(entry.get())
        if n > 1:
            entry.delete(0, END)
            entry.insert(0, n - 1)
            history.set_current_game(n - 2)
    else:
        history.undo()
    draw_game(canvas, game_infos_var)

def start_of_game(entry, canvas, game_infos_var):
    if not history.valid(): return
    if history.get_current_game().nb_plies() == 0:       
        entry.delete(0, END)
        entry.insert(0, 1)
        history.set_current_game(0)
    else:
        history.begin()
    draw_game(canvas, game_infos_var)

def end_of_game(entry, canvas, game_infos_var):
    if not history.valid(): return
    if history.get_current_game().game_over():       
        entry.delete(0, END)
        entry.insert(0, len(history))
        history.set_current_game(len(history) - 1)
    else: history.end()
    draw_game(canvas, game_infos_var)

def load_model(canvas, game_infos_var):
    filename = askopenfilename()
    model_name = Path(filename).stem
    match model_name:
        case "nnue":
            device = "cpu"
            net = nnue.Net()
            net.to(device)
            model.set_net(net)
            net.load_state_dict(torch.load(filename))
            net.eval()
            def get_prediction(yolah):
                board = nnue.GameDataset.encode_yolah(history.get_current_game()).unsqueeze(0)
                board.to(device)
                logits = net(board)[0]
                black, draw, white = logits[0].item(), logits[1].item(), logits[2].item()
                return black, draw, white
            model.set_fn(get_prediction)
        case _:
            messagebox.showerror("Model Error", f"{filename} does not represent a known model")
    draw_game(canvas, game_infos_var)
            
def main():
    root=Tk()
    root.title("Yolah Replayer")
    root.resizable(width=FALSE, height=FALSE)
    WIDTH  = 920
    HEIGHT = 800    
    FONT = "fixed 10 bold"
    x = (root.winfo_screenwidth() // 2) - (WIDTH // 2)
    y = (root.winfo_screenheight() // 2) - (HEIGHT // 2)
    root.geometry("%dx%d+%d+%d" % (WIDTH, HEIGHT, x, y))    
    canvas = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="black")
    canvas.grid(row=0, column=0, columnspan=7)        
    nb_games_var = StringVar()
    nb_games_var.set(NB_GAMES_TXT)
    nb_games = Label(root, textvariable=nb_games_var, font=FONT)
    nb_games.grid(row=2, column=5)
    game_infos_var = StringVar()
    game_infos_var.set(GAME_INFOS_TXT.format("BLACK", 0, 0, "", "", "", "", ""))
    game_infos = Label(root, textvariable=game_infos_var, font=FONT, justify="left")
    game_infos.grid(row=0, column=7, pady=20, sticky=N)
    entry = Entry(root)
    entry.bind("<Return>", lambda _: entry_update(entry, canvas, game_infos_var))
    entry.grid(row=1, column=5)
    menu = Menu(root)
    root.config(menu=menu)
    file = Menu(menu, tearoff=0)
    file.add_command(label="Load Games", command=lambda: read_file(entry, nb_games_var, canvas, game_infos_var))
    file.add_command(label="Load Model", command=lambda: load_model(canvas, game_infos_var))
    file.add_command(label="Exit", command=root.destroy)
    menu.add_cascade(label="File", menu=file)
    begin = Button(root, text="Begin", font=FONT, command=lambda: start_of_game(entry, canvas, game_infos_var))
    begin.grid(row=1, column=0) 
    prev = Button(root, text="Previous", font=FONT, command=lambda: previous_move(entry, canvas, game_infos_var))
    prev.grid(row=1, column=1)
    play_var = StringVar()
    play_var.set("Play")
    play = Button(root, textvariable=play_var, font=FONT, command=lambda: play_game(play_var, canvas, game_infos_var))
    play.grid(row=1, column=2)
    nxt = Button(root, text="Next", font=FONT, command=lambda: next_move(entry, canvas, game_infos_var))
    nxt.grid(row=1, column=3)
    end = Button(root, text="End", font=FONT, command=lambda: end_of_game(entry, canvas, game_infos_var))
    end.grid(row=1, column=4)
    draw_game(canvas, game_infos_var)
    root.mainloop()

if __name__ == "__main__":
    main()
