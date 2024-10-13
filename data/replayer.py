from tkinter import Tk, Canvas, Button, Menu, Entry, Label, N
from tkinter import StringVar, END, FALSE
from tkinter.filedialog import askopenfilename
import re
import sys
sys.path.append("../server")
from yolah import Yolah, Move, Cell

class GameHistory:
    GAME_RE  = re.compile(r"""(\w\d:\w\d)""")
    SCORE_RE = re.compile(r"""(\d+)/(\d+)""")
    def __init__(self):
        self.games = []
        self.yolah = Yolah()
        self.moves = []
        self.scores = (0, 0)
        self.index = -1

    def read(self, filename):
        with open(filename) as f:
            for line in f:
                self.games.append(line.strip())

    def set_current_game(self, index):
        self.yolah.reset()
        line = self.games[index]
        self.moves = GameHistory.GAME_RE.findall(line)
        self.index = 0
        self.scores = GameHistory.SCORE_RE.findall(line)[0]

    def get_current_game(self):
        return self.yolah

    def get_scores(self): 
        return self.scores

    def play(self):
        if self.yolah.game_over(): return
        self.yolah.play(Move.from_str(self.moves[self.index]))
        self.index += 1

    def undo(self):
        if self.index == 0: return
        self.index -= 1
        self.yolah.undo(Move.from_str(self.moves[self.index]))

    def __getitem__(self, i):
        if i < 0 or i >= len(self): raise IndexError
        return self.games[i]

    def __len__(self):
        return len(self.games)

history = GameHistory()

NB_GAMES_TXT = "# games: "
GAME_INFOS_TXT = "Turn: {}\n\nBlack score: {}\nWhite score: {}"

CANVAS_WIDTH  = 700
CANVAS_HEIGHT = 700

def update_yolah(index):
    history.set_current_game(index)

def read_file(entry, nb_games_var, canvas, game_infos_var):
    filename = askopenfilename()
    history.read(filename)
    entry.delete(0, END)
    entry.insert(0, "1")
    nb_games_var.set(NB_GAMES_TXT + str(len(history)))
    update_yolah(0)
    draw_game(canvas, game_infos_var)
    
def entry_update(entry, canvas, game_infos_var):
    update_yolah(int(entry.get()) - 1)
    draw_game(canvas, game_infos_var)

def draw_game(canvas, game_infos_var):
    black_score, white_score = history.get_scores()
    game_infos_var.set(GAME_INFOS_TXT.format("BLACK", black_score, white_score))
    yolah = history.get_current_game()
    canvas.create_rectangle((0, 0), (CANVAS_WIDTH, CANVAS_HEIGHT), fill='black')
    w = CANVAS_WIDTH // Yolah.DIM
    h = CANVAS_HEIGHT // Yolah.DIM
    grid = yolah.grid()
    for i in range(Yolah.DIM):
        for j in range(Yolah.DIM):
            x, y = i * w, j * h
            dx, dy = w // 8, h // 8            
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

def next_move(canvas, game_infos_var):
    history.play()
    draw_game(canvas, game_infos_var)

def previous_move(canvas, game_infos_var):
    history.undo()
    draw_game(canvas, game_infos_var)

def main():
    root=Tk()
    root.title("Yolah Replayer")
    root.resizable(width=FALSE, height=FALSE)
    WIDTH  = 870
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
    game_infos_var.set(GAME_INFOS_TXT.format("BLACK", 0, 0))
    game_infos = Label(root, textvariable=game_infos_var, font=FONT, justify="left")
    game_infos.grid(row=0, column=7, pady=20, sticky=N)
    entry = Entry(root)
    entry.bind("<Return>", lambda _: entry_update(entry, canvas, game_infos_var))
    entry.grid(row=1, column=5)
    menu = Menu(root)
    root.config(menu=menu)
    file = Menu(menu, tearoff=0)
    file.add_command(label="Load", command=lambda: read_file(entry, nb_games_var, canvas, game_infos_var))
    file.add_command(label="Exit", command=root.destroy)
    menu.add_cascade(label="File", menu=file)
    begin = Button(root, text="Begin", font=FONT)
    begin.grid(row=1, column=0) 
    prev = Button(root, text="Previous", font=FONT, command=lambda: previous_move(canvas, game_infos_var))
    prev.grid(row=1, column=1)
    play = Button(root, text="Play", font=FONT)
    play.grid(row=1, column=2)
    nxt = Button(root, text="Next", font=FONT, command=lambda: next_move(canvas, game_infos_var))
    nxt.grid(row=1, column=3)
    end = Button(root, text="End", font=FONT)
    end.grid(row=1, column=4)
    draw_game(canvas, game_infos_var)
    root.mainloop()

main()

