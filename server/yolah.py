from enum import Enum

class Cell(Enum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    FREE  = 3

class Yolah:
    DIM = 8
    def __init__(self):
        self.black = 0b1000000000000000000000000000100000010000000000000000000000000001
        self.white = 0b0000000100000000000000000001000000001000000000000000000010000000
        self.empty = 0
        self.black_score = 0
        self.white_score = 0
        self.ply = 0

    def positions(self, bitboard):
        res = []
        i = 0
        while bitboard:
            if bitboard & 1:
                res.append((i // Yolah.DIM, i % Yolah.DIM))
            bitboard >>= 1
            i += 1
        return res

    def grid(self):
        g = [[Cell.FREE for _ in range(Yolah.DIM)] for _ in range(Yolah.DIM)]
        for (i, j) in self.positions(self.black):
            g[i][j] = Cell.BLACK
        for (i, j) in self.positions(self.white):
            g[i][j] = Cell.WHITE
        for (i, j) in self.positions(self.empty):
            g[i][j] = Cell.EMPTY
        return g

    def __str__(self):
        g = self.grid()
        letters = '  a  b  c  d  e  f  g  h'
        res = letters + '\n'
        for i, row in enumerate(g):
            res += str(Yolah.DIM - i)
            for j, c in enumerate(row):
                match c:
                    case Cell.BLACK: res += ' \u2B24 '
                    case Cell.WHITE: res += ' \u25EF '
                    case Cell.EMPTY: res += ' \u2B1B '  
                    case Cell.FREE:  res += ' . '
            res += str(Yolah.DIM - i) + '\n'
        res += letters + '\n'
        res += f'score: {self.black_score}/{self.white_score}'
        return res

if __name__ == '__main__':
    print(Yolah())
