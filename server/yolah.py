from enum import Enum
import json

def bit_not(n, numbits=64):
    return (1 << numbits) - 1 - n

class Cell(Enum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    FREE  = 3

class Direction(Enum):
    NORTH = 8
    EAST  = 1
    SOUTH = -NORTH
    WEST  = -EAST
    NORTH_EAST = NORTH + EAST
    SOUTH_EAST = SOUTH + EAST
    SOUTH_WEST = SOUTH + WEST
    NORTH_WEST = NORTH + WEST
    
    @staticmethod
    def all():
        return [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST, 
                Direction.NORTH_EAST, Direction.SOUTH_EAST, Direction.SOUTH_WEST, Direction.NORTH_WEST]

    @staticmethod
    def shift(dir, b):
        file_a = 0x0101010101010101
        file_h = file_a << 7
        match dir:
            case Direction.NORTH: return b << 8
            case Direction.EAST:  return (b & bit_not(file_h)) << 1
            case Direction.SOUTH: return b >> 8
            case Direction.WEST:  return (b & bit_not(file_a)) >> 1 
            case Direction.NORTH_EAST: return (b & bit_not(file_h)) << 9 
            case Direction.SOUTH_EAST: return (b & bit_not(file_h)) >> 7
            case Direction.SOUTH_WEST: return (b & bit_not(file_a)) >> 9
            case Direction.NORTH_WEST: return (b & bit_not(file_a)) << 7

FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']
    
class Square(Enum):
    SQ_A1 = 0;  SQ_B1 = 1;  SQ_C1 = 2;  SQ_D1 = 3;  SQ_E1 = 4;  SQ_F1 = 5;  SQ_G1 = 6;  SQ_H1 = 7
    SQ_A2 = 8;  SQ_B2 = 9;  SQ_C2 = 10; SQ_D2 = 11; SQ_E2 = 12; SQ_F2 = 13; SQ_G2 = 14; SQ_H2 = 15
    SQ_A3 = 16; SQ_B3 = 17; SQ_C3 = 18; SQ_D3 = 19; SQ_E3 = 20; SQ_F3 = 21; SQ_G3 = 22; SQ_H3 = 23
    SQ_A4 = 24; SQ_B4 = 25; SQ_C4 = 26; SQ_D4 = 27; SQ_E4 = 28; SQ_F4 = 29; SQ_G4 = 30; SQ_H4 = 31
    SQ_A5 = 32; SQ_B5 = 33; SQ_C5 = 34; SQ_D5 = 35; SQ_E5 = 36; SQ_F5 = 37; SQ_G5 = 38; SQ_H5 = 39
    SQ_A6 = 40; SQ_B6 = 41; SQ_C6 = 42; SQ_D6 = 43; SQ_E6 = 44; SQ_F6 = 45; SQ_G6 = 46; SQ_H6 = 47
    SQ_A7 = 48; SQ_B7 = 49; SQ_C7 = 50; SQ_D7 = 51; SQ_E7 = 52; SQ_F7 = 53; SQ_G7 = 54; SQ_H7 = 55
    SQ_A8 = 56; SQ_B8 = 57; SQ_C8 = 58; SQ_D8 = 59; SQ_E8 = 60; SQ_F8 = 61; SQ_G8 = 62; SQ_H8 = 63
    
    def __init__(self, sq):
        self.sq = sq

    @staticmethod
    def from_bitboard(bb):
        return Square(bb.bit_length() - 1)

    @staticmethod
    def from_str(s):
        file, rank = s[0], s[1]
        return Square(RANKS.index(rank) * 8 + FILES.index(file))

    def to_coordinates(self):
        return (self.sq // 8, self.sq % 8)

    def to_bitboard(self):
        return 1 << self.sq

    def __str__(self):
        return FILES[self.sq % 8] + str(self.sq // 8 + 1)
        
class Move:
    def __init__(self, from_sq, to_sq):
        self.from_sq = from_sq
        self.to_sq = to_sq

    @staticmethod
    def none():
        return Move(Square.SQ_A1, Square.SQ_A1)

    @staticmethod
    def from_str(m):
        from_sq, to_sq = m.split(':')
        return Move(Square.from_str(from_sq), Square.from_str(to_sq))
    
    def __eq__(self, other):
        return self.from_sq == other.from_sq and self.to_sq == other.to_sq

    def __str__(self):
        return str(self.from_sq) + ':' + str(self.to_sq)

class Yolah:
    DIM = 8
    BLACK_PLAYER = 0
    WHITE_PLAYER = 1
    def __init__(self):        
        self.full = 0xFFFFFFFFFFFFFFFF
        self.reset()

    def reset(self):
        self.black = 0b1000000000000000000000000000100000010000000000000000000000000001
        self.white = 0b0000000100000000000000000001000000001000000000000000000010000000
        self.empty = 0
        self.black_score = 0
        self.white_score = 0
        self.ply = 0

    def get_state(self):
        return (self.black, self.white, self.empty, self.black_score, self.white_score, self.ply)

    def positions(self, bitboard):
        res = []
        n = 0
        while bitboard:
            if bitboard & 1:
                res.append((n // Yolah.DIM, n % Yolah.DIM))
            bitboard >>= 1
            n += 1
        return res

    def grid(self):
        g = [[Cell.FREE for _ in range(Yolah.DIM)] for _ in range(Yolah.DIM)]
        for i, j in self.positions(self.black):
            g[i][j] = Cell.BLACK
        for i, j in self.positions(self.white):
            g[i][j] = Cell.WHITE
        for i, j in self.positions(self.empty):
            g[i][j] = Cell.EMPTY
        return g

    def nb_plies(self):
        return self.ply

    def current_player(self):
        return Yolah.WHITE_PLAYER if self.ply & 1 else Yolah.BLACK_PLAYER 

    def moves_for(self, player):
        res = []
        free = bit_not(self.black | self.white | self.empty)
        bitboard = self.black if player == Yolah.BLACK_PLAYER else self.white 
        while bitboard:
            pos = bitboard & -bitboard
            from_sq = Square.from_bitboard(pos)  
            for dir in Direction.all():
                dst = Direction.shift(dir, pos)
                while dst & free:
                    res.append(Move(from_sq, Square.from_bitboard(dst)))
                    dst = Direction.shift(dir, dst)
            bitboard &= bit_not(pos)
        return res if res != [] else [Move.none()]
    
    def moves(self):
        return self.moves_for(self.current_player())

    def game_over(self):
        possible = bit_not(self.black) & bit_not(self.white) & bit_not(self.empty)
        return (
            Direction.shift(Direction.NORTH, self.black) & possible == 0 and
            Direction.shift(Direction.EAST, self.black) & possible == 0 and
            Direction.shift(Direction.SOUTH, self.black) & possible == 0 and
            Direction.shift(Direction.WEST, self.black) & possible == 0 and
            Direction.shift(Direction.NORTH_EAST, self.black) & possible == 0 and
            Direction.shift(Direction.SOUTH_EAST, self.black) & possible == 0 and
            Direction.shift(Direction.SOUTH_WEST, self.black) & possible == 0 and
            Direction.shift(Direction.NORTH_WEST, self.black) & possible == 0 and
            Direction.shift(Direction.NORTH, self.white) & possible == 0 and
            Direction.shift(Direction.EAST, self.white) & possible == 0 and
            Direction.shift(Direction.SOUTH, self.white) & possible == 0 and
            Direction.shift(Direction.WEST, self.white) & possible == 0 and
            Direction.shift(Direction.NORTH_EAST, self.white) & possible == 0 and
            Direction.shift(Direction.SOUTH_EAST, self.white) & possible == 0 and
            Direction.shift(Direction.SOUTH_WEST, self.white) & possible == 0 and
            Direction.shift(Direction.NORTH_WEST, self.white) & possible == 0
        )

    def play(self, m):
        if m == Move.none():
            self.ply += 1
            return
        from_bb = m.from_sq.to_bitboard()
        to_bb   = m.to_sq.to_bitboard()
        if self.current_player() == Yolah.BLACK_PLAYER:
            self.black = self.black & bit_not(from_bb) | to_bb
            self.black_score += 1
        else:
            self.white = self.white & bit_not(from_bb) | to_bb
            self.white_score += 1
        self.empty |= from_bb
        self.ply += 1

    def undo(self, m):
        self.ply -= 1
        if m == Move.none():
            return
        from_bb = m.from_sq.to_bitboard()
        to_bb   = m.to_sq.to_bitboard()
        if self.current_player() == Yolah.BLACK_PLAYER:
            self.black = self.black & bit_not(to_bb) | from_bb
            self.black_score -= 1
        else:
            self.white = self.white & bit_not(to_bb) | from_bb
            self.white_score -= 1
        self.empty &= bit_not(from_bb)

    def to_json(self):
        state = {
            "black": str(self.black),
            "white": str(self.white),
            "empty": str(self.empty),
            "black score": str(self.black_score),
            "white score": str(self.white_score),
            "ply": str(self.ply)
        }
        return json.dumps(state)

    def from_json(self, state):
        state = json.loads(state)
        self.black = int(state["black"])
        self.white = int(state["white"])
        self.empty = int(state["empty"])
        self.black_score = int(state["black score"])
        self.white_score = int(state["white score"])
        self.ply = int(state["ply"])

    def __str__(self):
        g = self.grid()
        letters = '  a  b  c  d  e  f  g  h'
        res = letters + '\n'
        for i in range(Yolah.DIM):
            res += str(Yolah.DIM - i)
            for j in range(Yolah.DIM):
                match g[Yolah.DIM - 1 - i][j]:
                    case Cell.BLACK: res += ' \u25EF '
                    case Cell.WHITE: res += ' \u2B24 '
                    case Cell.EMPTY: res += '   '  
                    case Cell.FREE:  res += ' . '
            res += str(Yolah.DIM - i) + '\n'
        res += letters + '\n'
        res += f'score: {self.black_score}/{self.white_score}'
        return res

if __name__ == '__main__':
    y = Yolah()
    print(y.to_json())
    # while not y.game_over():
    #     print(y)
    #     for m in y.moves():
    #         print(m, end=' ')
    #     print()
    #     print('Your move: ', end='')
    #     m = input()
    #     y.play(Move.from_str(m))
    import random
    while not y.game_over():
        print(y)
        input()
        for m in y.moves():
            print(m, end=' ')
        print()
        moves = y.moves()
        m = moves[random.randint(0, len(moves) - 1)]
        print(m)
        y.play(m)
    print(y)
