# from z3 import *

# from enum import Enum
# import json
# from itertools import combinations

# def bit_not(n, numbits=64):
#     return (1 << numbits) - 1 - n

# class Cell(Enum):
#     BLACK = 0
#     WHITE = 1
#     EMPTY = 2
#     FREE  = 3

# class Direction(Enum):
#     NORTH = 8
#     EAST  = 1
#     SOUTH = -NORTH
#     WEST  = -EAST
#     NORTH_EAST = NORTH + EAST
#     SOUTH_EAST = SOUTH + EAST
#     SOUTH_WEST = SOUTH + WEST
#     NORTH_WEST = NORTH + WEST
    
#     @staticmethod
#     def all():
#         return [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST, 
#                 Direction.NORTH_EAST, Direction.SOUTH_EAST, Direction.SOUTH_WEST, Direction.NORTH_WEST]

#     @staticmethod
#     def diagonal():
#         return [Direction.NORTH_EAST, Direction.SOUTH_EAST, Direction.SOUTH_WEST, Direction.NORTH_WEST]

#     @staticmethod
#     def horizontal():
#         return [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

#     @staticmethod
#     def shift(dir, b):
#         file_a = 0x0101010101010101
#         file_h = file_a << 7
#         match dir:
#             case Direction.NORTH: return b << 8
#             case Direction.EAST:  return (b & bit_not(file_h)) << 1
#             case Direction.SOUTH: return b >> 8
#             case Direction.WEST:  return (b & bit_not(file_a)) >> 1 
#             case Direction.NORTH_EAST: return (b & bit_not(file_h)) << 9 
#             case Direction.SOUTH_EAST: return (b & bit_not(file_h)) >> 7
#             case Direction.SOUTH_WEST: return (b & bit_not(file_a)) >> 9
#             case Direction.NORTH_WEST: return (b & bit_not(file_a)) << 7

# FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']
    
# class Square(Enum):
#     SQ_A1 = 0;  SQ_B1 = 1;  SQ_C1 = 2;  SQ_D1 = 3;  SQ_E1 = 4;  SQ_F1 = 5;  SQ_G1 = 6;  SQ_H1 = 7
#     SQ_A2 = 8;  SQ_B2 = 9;  SQ_C2 = 10; SQ_D2 = 11; SQ_E2 = 12; SQ_F2 = 13; SQ_G2 = 14; SQ_H2 = 15
#     SQ_A3 = 16; SQ_B3 = 17; SQ_C3 = 18; SQ_D3 = 19; SQ_E3 = 20; SQ_F3 = 21; SQ_G3 = 22; SQ_H3 = 23
#     SQ_A4 = 24; SQ_B4 = 25; SQ_C4 = 26; SQ_D4 = 27; SQ_E4 = 28; SQ_F4 = 29; SQ_G4 = 30; SQ_H4 = 31
#     SQ_A5 = 32; SQ_B5 = 33; SQ_C5 = 34; SQ_D5 = 35; SQ_E5 = 36; SQ_F5 = 37; SQ_G5 = 38; SQ_H5 = 39
#     SQ_A6 = 40; SQ_B6 = 41; SQ_C6 = 42; SQ_D6 = 43; SQ_E6 = 44; SQ_F6 = 45; SQ_G6 = 46; SQ_H6 = 47
#     SQ_A7 = 48; SQ_B7 = 49; SQ_C7 = 50; SQ_D7 = 51; SQ_E7 = 52; SQ_F7 = 53; SQ_G7 = 54; SQ_H7 = 55
#     SQ_A8 = 56; SQ_B8 = 57; SQ_C8 = 58; SQ_D8 = 59; SQ_E8 = 60; SQ_F8 = 61; SQ_G8 = 62; SQ_H8 = 63
    
#     def __init__(self, sq):
#         self.sq = sq

#     @staticmethod
#     def from_bitboard(bb):
#         return Square(bb.bit_length() - 1)

#     @staticmethod
#     def from_str(s):
#         file, rank = s[0], s[1]
#         return Square(RANKS.index(rank) * 8 + FILES.index(file))

#     def to_coordinates(self):
#         return (self.sq // 8, self.sq % 8)

#     def to_rank(self):
#         return self.value // 8

#     def to_file(self):
#         return self.value % 8

#     def to_bitboard(self):
#         return 1 << self.sq

#     def __str__(self):
#         return FILES[self.sq % 8] + str(self.sq // 8 + 1)

#     def __add__(self, direction):
#         """Add a direction to this square. Returns None if the result is invalid."""
#         if not isinstance(direction, Direction):
#             return None

#         new_sq = self.value + direction.value

#         # Check if the new square is out of bounds
#         if new_sq < Square.SQ_A1.value or new_sq > Square.SQ_H8.value:
#             return None

#         # Check for wrapping around files (horizontal wrapping)
#         old_file = self.value % 8
#         new_file = new_sq % 8

#         # If we moved east/west and the file changed by more than 1, we wrapped
#         if direction in [Direction.EAST, Direction.WEST]:
#             if abs(new_file - old_file) > 1:
#                 return None
#         # For diagonal moves, check both file and rank distance
#         elif direction in [Direction.NORTH_EAST, Direction.SOUTH_EAST,
#                           Direction.NORTH_WEST, Direction.SOUTH_WEST]:
#             if abs(new_file - old_file) != 1:
#                 return None

#         return Square(new_sq)
        
# class Move:
#     def __init__(self, from_sq, to_sq):
#         self.from_sq = from_sq
#         self.to_sq = to_sq

#     @staticmethod
#     def none():
#         return Move(Square.SQ_A1, Square.SQ_A1)

#     @staticmethod
#     def from_str(m):
#         from_sq, to_sq = m.split(':')
#         return Move(Square.from_str(from_sq), Square.from_str(to_sq))
    
#     def __eq__(self, other):
#         return self.from_sq == other.from_sq and self.to_sq == other.to_sq

#     def __str__(self):
#         return str(self.from_sq) + ':' + str(self.to_sq)

# class Yolah:
#     DIM = 8
#     BLACK_PLAYER = 0
#     WHITE_PLAYER = 1
#     def __init__(self):        
#         self.full = 0xFFFFFFFFFFFFFFFF
#         self.reset()

#     def reset(self):
#         self.black = 0b1000000000000000000000000000100000010000000000000000000000000001
#         self.white = 0b0000000100000000000000000001000000001000000000000000000010000000
#         self.empty = 0
#         self.black_score = 0
#         self.white_score = 0
#         self.ply = 0

#     def get_state(self):
#         return (self.black, self.white, self.empty, self.black_score, self.white_score, self.ply)

#     def positions(self, bitboard):
#         res = []
#         n = 0
#         while bitboard:
#             if bitboard & 1:
#                 res.append((n // Yolah.DIM, n % Yolah.DIM))
#             bitboard >>= 1
#             n += 1
#         return res

#     def grid(self):
#         g = [[Cell.FREE for _ in range(Yolah.DIM)] for _ in range(Yolah.DIM)]
#         for i, j in self.positions(self.black):
#             g[i][j] = Cell.BLACK
#         for i, j in self.positions(self.white):
#             g[i][j] = Cell.WHITE
#         for i, j in self.positions(self.empty):
#             g[i][j] = Cell.EMPTY
#         return g

#     def nb_plies(self):
#         return self.ply

#     def current_player(self):
#         return Yolah.WHITE_PLAYER if self.ply & 1 else Yolah.BLACK_PLAYER 

#     def moves_for(self, player):
#         res = []
#         free = bit_not(self.black | self.white | self.empty)
#         bitboard = self.black if player == Yolah.BLACK_PLAYER else self.white 
#         while bitboard:
#             pos = bitboard & -bitboard
#             from_sq = Square.from_bitboard(pos)  
#             for dir in Direction.all():
#                 dst = Direction.shift(dir, pos)
#                 while dst & free:
#                     res.append(Move(from_sq, Square.from_bitboard(dst)))
#                     dst = Direction.shift(dir, dst)
#             bitboard &= bit_not(pos)
#         return res if res != [] else [Move.none()]
    
#     def moves(self):
#         return self.moves_for(self.current_player())

#     def game_over(self):
#         possible = bit_not(self.black) & bit_not(self.white) & bit_not(self.empty)
#         return (
#             Direction.shift(Direction.NORTH, self.black) & possible == 0 and
#             Direction.shift(Direction.EAST, self.black) & possible == 0 and
#             Direction.shift(Direction.SOUTH, self.black) & possible == 0 and
#             Direction.shift(Direction.WEST, self.black) & possible == 0 and
#             Direction.shift(Direction.NORTH_EAST, self.black) & possible == 0 and
#             Direction.shift(Direction.SOUTH_EAST, self.black) & possible == 0 and
#             Direction.shift(Direction.SOUTH_WEST, self.black) & possible == 0 and
#             Direction.shift(Direction.NORTH_WEST, self.black) & possible == 0 and
#             Direction.shift(Direction.NORTH, self.white) & possible == 0 and
#             Direction.shift(Direction.EAST, self.white) & possible == 0 and
#             Direction.shift(Direction.SOUTH, self.white) & possible == 0 and
#             Direction.shift(Direction.WEST, self.white) & possible == 0 and
#             Direction.shift(Direction.NORTH_EAST, self.white) & possible == 0 and
#             Direction.shift(Direction.SOUTH_EAST, self.white) & possible == 0 and
#             Direction.shift(Direction.SOUTH_WEST, self.white) & possible == 0 and
#             Direction.shift(Direction.NORTH_WEST, self.white) & possible == 0
#         )

#     def play(self, m):
#         if m == Move.none():
#             self.ply += 1
#             return
#         from_bb = m.from_sq.to_bitboard()
#         to_bb   = m.to_sq.to_bitboard()
#         if self.current_player() == Yolah.BLACK_PLAYER:
#             self.black = self.black & bit_not(from_bb) | to_bb
#             self.black_score += 1
#         else:
#             self.white = self.white & bit_not(from_bb) | to_bb
#             self.white_score += 1
#         self.empty |= from_bb
#         self.ply += 1

#     def undo(self, m):
#         self.ply -= 1
#         if m == Move.none():
#             return
#         from_bb = m.from_sq.to_bitboard()
#         to_bb   = m.to_sq.to_bitboard()
#         if self.current_player() == Yolah.BLACK_PLAYER:
#             self.black = self.black & bit_not(to_bb) | from_bb
#             self.black_score -= 1
#         else:
#             self.white = self.white & bit_not(to_bb) | from_bb
#             self.white_score -= 1
#         self.empty &= bit_not(from_bb)

#     def to_json(self):
#         state = {
#             "black": str(self.black),
#             "white": str(self.white),
#             "empty": str(self.empty),
#             "black score": str(self.black_score),
#             "white score": str(self.white_score),
#             "ply": str(self.ply)
#         }
#         return json.dumps(state)

#     def from_json(self, state):
#         state = json.loads(state)
#         self.black = int(state["black"])
#         self.white = int(state["white"])
#         self.empty = int(state["empty"])
#         self.black_score = int(state["black score"])
#         self.white_score = int(state["white score"])
#         self.ply = int(state["ply"])

#     def __str__(self):
#         g = self.grid()
#         letters = '  a  b  c  d  e  f  g  h'
#         res = letters + '\n'
#         for i in range(Yolah.DIM):
#             res += str(Yolah.DIM - i)
#             for j in range(Yolah.DIM):
#                 match g[Yolah.DIM - 1 - i][j]:
#                     case Cell.BLACK: res += ' \u25EF '
#                     case Cell.WHITE: res += ' \u2B24 '
#                     case Cell.EMPTY: res += '   '  
#                     case Cell.FREE:  res += ' . '
#             res += str(Yolah.DIM - i) + '\n'
#         res += letters + '\n'
#         res += f'score: {self.black_score}/{self.white_score}'
#         return res

# def distance(square1, square2):
#     d_rank = abs(square1.to_rank() - square2.to_rank())
#     d_file = abs(square1.to_file() - square2.to_file())
#     return max(d_rank, d_file)

# def sliding_moves(square, directions, occupied):
#     moves = 0
#     for dir in directions:
#         sq = square
#         while True:
#             dst = sq + dir
#             if dst is None or (occupied & dst.to_bitboard()) != 0:
#                 break
#             moves |= dst.to_bitboard()
#             sq = dst            
#     return moves

# def pretty_bitboard(bb):
#     s = "+---+---+---+---+---+---+---+---+\n"
#     for rank in range(7, -1, -1):  # RANK_8 down to RANK_1
#         for file in range(8):  # FILE_A to FILE_H
#             square_bit = 1 << (rank * 8 + file)
#             s += "| X " if bb & square_bit else "|   "

#         s += "| " + str(rank + 1) + "\n+---+---+---+---+---+---+---+---+\n"
#     s += "  a   b   c   d   e   f   g   h\n"
#     return s

# def index(magic, shift, bitboard):
#     return magic * bitboard >> shift

# def rank_bb(square):
#     """Returns a bitboard with all squares on the same rank as the given square."""
#     rank = square.to_rank()
#     return 0xFF << (rank * 8)

# def file_bb(square):
#     """Returns a bitboard with all squares on the same file as the given square."""
#     file = square.to_file()
#     return 0x0101010101010101 << file

# def magic_for_square(square, k):
#     rank_1 = 0x00000000000000FF
#     rank_8 = 0xFF00000000000000
#     file_a = 0x0101010101010101
#     file_h = 0x8080808080808080
#     edges = ((rank_1 | rank_8) & bit_not(rank_bb(square))) | ((file_a | file_h) & bit_not(file_bb(square)))
#     moves_bitboard = sliding_moves(square, Direction.horizontal(), 0) & bit_not(edges)
#     occupancies = []
#     moves = []
#     b = size = 0
#     moves_to_occupancies = {}
#     while True:
#         occupancies.append(b)
#         move = sliding_moves(square, Direction.horizontal(), b)
#         moves.append(move)
#         if move not in moves_to_occupancies:
#             moves_to_occupancies[move] = []
#         moves_to_occupancies[move].append(b)
#         size += 1
#         b = (b - moves_bitboard) & moves_bitboard
#         if b == 0:
#             break
#     # for i in range(size):
#     #     print('#' * 80)
#     #     print(pretty_bitboard(occupancies[i]))
#     #     print(pretty_bitboard(moves[i]))
#     print(size)
#     # for m, occ in moves_to_occupancies.items():
#     #     print(f'{m:#x} {len(occ)}')
#     # min_k = len(moves_to_occupancies).bit_length()
#     solver = Solver()
#     magic  = BitVec('magic', 64)
#     shift = 64 - k
#     moves_to_indices = {}
#     indices = {}
#     for m, occ in moves_to_occupancies.items():
#         l = []
#         for bb in occ:
#             l.append(index(magic, shift, bb))
#         for idx in l[1:]:
#             solver.add(idx == l[0])
#         indices.append(l[0])
#     move_list = list(moves_to_indices.keys())
#     n = len(indices)
#     for idx1, idx2 in combinations(indices, 2):
#         solver.add(idx1 != idx2)
#     print('solve')
#     res = solver.check()
#     if res == sat:
#         model = solver.model()
#         m = model[magic].as_long()
#         print(f'Found magic for K = {k}: {m:#x}')
#     elif res == unsat:
#         print('unsat')
#     else:
#         print('unknown')

# if __name__ == '__main__':
#     magic_for_square(Square.SQ_A1, 12)




from z3 import *

from enum import Enum
import json
from itertools import combinations

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
    def diagonal():
        return [Direction.NORTH_EAST, Direction.SOUTH_EAST, Direction.SOUTH_WEST, Direction.NORTH_WEST]

    @staticmethod
    def horizontal():
        return [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

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

    def to_rank(self):
        return self.value // 8

    def to_file(self):
        return self.value % 8

    def to_bitboard(self):
        return 1 << self.sq

    def __str__(self):
        return FILES[self.sq % 8] + str(self.sq // 8 + 1)

    def __add__(self, direction):
        """Add a direction to this square. Returns None if the result is invalid."""
        if not isinstance(direction, Direction):
            return None

        new_sq = self.value + direction.value

        # Check if the new square is out of bounds
        if new_sq < Square.SQ_A1.value or new_sq > Square.SQ_H8.value:
            return None

        # Check for wrapping around files (horizontal wrapping)
        old_file = self.value % 8
        new_file = new_sq % 8

        # If we moved east/west and the file changed by more than 1, we wrapped
        if direction in [Direction.EAST, Direction.WEST]:
            if abs(new_file - old_file) > 1:
                return None
        # For diagonal moves, check both file and rank distance
        elif direction in [Direction.NORTH_EAST, Direction.SOUTH_EAST,
                          Direction.NORTH_WEST, Direction.SOUTH_WEST]:
            if abs(new_file - old_file) != 1:
                return None

        return Square(new_sq)
        
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

def distance(square1, square2):
    d_rank = abs(square1.to_rank() - square2.to_rank())
    d_file = abs(square1.to_file() - square2.to_file())
    return max(d_rank, d_file)

def sliding_moves(square, directions, occupied):
    moves = 0
    for dir in directions:
        sq = square
        while True:
            dst = sq + dir
            if dst is None or (occupied & dst.to_bitboard()) != 0:
                break
            moves |= dst.to_bitboard()
            sq = dst            
    return moves

def pretty_bitboard(bb):
    s = "+---+---+---+---+---+---+---+---+\n"
    for rank in range(7, -1, -1):  # RANK_8 down to RANK_1
        for file in range(8):  # FILE_A to FILE_H
            square_bit = 1 << (rank * 8 + file)
            s += "| X " if bb & square_bit else "|   "

        s += "| " + str(rank + 1) + "\n+---+---+---+---+---+---+---+---+\n"
    s += "  a   b   c   d   e   f   g   h\n"
    return s

def index(magic, shift, bitboard):
    return magic * bitboard >> shift

def rank_bb(square):
    """Returns a bitboard with all squares on the same rank as the given square."""
    rank = square.to_rank()
    return 0xFF << (rank * 8)

def file_bb(square):
    """Returns a bitboard with all squares on the same file as the given square."""
    file = square.to_file()
    return 0x0101010101010101 << file

def magic_for_square(square):
    rank_1 = 0x00000000000000FF
    rank_8 = 0xFF00000000000000
    file_a = 0x0101010101010101
    file_h = 0x8080808080808080
    edges = ((rank_1 | rank_8) & bit_not(rank_bb(square))) | ((file_a | file_h) & bit_not(file_bb(square)))
    moves_bitboard = sliding_moves(square, Direction.horizontal(), 0) & bit_not(edges)
    occupancies = []
    moves = []
    b = size = 0
    moves_to_occupancies = {}
    while True:
        occupancies.append(b)
        move = sliding_moves(square, Direction.horizontal(), b)
        moves.append(move)
        if move not in moves_to_occupancies:
            moves_to_occupancies[move] = []
        moves_to_occupancies[move].append(b)
        size += 1
        b = (b - moves_bitboard) & moves_bitboard
        if b == 0:
            break
    for i in range(size):
        print('#' * 80)
        print(pretty_bitboard(occupancies[i]))
        print(pretty_bitboard(moves[i]))
    print(size)
    for m, occ in moves_to_occupancies.items():
        print(f'{m:#x} {len(occ)}')
    min_k = len(moves_to_occupancies).bit_length()
    for k in range(10, 14):
        solver = Solver()
        magic  = BitVec('magic', 64)
        shift = 64 - k
        moves_to_indices = {}
        for m, occ in moves_to_occupancies.items():
            if len(occ) > 32: continue
            indices = []
            for bb in occ:
                indices.append(index(magic, shift, bb))
            moves_to_indices[m] = indices
        move_list = list(moves_to_indices.keys())
        n = len(move_list)
        for i in range(n):
            #print(f'i = {i}')
            for j in range(i + 1, n):
                #print(f'j = {j}')
                #print(f'{len(moves_to_indices[move_list[i]])} X {len(moves_to_indices[move_list[j]])}')
                for idx1 in moves_to_indices[move_list[i]]:
                    for idx2 in moves_to_indices[move_list[j]]:
                        solver.add(idx1 != idx2)
        print('solve')
        if solver.check() == sat:
            model = solver.model()
            m = model[magic].as_long()
            print(f'Found magic for K = {k}: {m:#x}')
            print(f'Table size: {1 << k} entries')
 

if __name__ == '__main__':
    # import random
    # while not y.game_over():
    #     print(y)
    #     input()
    #     for m in y.moves():
    #         print(m, end=' ')
    #     print()
    #     moves = y.moves()
    #     m = moves[random.randint(0, len(moves) - 1)]
    #     print(m)
    #     y.play(m)
    # print(y)
    magic_for_square(Square.SQ_A1)
