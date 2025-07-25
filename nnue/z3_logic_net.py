import os
from tqdm import tqdm
import re
import glob
import sys
sys.path.append("../server")
from yolah import Yolah, Move
import itertools
from z3 import *
import random

def bit_not(n, numbits=64):
    return (1 << numbits) - 1 - n

def bitboard64_to_list(n):
    b = [int(digit) for digit in bin(n)[2:]]
    return [0]*(64 - len(b)) + b

class GameDataIterator:
    def __init__(self, data):
        self.data = data
        self.count = 0
        self.max_count = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.max_count:
            raise StopIteration
        res = self.data[self.count]
        self.count += 1        
        return res

class GameData:
    GAME_RE  = re.compile(r"""(\w\d:\w\d)""")
    SCORE_RE = re.compile(r"""(\d+)/(\d+)""")
    
    def __init__(self, games_dir):
        self.inputs = []
        self.outputs = []
        self.infos = []
        nb_positions = 0
        random_moves_re = re.compile(r""".*(\d)r.*""")
        for filename in glob.glob(games_dir + "/games*"):
            l = random_moves_re.findall(filename)
            if l != []:
                r = int(l[0])
                with open(filename) as f:
                    for line in f:
                        moves = GameData.GAME_RE.findall(line)
                        nb_positions += len(moves[r:]) + 1
                        self.inputs.append((nb_positions, moves))
                        black_score, white_score = GameData.SCORE_RE.findall(line)[0]                              
                        black_score = int(black_score)
                        white_score = int(white_score)
                        res = 1
                        if black_score > white_score: 
                            res = 0
                        elif white_score > black_score: 
                            res = 2                       
                        self.outputs.append(res)
                self.infos.append((filename, r, nb_positions))
        self.size = nb_positions

    @staticmethod
    def encode_yolah(yolah):
        res = list(itertools.chain.from_iterable([
                    bitboard64_to_list(yolah.black), 
                    bitboard64_to_list(yolah.white), 
                    bitboard64_to_list(yolah.empty),                    
                    [Yolah.WHITE_PLAYER if yolah.nb_plies() & 1 else Yolah.BLACK_PLAYER]]))
        return res

    @staticmethod
    def encode(moves):
        yolah = Yolah()
        for m in map(lambda m: Move.from_str(m), moves):
            yolah.play(m)
        return GameData.encode_yolah(yolah)

    def get_infos(self):
        return self.infos
    
    def get_nb_random_moves(self, idx):
        for (_, r, n) in self.infos:
            if idx < n: return r
        print(n, idx)
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
        return GameData.encode(moves[: r + idx - n]), self.outputs[lo]

    def __iter__(self):
        return GameDataIterator(self)

# +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+        
# | A | B | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12| 13| 14| 15|
# +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
# +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 |
# +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
# +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
# | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
# +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+

# 0: FALSE           8: A AND B
# 1: A NOR B         9: ~(A XOR B)
# 2: ~(B=>A)        10: B 
# 3: ~A             11: A=>B
# 4: ~(A=>B)        12: A
# 5: ~B             13: B=>A
# 6: A XOR B        14: A OR B
# 7: A NAND B       15: TRUE

# black positions + white positions + empty positions + turn 
INPUT_SIZE = 64 + 64 + 64 + 1

class Expression:
    cache = {}
    def __init__(self, *children):
        self.children = children

    def add(self, child):
        self.children.append(child)

    def eval(self, inputs):
        raise NotImplementedError
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        raise NotImplementedError

    @staticmethod
    def reset():
        cache = {}

    @staticmethod
    def memo(e, inputs):
        if e in Expression.cache:
            return Expression.cache[e]
        v = e.eval(inputs)
        Expression.cache[e] = v
        return v

class EInput(Expression):
    def __init__(self, id):
        super().__init__()
        self.id = id
    
    def eval(self, inputs):
        return inputs[self.id]

    def __repr__(self):
        return f'I_{self.id}'

class EFalse(Expression):
    def __init__(self):
        super().__init__()
    
    def eval(self, inputs):
        return False

    def __repr__(self):
        return "F"

class ENor(Expression):
    def __init__(self, i1, i2):
        super().__init__(i1, i2)

    def eval(self, inputs):
        v1 = Expression.memo(self.children[0], inputs)
        v2 = Expression.memo(self.children[1], inputs)
        res = not(v1 or v2)
        Expression.cache[self] = res
        return res

    def __repr__(self):
        return f'({str(self.children[0])} nor {str(self.children[1])})'

def random_indices(n1, n2):
    return [random.randint(0, n1 - 1) for _ in range(2 * n2)]

class LogicNet:
    def __init__(self, data, layers):
        self.data = data
        self.solver = Solver()
        self.input_indices = [
            random_indices(n1, n2) for n1, n2 in zip(layers, layers[1:])
        ]
        self.selects = [
            [BitVec(f'sel_{i}_{j}', 4) for j in range(n)] for i, n in enumerate(layers[1:]) 
        ]
        # print(self.input_indices)
        # print(self.selects)
        self.variables = self.selects[:]        
        self._add_constraints()

    def _add_constraints(self):
        for i, (input, res) in enumerate(self.data):
            print(i)
            print(input, res)
            input_vars = []
            for j, value in enumerate(input):
                var = Bool(f'I_{i}_{j}')
                self.variables.append(var)
                input_vars.append(var)
                self.solver.add(var if value else Not(var))
            previous_layer = input_vars
            for j, indices in enumerate(self.input_indices):
                output_vars = []
                for k, (i1, i2) in enumerate(zip(indices[::2], indices[1::2])):                     
                     output = self._gate_output(previous_layer, i, j, k, i1, i2)
                     self.variables.append(output)
                     output_vars.append(output)
                previous_layer = output_vars
            black = BitVec(f'black_{i}', 16)
            draw  = BitVec(f'draw_{i}', 16)
            white = BitVec(f'white_{i}', 16)
            self.variables += [black, draw, white]
            n = len(previous_layer) // 3
            self._add_output_constraint(previous_layer[0:n], black)
            self._add_output_constraint(previous_layer[n:2*n], draw)
            self._add_output_constraint(previous_layer[2*n:], white)
            if res == 0:
                self.solver.add(black > draw, black > white)
            elif res == 1:
                self.solver.add(draw > black, draw > white)
            else:
                self.solver.add(white > black, white > draw)

    def _add_output_constraint(self, prev, output):
        self.solver.add(output == Sum([If(v, BitVecVal(1, 16), BitVecVal(0, 16)) for v in prev]))

    def _gate_output(self, prev, i, j, k, i1, i2):
        in1 = prev[i1]
        in2 = prev[i2]
        sel = self.selects[j][k]
        out = Bool(f'out_{i}_{j}_{k}')
        self.solver.add(
            If(sel == 0, out == False,
            If(sel == 1, out == Not(Or(in1, in2)),
            If(sel == 2, out == Not(Implies(in2, in1)),
            If(sel == 3, out == Not(in1),
            If(sel == 4, out == Not(Implies(in1, in2)),
            If(sel == 5, out == Not(in2),
            If(sel == 6, out == Xor(in1, in2),
            If(sel == 7, out == Not(And(in1, in2)),
            If(sel == 8, out == And(in1, in2),
            If(sel == 9, out == Not(Xor(in1, in2)),
            If(sel == 10, out == in2,
            If(sel == 11, out == Implies(in1, in2),
            If(sel == 12, out == in1,
            If(sel == 13, out == Implies(in2, in1),
            If(sel == 14, out == Or(in1, in2), out == True
        ))))))))))))))))
        return out

    def solve(self):
        if self.solver.check() == unsat:
            return None
        return self.solver.model()

    def expression(self):
        pass

    def c_expression(self):
        pass

data = GameData("data/data_test")
net = LogicNet(data, [INPUT_SIZE, 1000, 10, 90])
print(net.solve())
# x1 = EInput('0')
# x2 = EInput('1')
# x3 = ENor(x1, x2)
# print(x3.eval({'0': False, '1': False}))
