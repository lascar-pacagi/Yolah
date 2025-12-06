from z3 import *

bitscan = {}
for i in range(64):
    bitscan[1 << i] = i

solver = Solver()
magic = BitVec('magic', 64)
k = 6

bitboards = list(bitscan.keys())

def index(magic, k, bitboard):
    return magic * bitboard >> (64 - k)

for i in range(64):
    index1 = index(magic, k, bitboards[i])
    for j in range(i + 1, 64):
        index2 = index(magic, k, bitboards[j])
        solver.add(index1 != index2)

result = solver.check()

if solver.check() == sat:
    model = solver.model()
    m = model[magic].as_long()
    print(f'found magic for k = {k}: {m:x}')
    table = [0] * 64
    for bitboard, pos in bitscan.items():
        #print(hex(bitboard * m >> (64 - k) & (1 << k) - 1))
        table[index(m, k, bitboard) & (1 << k) - 1] = pos
    print(f'constexpr uint8_t bitscan[{1 << k}] = {{')
    for i in range(64):
        print(f'{table[i]},', end='')
    print('\n};')
