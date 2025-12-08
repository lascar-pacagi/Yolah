from z3 import *

positions = {}
for i in range(64):
    positions[1 << i] = i

solver = Solver()
MAGIC  = BitVec('magic', 64)
K = 6
bitboards = list(positions.keys())

def index(magic, k, bitboard):
    return magic * bitboard >> (64 - k)

for i in range(64):
    index1 = index(MAGIC, K, bitboards[i])
    for j in range(i + 1, 64):
        index2 = index(MAGIC, K, bitboards[j])
        solver.add(index1 != index2)

if solver.check() == sat:
    model = solver.model()
    m = model[MAGIC].as_long()
    print(f'found magic for K = {K}: {m:#x}')
    size = 1 << K
    table = [-1] * size
    for bitboard, pos in positions.items():
        #print(hex(bitboard * m >> (64 - k) & (1 << k) - 1))
        table[index(m, K, bitboard) & size - 1] = pos
    print(f'constexpr uint8_t bitscan[{size}] = {{')
    for i in range(size):
        print(f'{table[i]},', end='')
    print('\n};')
