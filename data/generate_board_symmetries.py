files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
rfiles = list(reversed(files))

def diag1():    
    res = [(f + str(j + 1), rfiles[j] + str(i + 1)) for (i, f) in enumerate(rfiles) for j in range(i, 8)]
    print("map<string, string> diag1{", end='')
    for (sq1, sq2) in res:
        if sq1 != sq2: 
            print(f'{{"{sq1}","{sq2}"}},{{"{sq2}","{sq1}"}},', end='')
        else: 
            print(f'{{"{sq1}","{sq2}"}},', end='')
    print("};")
    
def diag2():
    res = [(f + str(j + 1), files[j] + str(i + 1)) for (i, f) in enumerate(files) for j in range(i, 8)]
    print("map<string, string> diag2{", end='')
    for (sq1, sq2) in res:
        if sq1 != sq2: 
            print(f'{{"{sq1}","{sq2}"}},{{"{sq2}","{sq1}"}},', end='')
        else: 
            print(f'{{"{sq1}","{sq2}"}},', end='')
    print("};")

def central_symmetry():
    file_map = dict(zip(files, rfiles))
    rank_map = dict(zip(range(1, 9), range(8, 0, -1)))
    res = set([(f + str(r), file_map[f] + str(rank_map[r])) for f in files for r in range(1, 9)])
    print("map<string, string> central{", end='')
    for (sq1, sq2) in res:         
        print(f'{{"{sq1}","{sq2}"}},', end='')
    print("};")
    
diag1()
diag2()
central_symmetry()

