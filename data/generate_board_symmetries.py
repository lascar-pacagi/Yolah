files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

def diag1():
    rfiles = list(reversed(files))
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
    
diag1()
diag2()
