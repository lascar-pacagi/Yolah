def fp16(b16):
    s = b16[0]
    e = b16[1:6]
    m = b16[6:]
#    print(s, e, m)
    if e == '00000' and m == '0000000000':
#        print(0)
        return 0
    if e == '11111':
        # if m == '0000000000':
        #     print('+inf' if s == '0' else '-inf')            
        # else:
        #     print('nan')
        return
    s_v = 1 if s == '0' else -1
    s_e = 2**(-14 if e == '00000' else int(e, 2) - 15)
    s_m = 0
    for b in reversed(m):
        s_m = 0.5 * s_m + int(b)
    s_m = (1 if e != '00000' else 0) + 0.5 * s_m
    v = s_v * s_e * s_m
#    print(v)
    return v

scaling_factor = 127 / fp16('0111101111111111')

def fp16_to_int8(b16):
    return round(fp16(b16) * scaling_factor)

def print_all():
    for i in range(65536):
        print('---------------------')
        b16 = f'{bin(i)[2:]:0>16}' 
        f = fp16(b16)
        if f is not None: 
            print(b16)
            print(f)
            print(fp16_to_int8(b16))
            if i % 100 == 0: input()

r = dict()

for i in range(-127, 128):
    r[i] = (1000000, -1000000)

for i in range(65536):
    b16 = f'{bin(i)[2:]:0>16}'
    f = fp16(b16)
    if f is None: continue
    f = f / fp16('0111101111111111')
    int8 = fp16_to_int8(b16)
    mn, mx = r[int8]
    r[int8] = (min(mn, f), max(mx, f))

