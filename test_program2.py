

def ggbabe(a, b):
    if a == 0:
        res = b
    else:
        while b != 0:
            if a > b:
                a = a - b
            else:
                b = b - a
        res = a
    #some = res
    return res


def fibobabe(p):
    if p <= 1:
        res = p
    else:
        a = 0
        b = 1
        for _ in (1, p, 1):
            c = b
            b = a + b
            a = c
        res = b
    return res


def einfaches_babe():
    babe = 2
    return babe