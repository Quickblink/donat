

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


def bubble_babe(a, l):
    for i in (0, l, 1):
        for k in (1, l - i, 1):
            km1 = k - 1
            ak = a[k]
            akm1 = a[km1]
            if akm1 > ak:
                a[km1] = ak
                a[k] = akm1


def selection_babe(a, l):
    for i in (0, l, 1):
        ami = 10000
        for k in (i, l, 1):
            if a[k] < ami:
                mi = k
                ami = a[mi]
        a[mi] = a[i]
        a[i] = ami


def teile_babe(l, links, rechts):
    i = links
    j = rechts - 1
    p = l[rechts]
    while i < j:
        while i < rechts and l[i] < p:
            i = i + 1
        while j > links and l[j] >= p:
            j = j - 1
        if i < j:
            tmp = l[i]
            l[i] = l[j]
            l[j] = tmp
    if l[i] > p:
        tmp = l[i]
        l[i] = l[rechts]
        l[rechts] = tmp
    return i


def quick_babe(l, links, rechts):
    if links < rechts:
        t = teile_babe(l, links, rechts)
        quick_babe(l, links, t - 1)
        quick_babe(l, t + 1, rechts)


def rec_test(a):
    if a:
        return rec_test(a-1)
    return 45


def main():
    #a = fibobabe(10)
    #b = ggbabe(100, 200)
    a = 123456 #[]
    # selection_babe(a, 100)
    # bubble_babe(a, 100)
    #quick_babe(a, 0, 99)
    # c = rec_test(10)
    #t = teile_babe(a, 0, 4)
    #i = 0
    #while i < 4 and a[i] < 3:
    #    i = i + 1
    return 500 * 800
