from operator import itemgetter, add
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp


def distributed_load():
    """Computes uniformly distributed loads using the superposition principle"""
    start = float(input("Input starting position: "))
    fin = float(input("Input final position: "))
    if fin <= start:  # testing if the input is correct
        print("Wrong order of position - try again")
        return None
    load = float(input("Input distributed load: "))
    Q = (fin - start) * load  # computes the equivalent concetrated load using "length * load"
    pos_Q = (fin - start) * 0.5 + start  # computes the position of concentrated load
    load_type = "D"
    return Q, pos_Q, load_type, "", load, start, fin


def single_load():
    """Computes singular load"""
    pos = float(input("Input position: "))
    load = float(input("Input load: "))
    load_type = "S"
    return load, pos, load_type, "", "", "", ""


def ask():
    """Function asks the user for the input of loads"""
    loads = []
    while True:
        y = input('Choose load("S" for Single, "D" for Distributed) or get out of loop by typing "X": ').upper()
        if y == "S":
            loads.append(single_load())
        if y == "D":
            d = distributed_load()
            if d != None:  # if the input is not correct it will not be appended to the list of loads
                loads.append(d)
        if y == "X":  # when the user is done typing, while loop breaks
            break
        if y not in ("S", "D", "X"):  # checks the right input
            print('Wrong input, type either "S","D" or "X"')
    return loads


def compute():
    # computes the support reactions of the beam
    a = float(input("Input position of point a: "))
    b = float(input("Input position of point b: "))

    all_loads = []
    left_side = [0]
    right_side = [0]
    for i in ask():
        all_loads.append(i)
        if abs(i[1]) > i[1]:
            left_side[0] = left_side[0] - (i[0] * (b + abs(i[1])))
        elif b < i[1]:
            left_side[0] = left_side[0] + (i[0] * (i[1] - b))
        else:
            left_side[0] = left_side[0] - (i[0] * (b - i[1]))
        if i[1] < 0:
            right_side[0] = right_side[0] + (i[0] * (a + abs(i[1])))
        elif 0 < i[1] < a:
            right_side[0] = right_side[0] + (i[0] * (a - i[1]))
        else:
            right_side[0] = right_side[0] - (i[0] * (i[1] - a))

    Raz = sp.Symbol("Raz")
    Rbz = sp.Symbol("Rbz")
    Ma = sp.Eq(right_side[0] + (b - a) * Rbz, 0)
    Mb = sp.Eq(left_side[0] - (b - a) * Raz, 0)
    for i in sp.solveset(Ma).n():
        Rbz = -i
        all_loads.append(tuple([Rbz, b, "R", "b", "", "", ""]))
    for i in sp.solveset(Mb).n():
        Raz = i
        all_loads.append(tuple([Raz, a, "R", "a", "", "", ""]))
    print(f"Raz = {Raz:.2f}", f"Rbz = {Rbz:.2f}")
    print(all_loads)
# udělat pytest Raz + Rab = ΣF
compute()