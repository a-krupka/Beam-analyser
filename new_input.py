from operator import itemgetter, add
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
import copy #needed to deepcopy my absolute positions list


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
    return [Q, pos_Q, load_type, "", load, start, fin]


def single_load():
    """Computes singular load"""
    pos = float(input("Input position: "))
    load = float(input("Input load: "))
    load_type = "S"
    return [load, pos, load_type, "", "", "", ""]
def point_moment():
    """Computes point moment"""
    pos = float(input("Input position: "))
    load = float(input("Input load: "))
    load_type = "M"
    return [load, pos, load_type, "", "", "", ""]


def ask():
    """Function asks the user for the input of loads"""
    loads = []
    while True:
        y = input('Choose load("S" for Single, "D" for Distributed, "M" for point moment) or get out of loop by typing "X": ').upper()
        if y == "S":
            loads.append(single_load())
        if y == "D":
            d = distributed_load()
            if d != None:  # if the input is not correct it will not be appended to the list of loads
                loads.append(d)
        if y == "M":
            loads.append(point_moment())
        if y == "X":  # when the user is done typing, while loop breaks
            break
        if y not in ("S", "D","M", "X"):  # checks the right input
            print('Wrong input, type either "S","D" or "X"')
    return loads


def compute():
    # computes the support reactions of the beam
    a = float(input("Input position of point a: "))
    b = float(input("Input position of point b: "))
    a_absolute = a
    b_absolute = b

    all_loads = [] # list keeping track of absolute positions written by the user
    for i in ask():
        all_loads.append(i)
    relative_pos = copy.deepcopy(all_loads) # list keeping track of relative positions in case of converting negative postions to positive
    positions = [i[1] if i[2] != 'D' else i[5] for i in all_loads]
    starting_pos = min(positions)  # gets the starting (most left) position of force
    print("Minimal position is ",starting_pos)
    if starting_pos < 0:
        a += abs(starting_pos)
        b += abs(starting_pos)
        for i in relative_pos:
            i[1] += abs(starting_pos)

    left_side = [0]
    right_side = [0]
    for i in relative_pos:
        if i[2] !="M":
            if b < i[1]:
                left_side[0] = left_side[0] - (i[0] * (i[1] - b))
            else:
                left_side[0] = left_side[0] + (i[0] * (b - i[1]))

            if a < i[1]:
                right_side[0] = right_side[0] - (i[0] * (i[1] - a))
            else:
                right_side[0] = right_side[0] + (i[0] * (a - i[1]))

    print(right_side)
    print(left_side)

    moments_sum = 0 # creating a variable for potential sum of point moments
    bool_mom = False
    for i in relative_pos:
        if i[2] == "M":
            moments_sum += i[0]
            bool_mom = True
            print(moments_sum,"here")
        if i[2] == "D":
            i[5] += abs(starting_pos)
            i[6] += abs(starting_pos)
    if bool_mom == True:
        Rbz = (right_side[0] + moments_sum) / (b - a)
        Raz = (left_side[0] + moments_sum) / (-(b - a))
    else:
        Rbz = (right_side[0]) / (b - a)
        Raz = (left_side[0]) / (-(b - a))

    all_loads.append([Rbz, b_absolute, "R", "b", "", "", ""])
    all_loads.append([Raz, a_absolute, "R", "a", "", "", ""])
    relative_pos.append([Rbz, b, "R", "b", "", "", ""])
    relative_pos.append([Raz, a, "R", "a", "", "", ""])


    print(f"Raz = {Raz:.2f}", f"Rbz = {Rbz:.2f}")
    print("all = ",all_loads)
    print("relative = ",relative_pos)
# udělat pytest Raz + Rab = ΣF
compute()