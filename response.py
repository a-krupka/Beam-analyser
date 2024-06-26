from operator import itemgetter, add
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil as ceil



def sign_pos(num):
    """"
    sign function
    """
    if num > 0:
        return 0
    else:
        return 1
#Konzola z obou stran - funguje
all_loads = [(12.0, 0.0, 'S', '', '', '', ''), (-83.8000000000000, 3.0, 'R', 'a', '', '', ''), (100.0, 4.5, 'D', '', 20.0, 2.0, 7.0), (-46.2000000000000, 8.0, 'R', 'b', '', '', ''), (18.0, 9.5, 'S', '', '', '', '')]
#jinak deflections
all_loads = [[10.0, 0.0, 'S', '', '', '', '', '', ''], [45.0, 3.5, 'D', '', 15.0, 2.0, 5.0, '', ''], [11.0, 8.0, 'S', '', '', '', '', '', ''], [-35.9, 6.0, 'R', 'b', '', '', ''], [-30.1, 1.0, 'R', 'a', '', '', '']]
#Naděje - funguje
#all_loads=[(80.0, 2.0, 'D', '', 20.0, 0.0, 4.0), (80.0, 10.0, 'D', '', 20.0, 8.0, 12.0), (-80.0000000000000, 8.0, 'R', 'b', '', '', ''), (-80.0000000000000, 4.0, 'R', 'a', '', '', '')]
# Mohrovka, příklad C
#all_loads = [(-3.60000000000000, 0.0, 'R', 'a', '', '', ''), (2.0, 2.0, 'S', '', '', '', ''), (6.0, 3.0, 'S', '', '', '', ''), (-4.40000000000000, 5.0, 'R', 'b', '', '', '')]
#Mohrovka zprava
#all_loads =[(-1.50000000000000, 0.0, 'R', 'a', '', '', ''), (12.0, 3.5, 'D', '', 4.0, 2.0, 5.0), (-17.5000000000000, 5.0, 'R', 'b', '', '', ''), (7.0, 6.5, 'S', '', '', '', '')]
#Mohrovka příklad F
#all_loads = [(-35.9500000000000, 0.0, 'R', 'a', '', '', ''), (65.0, 4.5, 'D', '', 13.0, 2.0, 7.0), (2.0, 9.0, 'S', '', '', '', ''), (-31.0500000000000, 10.0, 'R', 'b', '', '', '')]
#Mohrovka zprava i zleva # opraveno
#all_loads = [[10.0, 0.0, 'S', '', '', '', ''], [20.0, 1.0, 'S', '', '', '', ''], [-87.33333333333333, 1.0, 'R', 'a', '', '', ''], [25.0, 4.0, 'S', '', '', '', ''], [132.0, 7.0, 'D', '', 33.0, 5.0, 9.0], [-179.66666666666666, 10.0, 'R', 'b', '', '', ''], [80.0, 10.5, 'S', '', '', '', '']]
#Mohrovka, příklad ?
#all_loads=[(-5.84615384615385, 0.0, 'R', 'a', '', '', ''), (2.0, 1.0, 'S', '', '', '', ''), (9.0, 3.5, 'D', '', 3.0, 2.0, 5.0), (-5.15384615384615, 6.5, 'R', 'b', '', '', '')]
#Mohrovka převislý konec zleva
#all_loads = [(16.0, 2.0, 'D', '', 4.0, 0.0, 4.0), (-21.3333333333333, 4.0, 'R', 'a', '', '', ''), (5.33333333333333, 10.0, 'R', 'b', '', '', '')]
#Mohrovka převislý konec zprava
#all_loads = [(60.0, 7.0, 'D', '', 10.0, 4.0, 10.0), (-105.000000000000, 4.0, 'R', 'b', '', '', ''), (45.0000000000000, 0.0, 'R', 'a', '', '', '')]

#all_loads = [[16.0, 8.0, 'D', '', 4.0, 6.0, 10.0], [-21.333333333333332, 6.0, 'R', 'b', '', '', ''], [5.333333333333333, 0.0, 'R', 'a', '', '', '']]

#all_loads = [[16.0, 2.0, 'D', '', 4.0, 0.0, 4.0], [-21.333333333333332, 4.0, 'R', 'a', '', '', ''], [5.333333333333333, 10.0, 'R', 'b', '', '', '']]
#Mohrovka jednoduchý příklad
#all_loads = [(-12.0000000000000, 0.0, 'R', 'a', '', '', ''), (24.0, 5.0, 'D', '', 4.0, 2.0, 8.0), (-12.0000000000000, 10.0, 'R', 'b', '', '', '')]
#Misa příklad
#all_loads = [(24.0, 4.0, 'D', '', 4.0, 1.0, 7.0), (-12.0000000000000, 8.0, 'R', 'b', '', '', ''), (-12.0000000000000, 0.0, 'R', 'a', '', '', '')]
#momenty
#all_loads = [(5.0, 2.0, 'M', '', '', '', ''), (-7.0, 3.5, 'M', '', '', '', ''), (-0.5, 4.0, 'R', 'b', '', '', ''), (0.5, 0.0, 'R', 'a', '', '', '')]
#prostý ohyb
#all_loads =[(100.0, 0.0, 'M', '', '', '', ''), (-100.0, 5.0, 'M', '', '', '', ''), (0.0, 5.0, 'R', 'b', '', '', ''), (-0.0, 0.0, 'R', 'a', '', '', '')]
#prostý ohyb v půlce
#all_loads=[(100.0, 2.5, 'M', '', '', '', ''), (20.0, 5.0, 'R', 'b', '', '', ''), (-20.0, 0.0, 'R', 'a', '', '', '')]
#
#all_loads = [(15.0, 5.0, 'M', '', '', '', ''), (6.0, 1.5, 'D', '', 2.0, 0.0, 3.0), (6.0, 7.0, 'R', 'b', '', '', ''), (-12.0, 3.0, 'R', 'a', '', '', '')]
#new one with moments
#all_loads = [[10.0, -3.0, 'S', '', '', '', ''], [50.0, -2.0, 'M', '', '', '', ''], [20.0, 1.0, 'S', '', '', '', ''], [40.0, 5.5, 'D', '', 8.0, 3.0, 8.0], [50.0, 9.5, 'M', '', '', '', ''], [-27.5, 4.0, 'R', 'b', '', '', ''], [-42.5, 0.0, 'R', 'a', '', '', '']]
#relative of the one above
#all_loads = [[10.0, 0.0, 'S', '', '', '', ''], [50.0, 1.0, 'M', '', '', '', ''], [20.0, 4.0, 'S', '', '', '', ''], [40.0, 8.5, 'D', '', 8.0, 6.0, 11.0], [50.0, 12.5, 'M', '', '', '', ''], [-27.5, 7.0, 'R', 'b', '', '', ''], [-42.5, 3.0, 'R', 'a', '', '', '']]
#Triangl
#all_loads =  [[15.0, 5.0, 'T_P', '', 10.0, 3.0, 6.0, 3.0], [-10.714285714285714, 7.0, 'R', 'b', '', '', ''], [-4.285714285714286, 0.0, 'R', 'a', '', '', '']]
#Triangl 2
#all_loads = [[7.5, 3.0, 'T_L', '', 5.0, 2.0, 5.0, 3.0], [-4.5, 5.0, 'R', 'b', '', '', ''], [-3.0, 0.0, 'R', 'a', '', '', '']]
#kombinace Trianglu
#all_loads = [[40.0, 2.6666666666666665, 'T_P', '', 20.0, 0.0, 4.0, 4.0], [40.0, 5.333333333333333, 'T_L', '', 20.0, 4.0, 8.0, 4.0], [50.0, 8.0, 'M', '', '', '', ''], [20.0, 9.0, 'S', '', '', '', ''], [30.0, 11.5, 'D', '', 10.0, 10.0, 13.0], [-45.833333333333336, 10.0, 'R', 'b', '', '', ''], [-84.16666666666667, 4.0, 'R', 'a', '', '', '']]

#all_loads = [[60.0, 2.6666666666666665, 'T_P', '', 30.0, 0.0, 4.0, 4.0], [-73.33333333333334, 4.0, 'R', 'a', '', '', ''], [13.333333333333336, 10.0, 'R', 'b', '', '', '']]
#parabola
#all_loads = [[-6.666666666666667, 0.0, 'R', 'a', '', '', ''], [26.666666666666668, 3.75, 'P_P', '', 16.0, 0.0, 5.0, 5.0, 2], [-20.0, 5.0, 'R', 'b', '', '', '']]
#parabola
#all_loads = [[1.8, 4.5, 'P_P', '', 0.9, 0.0, 6.0, 6.0, 2], [1.7999999999999998, 3.0, 'D', '', 0.3, 0.0, 6.0, '', ''], [-2.25, 6.0, 'R', 'b', '', '', ''], [-1.3499999999999999, 0.0, 'R', 'a', '', '', '']]

#all_loads= [[12.0, 2.25, 'P_P', '', 12.0, 0.0, 3.0, 3.0, 2], [-5.4, 5.0, 'R', 'b', '', '', ''], [-6.6, 0.0, 'R', 'a', '', '', '']]
#all_loads = [[12.0, 2.75, 'P_L', '', 12.0, 2.0, 5.0, 3.0, 2], [-6.6, 5.0, 'R', 'b', '', '', ''], [-5.4, 0.0, 'R', 'a', '', '', '']]
#all_loads = [[12.0, 3.2, 'P_P', '', 12.0, 0.0, 4.0, 4.0, 3], [-7.6800000000000015, 5.0, 'R', 'b', '', '', ''], [-4.319999999999999, 0.0, 'R', 'a', '', '', '']]

#all_loads = [[5.0, 0.0, 'M', '', '', '', '', '', ''], [1.0, 5.0, 'R', 'b', '', '', ''], [-1.0, 0.0, 'R', 'a', '', '', '']]
#chyba
#all_loads = [[10.0, 0.0, 'S', '', '', '', '', '', ''], [10.0, 9.0, 'S', '', '', '', '', '', ''], [-10.0, 7.0, 'R', 'b', '', '', ''], [-10.0, 2.0, 'R', 'a', '', '', '']]
#nefungují
all_loads = [[26.666666666666668, 3.0, 'P_P', '', 20.0, 0.0, 4.0, 4.0, 2], [60.0, 7.0, 'D', '', 15.0, 5.0, 9.0, '', ''], [22.5, 11.0, 'P_L', '', 18.0, 10.0, 15.0, 5.0, 3], [-51.805555555555564, 10.0, 'R', 'b', '', '', ''], [-57.361111111111114, 4.0, 'R', 'a', '', '', '']]
#funguje?
all_loads = [[25.0, 2.0, 'S', '', '', '', '', '', ''], [16.666666666666668, 6.75, 'P_P', '', 10.0, 3.0, 8.0, 5.0, 2], [-20.3125, 8.0, 'R', 'b', '', '', ''], [-21.354166666666668, 0.0, 'R', 'a', '', '', '']]
#all_loads = [[16.666666666666668, 1.25, 'P_L', '', 10.0, 0.0, 5.0, 5.0, 2], [25.0, 6.0, 'S', '', '', '', '', '', ''], [-21.354166666666668, 8.0, 'R', 'b', '', '', ''], [-20.3125, 0.0, 'R', 'a', '', '', '']]

positions = [i[1] if i[2] not in ('D','T_L','T_P') else i[5] for i in all_loads] # gets the position of a load, in case of distributed load, outputs starting position
reaction_a = float("".join([str(i[1]) for i in all_loads if i[3] == 'a'])) #computing position of reaction a by list comprehension, then converting into single float
reaction_b = float("".join([str(i[1]) for i in all_loads if i[3] == 'b'])) #computing position of reaction a by list comprehension, then converting into single float
print(f"a = {reaction_a} b = {reaction_b}")
starting_pos = min(positions) # gets the starting (most left) position of force
t = starting_pos
accuracy = 1e-3
inverse_accuracy  = accuracy ** (-1)

print(positions,f"min pos is {min(positions)}")
all_loads= (sorted(all_loads,key=itemgetter(1)))
print(all_loads)
for i in all_loads:
    if i[2] == "D":
        if i[6] > max(positions):
            positions.append(i[6])
print(positions)
if t <= 0: #tady tweakuju
    max_pos = max(positions) + abs(t) + accuracy
    a= np.arange(accuracy,max_pos,accuracy)
    odchylka = 0
    inverse_odchylka = abs(t)
    posun = abs(t)
else:
    max_pos = max(positions) + accuracy
    a = np.arange(accuracy+t, max_pos, accuracy)
    odchylka = abs(t)
    inverse_odchylka = 0
    posun = 0
Vy = [0.0 for i in a] #Posouvačky # making a list of len(a) of every value 0.0 to which we'll be adding results of for loop
Vx = [i for i in a]
Vnun=[0.0 for i in a] # auxillary temporary list which stores results of most recent iteration

### zkusit změnit pořadí ve for loopu

for j in range(len(all_loads)):

    if Vnun != []:
        Vy = list(map(add, Vy, Vnun)) #adds the temporary list of values Vnun to Vy ,add is a function imported from operators
    Vnun = [] # restart of the temporary list
    if all_loads[j][2] == "S" or all_loads[j][2]=="R":
        for i in a:
            result = sign_pos(all_loads[j][1] + abs(t) - i) * all_loads[j][0]
            Vnun.append(result) # adding the result of each point load in shear
    if all_loads[j][2] == "D":
        for i in a:
            result = +sign_pos(all_loads[j][5]+abs(t) - i)*all_loads[j][4]*(i-(all_loads[j][5]+abs(t))) \
                     - sign_pos(all_loads[j][6]+abs(t) - i)*all_loads[j][4]*(i-(all_loads[j][6]+abs(t)))
            Vnun.append(result) # adding the result of each uniformly distributed load load in shear
    if all_loads[j][2] == "M":
        pass
    if all_loads[j][2] == "T_P":
        for i in a:
            q = (all_loads[j][4] + abs(t))
            L = all_loads[j][7]
            x = (i - (abs(t) + all_loads[j][5]))
            if all_loads[j][5] <= i <= all_loads[j][6]:
                result = ((q / L) * x**2/2)  # ((q / L) * x**2/2)

            elif all_loads[j][6] < i:
                result = (q * L)/2    # (q * L)/2
            else:
                result = 0
            Vnun.append(result)

    if all_loads[j][2] == "T_L":
        for i in a:
            if all_loads[j][5] <= i <= all_loads[j][6]:
                q = (all_loads[j][4] + abs(t))
                L = all_loads[j][7]
                x = (i - (abs(t) + all_loads[j][5]))
                result =  (q / L) * x**2/2 + x * (q - (q/L *x))                # ((q / L) * x**2/2 + x**2/2 * (q - (q/L *x))
            elif all_loads[j][6] < i:
                result = (q * L)/2  # (q * L)/2
            else:
                result = 0
            Vnun.append(result)
    if all_loads[j][2] == "P_P":
        for i in a:
            q = (all_loads[j][4] + abs(t))
            L = all_loads[j][7]
            x = (i - (abs(t) + all_loads[j][5]))
            n = all_loads[j][8]
            if all_loads[j][5] <= i <= all_loads[j][6]:
                result = ((q / L**n) * (x**(n+1))/(n+1))

            elif all_loads[j][6] < i:
                result = (q * L)/(n+1)    # (q * L)/2
            else:
                result = 0
            Vnun.append(result)
    if all_loads[j][2] == "P_L":
        for i in a:
            if all_loads[j][5] <= i <= all_loads[j][6]:
                q = (all_loads[j][4] + abs(t))
                L = all_loads[j][7]
                x = (i - (abs(t) + all_loads[j][5])) #+accuracy
                n = all_loads[j][8]
                h = q - ((q / L ** n) * (L-x) ** n)
                #h = ((q / L ** n) * x ** n)
                result =  (q*L-(q/L**n)*(L-x)**(n+1))/(n+1)               # ((q / L) * x**2/2 + x**2/2 * (q - (q/L *x))
            elif all_loads[j][6] < i:
                result = (q * L)/(n+1)  # (q * L)/2
            else:
                result = 0
            Vnun.append(result)

    ### potentional place for additional types of forces – "Triangular" "Parabolic" etc

### zkusit změnit pořadí ve for loopu ^^^^^^
print("Halo")
if Vnun != []:
    Vy = list(map(add, Vy, Vnun)) # adding the last remaining values
V_maxima = []
V_temp = [] # temporary list to check for duplicates in V_maxima
try:
    for j in range(len(Vy)):
        if (Vy[j] > Vy[j+1] + 1) or (Vy[j] +1 < Vy[j+1]):
            V_maxima.extend([[Vy[j],j]])
            V_maxima.extend([[Vy[j+1], j+1]])

except: IndexError
for i,j in zip(V_maxima,range(len(V_maxima))):
    if i[0] in V_temp:
        V_maxima.pop(j)
    V_temp.append(i[0])
print(V_maxima,"V maxima")
Vy[0] = 0 # rewriting the first value on the y axis as 0 so it starts from zero

xvpoints = np.array(Vx) # making a numpy array from points on the x-axis of shear force diagram
yvpoints = np.array(Vy) # making a numpy array from points on the y-axis of shear force diagram
print(f"length of x {len(xvpoints)}, length of y {len(yvpoints)}")



### Moment diagram
y = [0.0 for i in a]
x = [i for i in a]

nun=[0.0 for i in a]
for j in range(len(all_loads)):
    if nun != []:
        y = list(map(add, y, nun))
    nun = []
    if all_loads[j][2] == "S" or all_loads[j][2]=="R":
        for i in a:
            result = sign_pos(all_loads[j][1] + abs(t) - i) * all_loads[j][0] * (i - (abs(t) + (all_loads[j][1])))
            nun.append(result)
    if all_loads[j][2] == "D":
        for i in a:
            result = sign_pos(all_loads[j][5]+abs(t) - i)*all_loads[j][4]/2 * (i-(abs(t)+all_loads[j][5]))**2 \
                     - sign_pos(all_loads[j][6]+abs(t) - i)*all_loads[j][4]/2 * (i-(abs(t)+all_loads[j][6]))**2
            nun.append(result)
    if all_loads[j][2] == "T_L":
        for i in a:
            if all_loads[j][5] <= i <= all_loads[j][6]:
                result = (((((all_loads[j][4] + abs(t))/all_loads[j][7])*(i - (abs(t) + all_loads[j][5]))**3)/3) + (((all_loads[j][4] + abs(t))-((all_loads[j][4] + abs(t))/all_loads[j][7]*(i - (abs(t) + all_loads[j][5]))))*(i - (abs(t) + all_loads[j][5])))*(i - (abs(t) + all_loads[j][5]))/2)  # ((((q/L)*x**3)/3) + ((q-(q/L*x))*x)*x/2)

            elif all_loads[j][6] < i:
                result = (((all_loads[j][4]+abs(t)) * all_loads[j][7]) / 2) * ((i-(abs(t)+all_loads[j][5])) - all_loads[j][7] + 2* all_loads[j][7] / 3) # ((q * L)/2) * (x-L + 2*L/3)
                #není komplet   # (q * L)/2
            else:
                result = 0
            nun.append(result)
    if all_loads[j][2] == "T_P":
        for i in a:
            if all_loads[j][5] <= i <= all_loads[j][6]:
                result = (all_loads[j][4]+abs(t))/all_loads[j][7] * (i-(abs(t)+all_loads[j][5]))**3 / 6 # ((q / L) * x**3/6)
            elif all_loads[j][6] < i:
                result = (((all_loads[j][4]+abs(t)) * all_loads[j][7]) / 2) * ((i-(abs(t)+all_loads[j][5])) - all_loads[j][7] + all_loads[j][7] / 3) # ((q * L)/2) * (x-L + L/3)
            else: result = 0
            nun.append(result)
    if all_loads[j][2] == "M":
        for i in a:
            result = sign_pos(all_loads[j][1] + abs(t) - i) * all_loads[j][0]
            nun.append(result) # adding the result of each point moment in bending
    if all_loads[j][2] == "P_P":
        q = all_loads[j][4] + abs(t)
        n = all_loads[j][8]
        L = all_loads[j][7]

        for i in a:
            if all_loads[j][5] <= i <= all_loads[j][6]:
                i = (i - (abs(t) + all_loads[j][5]))
                h = (q / L ** n) * i ** n
                A = (h * i) / (n + 1)
                result = A * 1 / (n + 2) * i
                with open("parabolyy.csv","a") as file:
                    file.write(f"{A,i,result}\n")
                    file.close()

            elif all_loads[j][6] < i:
                h = (q / L ** n) * L ** n
                A = (h * L) / (n + 1)
                result = A * ((1/(n+2))*L + i-L)  # ((q * L)/2) * (x-L + L/3)

            else:
                result = 0
            nun.append(result)
    if all_loads[j][2] == "P_L":
        q = all_loads[j][4]
        n = all_loads[j][8]
        L = all_loads[j][7]

        for i in a:
            if all_loads[j][5] <= i <= all_loads[j][6]:
                i= i - all_loads[j][5] #udělat to stejně jako u posouvaček
                #h_low = ((q / L ** n) * (L-i)**(n))
                #A_rect = h_low * i
                #A = (h * i) / (n + 1)
                A = ((q * L) - (q / L ** n) * (L-i)**(n+1)) / (n + 1)
                result = A * (((1/A) * ((L**(n+2)-(L-i)**(n+2))/(n+2)) * q/ L**2) - (L-i)) #+ (q - h) * i ** 2 / 2
                if i == 4:
                    print("hh")
            elif all_loads[j][6] < i:
                h = (q / L ** n) * L ** n
                A = (h * L) / (n + 1)
                result = A * (((n+1) / (n + 2)) * L + i - L)
            else:
                result = 0
            nun.append(result)
y = list(map(add, y, nun))

#for i,j in zip(x,y):
#    print(i,j)
xy = [(x[i],y[i]) for i in range(len(y))]

xpoints = np.array(x)
ypoints = np.array(y)
with open("fff.csv", "w") as file:
    for i in ypoints:
        file.write(f"{i},\n")

mpoints = np.array(xy)
print(len(xvpoints),len(xpoints),"tady dole")
print(min(xpoints))
if abs(min(ypoints)) < abs(max(ypoints)): # computing if the maximal bending moment is positive or negative
    Mmax = max(ypoints)
else:
    Mmax= min(ypoints)
gs = gridspec.GridSpec(3, 2) # specifying how will the matplotlib grid look like
fig = plt.figure()
### setting up the Load diagram
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title(r"$Loads$")
ax1.plot([0+abs(t),0+abs(t)],[min(ypoints-10),max(ypoints)+10],linestyle = "dotted",color = "#bfbfbf")
ax1.plot([min(xpoints),max(xpoints)],[0,0],linestyle = "dotted",color = "#bfbfbf")
#plt.xticks(np.arange(t, positions[-1]+abs(t), step=1),labels=[i-abs(t) for i in range(round(positions[0]),ceil(positions[-1]+abs(t)))])
ax1.plot([min(xpoints)+odchylka,max(xpoints)+odchylka],[0,0],color = "black",alpha=0.8)   #black line on x-axis
for i in all_loads:
    if i[3] == "a":
        ax1.plot(i[1]+abs(t),-abs(0.01*min(ypoints)),marker = "^", color = "r")
        ax1.annotate("a",[i[1]+abs(t),0],[i[1]+abs(t)+(0.015*all_loads[-1][1]),-abs(0.05*min(ypoints))],color = "r")
    if i[3] == "b":
        ax1.plot(i[1] + abs(t), -abs(0.01*min(ypoints)), marker="^", color = "r")
        ax1.annotate("b",[i[1]+abs(t),0],[i[1]+abs(t)+(0.015*all_loads[-1][1]),-abs(0.05*min(ypoints))],color = "r")
    if i[2] == "S":
        ax1.annotate(f"{i[0]} kN",[i[1]+abs(t),0],(i[1]+abs(t),0.22 * abs(Mmax)),
        arrowprops = dict(facecolor ='#1f77b4',edgecolor='#1f77b4' ,headlength = 5, width = 1, ))
    if i[2] == "R":
        ax1.annotate(f"$Rz = {-i[0]:.2f} kN$",[i[1]+abs(t),0],(i[1]+abs(t),-0.22 * abs(Mmax)),
        arrowprops = dict(facecolor ='r',edgecolor='r' ,headlength = 5, width = 1, ))
        # stanovit novou hodnotu pro list, kde bude "a" neb "b" pro f"$R{hodnota}z = {-i[0]:.2f} kN$"
    if i[2] == "D":
        ax1.annotate(f"Q = {i[0]} kN", [i[1] + abs(t), 0], (i[1] + abs(t), 0.22 * abs(Mmax)),
        arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', headlength=5, width=1, ))
        ax1.annotate(f"q = {i[4]} "+r"$\frac{kN}{m}$", [i[6] + abs(t), 0], (i[6] + abs(t), 0.22 * abs(Mmax)/2),)
        ax1.plot([i[5]+abs(t), i[6]+abs(t)], [0.22 * abs(Mmax)/2, 0.22 * abs(Mmax)/2], color="#4d004d", alpha=0.8)  # purple line graphing the D load
        for j in np.linspace(i[5],i[6],4,endpoint=True):
            ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 0.22 * abs(Mmax)/2),
            arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
    if i[2] == "T_L":
        ax1.annotate(f"Q = {i[0]} kN", [i[1] + abs(t), 0], (i[1] + abs(t), 0.22 * abs(Mmax)),
                     arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', headlength=5, width=1, ))
        ax1.annotate(f"q = {i[4]} " + r"$\frac{kN}{m}$", [i[6] + abs(t), 0], (i[6] + abs(t), 0.22 * abs(Mmax) / 2), )

        for j in np.linspace(i[5], i[6], 4, endpoint=True):
            ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 0.22 * abs(Mmax) / 2),
                         arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
    if i[2] == "T_P":
        ax1.annotate(f"Q = {i[0]} kN", [i[1] + abs(t), 0], (i[1] + abs(t), 0.22 * abs(Mmax)),
                     arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', headlength=5, width=1, ))
        ax1.annotate(f"q = {i[4]} " + r"$\frac{kN}{m}$", [i[6] + abs(t), 0], (i[6] + abs(t), 0.22 * abs(Mmax) / 2), )
        ax1.plot([i[5] + abs(t), i[6] + abs(t)], [1, 0.22 * abs(Mmax) / 2+i[7]+ (i[6]- i[5])], color="#4d004d",
                 alpha=0.8)  # purple line graphing the T load
        for j in np.linspace(i[5], i[6], 4, endpoint=True):
            if j == i[5]:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t),1),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            elif j==i[6]:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 0.22 * abs(Mmax) / 2 + j),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            else:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 0.10 * abs(Mmax) / 2 +j/i[6]*j),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
    if i[2] == "T_L":
        ax1.annotate(f"Q = {i[0]} kN", [i[1] + abs(t), 0], (i[1] + abs(t), 0.22 * abs(Mmax)),
                     arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', headlength=5, width=1, ))
        ax1.annotate(f"q = {i[4]} " + r"$\frac{kN}{m}$", [i[6] + abs(t), 0], (i[6] + abs(t), 0.22 * abs(Mmax) / 2), )
        ax1.plot([i[5] + abs(t), i[6] + abs(t)], [0.22 * abs(Mmax) / 2+i[7]+ i[6]- i[5],1], color="#4d004d",
                 alpha=0.8)  # purple line graphing the T load
        for j in np.linspace(i[5], i[6], 4, endpoint=True):
            if j == i[6]:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t),1),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            elif j==i[5]:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 0.22 * abs(Mmax) / 2+i[7]+ i[6]- i[5]),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            else:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 0.10 * abs(Mmax) / 2 + 40/j**2),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
    if i[2] == "P_P":
        n = i[8]
        ax1.annotate(f"Q = {i[0]} kN", [i[1] + abs(t), 0], (i[1] + abs(t), 0.22 * abs(Mmax)),
                     arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', headlength=5, width=1, ))
        ax1.annotate(f"q = {i[4]} " + r"$\frac{kN}{m}$", [i[6] + abs(t), 0], (i[6] + abs(t), 0.22 * abs(Mmax) / 2), )
        test_x = np.linspace(i[5],i[6],30)
        test_y = ((i[4]/i[7]) * test_x**n)+1

        #ax1.plot([i[5] + abs(t), i[6] + abs(t)], [1, 0.22 * abs(Mmax) / 2+i[7]+ (i[6]- i[5])], color="#4d004d",
        #         alpha=0.8)  # purple line graphing the P load
        ax1.plot(test_x, test_y, color="#4d004d",
                 alpha=0.8)  # purple line graphing the P load
        for j in np.linspace(i[5], i[6], 6, endpoint=True):
            if j == i[5]:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t),1),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            elif j==i[6]:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 1+(i[4]/i[7])*j**n),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            else:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 1+(i[4]/i[7])*j**n),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
    if i[2] == "P_L":
        n = i[8]
        ax1.annotate(f"Q = {i[0]} kN", [i[1] + abs(t), 0], (i[1] + abs(t), 0.22 * abs(Mmax)),
                     arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', headlength=5, width=1, ))
        ax1.annotate(f"q = {i[4]} " + r"$\frac{kN}{m}$", [i[6] + abs(t), 0], (i[6] + abs(t), 0.22 * abs(Mmax) / 2), )
        test_x = np.linspace(i[5],i[6],30)
        test_y = ((i[4]/i[7]) * (test_x[::-1])**n)+1
        ax1.plot(test_x, test_y, color="#4d004d",
                 alpha=0.8)  # purple line graphing the P load
        for j,k in zip(np.linspace(i[5], i[6], 6, endpoint=True)[::-1],np.linspace(i[5], i[6], 6, endpoint=True)):
            print(j)
            if j == i[5]:
                ax1.annotate(f"", [j + abs(t), 0],(j + abs(t), 1+(i[4]/i[7])*k**n),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            elif j==i[6]:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t),1),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
            else:
                ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 1+(i[4]/i[7])*k**n),
                             arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))
    if i[2] == "M" and i[0] > 0:
        ax1.annotate(f"{i[0]} kNm",[i[1]+abs(t),0-0.5],(i[1]+abs(t),0.22 * abs(Mmax)),
        arrowprops = dict(arrowstyle="simple",connectionstyle="arc3,rad=1.5", facecolor ='#1f77b4',edgecolor='#1f77b4', ))
    if i[2] == "M" and i[0] < 0:
        ax1.annotate(f"{i[0]} kNm",[i[1]+abs(t),-0.5],(i[1]+abs(t)-0.5,0.22 * abs(Mmax)),
        arrowprops = dict(arrowstyle="simple",connectionstyle="arc3,rad=-2", facecolor ='#1f77b4',edgecolor='#1f77b4', ))
### setting up the
ax2 = fig.add_subplot(gs[2, 0])
for i in all_loads:
    if i[3] == "a":
        ax2.plot(i[1]+abs(t),-abs(0.01*min(ypoints)),marker = "^", color = "r")
        ax2.annotate("a",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(ypoints))],color = "r")
    if i[3] == "b":
        ax2.plot(i[1] + abs(t), -abs(0.01*min(ypoints)), marker="^", color = "r")
        ax2.annotate("b",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(ypoints))],color = "r")

ax2.set_title(r"$Bending\ moments \ M_y \ [kNm] $")
ax2.plot(xpoints, ypoints)



ax2.plot([0+abs(t),0+abs(t)],[min(ypoints-10),max(ypoints)+10],linestyle = "dotted",color = "#bfbfbf")
ax2.plot([min(xpoints)+odchylka,max(xpoints)+odchylka],[0,0],color = "black",alpha=0.8)   #black line on x-axis
ax2.plot(x[y.index(min(y))],min(ypoints),marker = "o")
ax2.plot(x[y.index(max(y))],max(ypoints),marker = "o")
#plt.xticks(np.arange(odchylka, positions[-1]+int(inverse_odchylka), step=1),labels=[i-abs(t) for i in range(round(positions[0])+int(inverse_odchylka),ceil(positions[-1]+int(inverse_odchylka)))])
if Mmax == min(ypoints):
    ax2.annotate(f"$M_{{{'y,max'}}} = {-Mmax:.2f}$",[x[y.index(min(y))],min(ypoints)],
             (x[y.index(min(y))]+0.5,min(ypoints)-5))
    ax2.annotate(f"$M_{{{'y,min'}}} = {-max(ypoints):.2f}$", [x[y.index(max(y))], max(ypoints)],(x[y.index(max(y))]+0.5,max(ypoints)-5))
else:
    ax2.annotate(f"$M_{{{'y,max'}}} = {-Mmax:.2f}$", [x[y.index(max(y))], max(ypoints)], (x[y.index(max(y))]+0.5,max(ypoints)-5))
    ax2.annotate(f"$M_{{{'y,min'}}} = {-min(ypoints):.2f}$", [x[y.index(min(y))], min(ypoints)],
                 (x[y.index(min(y))] + 0.5, min(ypoints) - 5))
#Bending_maxima = []
#for k in V_maxima:

for i,j in zip(range(len(V_maxima)),V_maxima):
    ax2.plot(x[V_maxima[i][1]], ypoints[j[1]], marker="o")
    ax2.annotate(f"$M_{{{f'({round(V_maxima[i][1]/1000)})'}}} = {-ypoints[j[1]]:.2f}$", [x[V_maxima[i][1]], -ypoints[j[1]]],
                 (x[V_maxima[i][1]] + 0.5, ypoints[j[1]] ))
ax3= fig.add_subplot(gs[1, 0])
for i in all_loads:
    if i[3] == "a":
        ax3.plot(i[1]+abs(t),-abs(0.01*min(yvpoints)),marker = "^", color = "r")
        ax3.annotate("a",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(yvpoints))],color = "r")
    if i[3] == "b":
        ax3.plot(i[1] + abs(t), -abs(0.01*min(yvpoints)), marker="^", color = "r")
        ax3.annotate("b",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(yvpoints))],color = "r")
ax3.set_title(r"$Shear\ forces\ V_x \ [kN]$")
ax3.plot(xvpoints,-yvpoints)
ax3.plot([0+abs(t),0+abs(t)],[min(ypoints-10),max(ypoints)+10],linestyle = "dotted",color = "#bfbfbf")
ax3.plot([min(xpoints)+odchylka,max(xpoints)+odchylka],[0,0],color = "black",alpha=0.8) # black line on x axis
#plt.xticks(np.arange(odchylka, positions[-1]+int(inverse_odchylka), step=1),labels=[i-abs(t) for i in range(round(positions[0])+int(inverse_odchylka),ceil(positions[-1]+int(inverse_odchylka)))])
for i in range(len(V_maxima)):
    ax3.plot(x[V_maxima[i][1]], -V_maxima[i][0], marker="o")
    if V_maxima[i] == min(V_maxima):
        ax3.annotate(f"$V_{{{'max'}}} = {-V_maxima[i][0]:.2f}$", [x[V_maxima[i][1]], -V_maxima[i][0]],
                 (x[V_maxima[i][1]] + 0.5, -V_maxima[i][0] - 5))
        Vmax = f"$V_{{{'max'}}} = {-V_maxima[i][0]:.2f} \ kN$"
    else:
        ax3.annotate(f"$V_{{{f'({round(V_maxima[i][1]/1000)})'}}} = {-V_maxima[i][0]:.2f}$", [x[V_maxima[i][1]], -V_maxima[i][0]],
                     (x[V_maxima[i][1]] + 0.5, -V_maxima[i][0] ))
#plt.show()
subcentroid_2 = 0
subcentroid=0
Area=0

count = 0
count_2 = 1 #test
Zero_divide_check = 0
centroids_in_M = []
Different_Areas_check = 0
if t < reaction_a and max(positions) > reaction_b:
    for i in a:
        if count == (reaction_a + inverse_odchylka) * inverse_accuracy and Zero_divide_check != 1:
            #if Area != 0:
            centroid_L = subcentroid / Area
            Area_L = Area
            subcentroid = 0
            Area = 0
            Zero_divide_check = 1
        if reaction_a * inverse_accuracy < count_2 <= reaction_b*inverse_accuracy+1:
            if (ypoints[count_2] < 0 and ypoints[count] > 0) or (ypoints[count_2] > 0 and ypoints[count] < 0) or (count == reaction_b * inverse_accuracy):
                # if theres a change of sign (+ to - or - to +) a new Area and centroid is created (principle of superposition)
                # when count is equal to the position of reaction b, the last Area is created (if the sign stays the same on the whole interval there will only be one Area and centroid)
                c = subcentroid / Area
                arm = Area
                centroids_in_M.extend([[c,arm]])
                Area =0
                subcentroid = 0
                Different_Areas_check = 1
        try:
            if count == 0:
                subcentroid += (xpoints[count] - accuracy / 2) * ((ypoints[count]) / 2)
                Area += ((ypoints[count]) / 2)
                count += 1
                count_2 += 1
            else:
                subcentroid += (xpoints[count] - accuracy/2) * ((ypoints[count-1] + ypoints[count])/2)
                Area += ((ypoints[count-1] + ypoints[count])/2)
                count += 1
                count_2 += 1
        except: IndexError

    print("centroids in M",centroids_in_M)
    centroid_P = subcentroid / Area
    Area_P = Area


elif max(positions)>reaction_b:
    for i in a:
        if count == (reaction_b+inverse_odchylka) * inverse_accuracy:
            centroid_L = subcentroid / Area
            Area_L = Area
            subcentroid = 0
            Area = 0
        subcentroid += (xpoints[count] - accuracy/2) * ypoints[count]
        Area += ypoints[count]
        count += 1
    centroid_P = subcentroid / Area
    Area_P = Area
    Area = Area_P + Area_L # net Area
    print(f"centroid P = {centroid_P} centroid L = {centroid_L}")
    print(f"Areas = L -{Area_L} P -{Area_P} ... NET {Area_P + Area_L}")
    centroid = (Area_P * centroid_P + Area_L * centroid_L) / Area
elif t < reaction_a:
    for i in a:
        if count == (reaction_a + inverse_odchylka) * inverse_accuracy and Zero_divide_check !=1:
            centroid_L = subcentroid / Area
            Area_L = Area
            subcentroid = 0
            Area = 0
            Zero_divide_check = 1
        subcentroid += (xpoints[count] - accuracy/2) * ypoints[count]
        Area += ypoints[count]
        count += 1
    centroid_P = subcentroid / Area
    Area_P = Area
    Area = Area_P + Area_L # net Area
    print(f"centroid P = {centroid_P} centroid L = {centroid_L}")
    print(f"Areas = L -{Area_L} P -{Area_P} ... NET {Area_P + Area_L}")
    centroid = (Area_P * centroid_P + Area_L * centroid_L) / Area

else:
    for i in a:
        subcentroid += (xpoints[count]-accuracy/2)*ypoints[count]
        Area+=ypoints[count]
        count += 1
    centroid = subcentroid/Area

if t < reaction_a and max(positions) > reaction_b:
    der3 = []
    F_M = 0
    Area_M = 0

    for i in centroids_in_M:
        F_M += ((i[0] - reaction_a)*i[1])
        Area_M += i[1]
        print("ARea_M =", Area_M)
    Rbz = (F_M/(reaction_b-reaction_a))
    Raz = Area_M - Rbz
    Mc = -Area_L*(centroid_L - t) - Raz * (reaction_a-t)
    Rcz = ((Mc-Area_L*(reaction_a-centroid_L))/reaction_a) #pytest odečíst síly a zjisit jestli se to rovná
    print("Raz=",Raz,f"Rbz= {Rbz}",f"Rcz= {Rcz}","Mc=",Mc)

    count = 0
    hodnota_pocatku = Rcz

    for i in a:
        hodnota_pocatku += ypoints[count]
        der3.append(hodnota_pocatku)
        count += 1
elif t < reaction_a:
    der3 = []
    if reaction_a < centroid_P < reaction_b:
        cantilever_R = -Area - ((-Area_P * (centroid_P-reaction_a)) / (reaction_b-reaction_a))
        cantilever_M = (-Area * centroid) -(((-Area_P * (centroid_P-reaction_a)) / (reaction_b-reaction_a))*reaction_b)
    else:
        cantilever_R = -Area
        cantilever_M = -Area * centroid
    print(f"cantilever_R = {cantilever_R}, reaction_b = {((-Area_P * (centroid_P-reaction_a)) / (reaction_b-reaction_a))}")
    count = 0
    hodnota_pocatku = cantilever_R
    for i in a:
        hodnota_pocatku += ypoints[count]
        der3.append(hodnota_pocatku)
        count += 1
elif max(positions) > reaction_b:
    der3 = []
    if reaction_a < centroid_L < reaction_b:
        cantilever_R = -Area - ((-Area_L * (reaction_b - centroid_L)) / (reaction_b - reaction_a))
        cantilever_M = (-Area * centroid) - (
                (Area_L * (reaction_b - centroid_L)) / (reaction_b - reaction_a) * reaction_b)
        print("here")
    else:
        cantilever_R = -Area
        cantilever_M = -Area * centroid
        print("proč")
    print(f"cantilever_R = {cantilever_R}, reaction_b = {((-Area_P * (centroid_P - reaction_a)) / (reaction_b - reaction_a))}")
    count = 0
    hodnota_pocatku = cantilever_R
    ypoints=ypoints[::-1]
    for i in a:
        hodnota_pocatku += ypoints[count]
        der3.append(hodnota_pocatku)
        count += 1
    der3=der3[::-1]
else:
    print("centroid =", centroid, "and", "area = ", Area)
    der3 = []
    pokus = int((-((reaction_b) - centroid)*Area/(reaction_b-reaction_a)))
    count = 0
    hodnota_pocatku = 0
    for i in a:
        hodnota_pocatku += ypoints[count]
        der3.append(hodnota_pocatku)
        if count * accuracy == reaction_a:
            hodnota_pocatku += pokus
            pokus =0
        count += 1

#print("der3 =", der3)
der3[-1]=0 # Making Shear forces go back to zero
der3 = [i*accuracy for i in der3]
der3points = np.array(der3)
#for i,j in zip(xpoints,der3points):
#   print(i,j )

ax4 = fig.add_subplot(gs[0, 1])
ax4.set_title(r"$Slopes\ \varphi \ [rad] $")
ax4.plot(xpoints,der3points)
ax4.plot([0+abs(t),0+abs(t)],[min(der3points-10),max(der3points)+10],linestyle = "dotted",color = "#bfbfbf")
ax4.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)
ax4.plot(x[der3.index(max(der3points))],max(der3points),marker = "o")
ax4.plot(x[int(reaction_a*inverse_accuracy)-1],der3points[int(reaction_a*inverse_accuracy)-1],marker = "o")
ax4.plot(x[int(reaction_b*inverse_accuracy)-1],der3points[int(reaction_b*inverse_accuracy)-1],marker = "o")
ax4.plot(x[0],der3points[0],marker = "o")
ax4.plot(x[-2],der3points[-2],marker = "o")
ax4.annotate(f"$\\varphi_{{{'max'}}} = {max(der3points):.2f}$",[x[der3.index(max(der3points))],max(der3points)],
         (x[der3.index(max(der3points))]+0.5,max(der3points)+5))
ax4.annotate(f"$\\varphi_{{{'a'}}} = {der3points[int(reaction_a*inverse_accuracy)]:.2f}$",[x[int(reaction_a*inverse_accuracy)],der3points[int(reaction_a*inverse_accuracy)-1]],
         (x[int(reaction_a*inverse_accuracy)]+0.5,der3points[int(reaction_a*inverse_accuracy)]-5))
ax4.annotate(f"$\\varphi_{{{'b'}}} = {der3points[int(reaction_b*inverse_accuracy)-1]:.2f}$",[x[int(reaction_b*inverse_accuracy)-1],der3points[int(reaction_b*inverse_accuracy)-1]],
         (x[int(reaction_b*inverse_accuracy-1)]+0.5,der3points[int(reaction_b*inverse_accuracy-1)]-5))
ax4.annotate(f"$\\varphi_{{{'(0)'}}} = {der3points[0]:.2f}$",[x[0],der3points[0]],
         (x[0]+0.5,der3points[0]-5))
ax4.annotate(f"$\\varphi_{{{f'({max(positions)})'}}} = {der3points[-2]:.2f}$",[x[-2],der3points[-2]],
         (x[-2]-2,der3points[-2]-0.5*max(der3points)))
#ax4.annotate(f"$\phi_{{{'min'}}} = {-max(ypoints):.2f}$", [x[y.index(max(y))], max(ypoints)],(x[y.index(max(y))]+0.5,max(ypoints)-5))

#plt.xticks(np.arange(odchylka, positions[-1]+int(inverse_odchylka), step=1),labels=[i-abs(t) for i in range(round(positions[0])+int(inverse_odchylka),ceil(positions[-1]+int(inverse_odchylka)))])
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_title(r"$Deflections \ w \ [m]$")
der4 = []
hokus=0
#hokus += -cantilever_M
count = 0
if t < reaction_a and max_pos <= reaction_b:
    print(f"R = {cantilever_R} M = {cantilever_M}")
for i in a:
    if max(positions) > reaction_b:
        hokus -= der3points[count]*accuracy
    else:
        hokus += der3points[count]*accuracy
    der4.append(hokus)
    if t < reaction_a and max(positions) > reaction_b:
        hokus += -Mc*accuracy
        Mc = 0
    elif t < reaction_a:
        hokus += -cantilever_M *accuracy
        cantilever_M = 0

    count += 1

der4points = np.array(der4)
#with open("hodnoty.csv","w") as file:
#    for i,j in zip(xpoints,der4points):
#        file.write(f"{i,j}\n")
ax5.plot(xpoints,-der4points)
ax5.plot([0+abs(t),0+abs(t)],[min(der4points-10),max(der4points)+10],linestyle = "dotted",color = "#bfbfbf")
ax5.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)
#plt.xticks(np.arange(t, positions[-1]+abs(t), step=1),labels=[i-abs(t) for i in range(round(positions[0]),ceil(positions[-1]+abs(t)))])
ax5.plot(x[der4.index(min(der4))],-min(der4points),marker = "o")
ax5.plot(x[der4.index(max(der4))],-max(der4points),marker = "o")
if abs(min(der4)) < abs(max(der4)):
    wmax = max(der4)
else:
    wmax= min(der4)
ax5.annotate(f"$w_{{{'max,upward'}}} = {{{-min(der4):.2f}}} / EI$",[x[der4.index(min(der4))],min(der4points)],
         (x[der4.index(min(der4))],-min(der4points)))
ax5.annotate(rf"$w_{{{'max,downward'}}} = {{{-max(der4):.2f}}} / EI$", [x[der4.index(max(der4))],-max(der4points)])
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title(r"Results")
ax6.text(0.5,0.9,r"$w_{max,downward}$"+f" = {-wmax:.2f}/EI m") #if větší než 0.00 něco
if abs(min(der4)) > 0.5:
    ax6.text(0.5,0.8,r"$w_{max,upward}$"+f" = {-min(der4):.2f}/EI m")
if V_maxima != []:
    ax6.text(0.01, 0.9, Vmax)
ax6.text(0.01,0.8,f"$M_{{{'y,max'}}} = {-Mmax:.2f}\ kNm$")
ax6.text(0.25,0.9,f"$\\varphi_{{{'max'}}} = {max(der3points):.2f}/EI \ rad $")
ax6.text(0.25,0.8,f"$\\varphi_{{{'a'}}} = {der3points[int(reaction_a*inverse_accuracy)-1]:.2f}/EI \ rad $")
ax6.text(0.25,0.7,f"$\\varphi_{{{'b'}}} = {der3points[int(reaction_b*inverse_accuracy)-1]:.2f}/EI \ rad $")
ax6.text(0.25,0.6,f"$\\varphi_{{{'(0)'}}} = {der3points[0]:.2f}/EI \  rad $")
ax6.text(0.25,0.5,f"$\\varphi_{{{f'({max(positions)})'}}} = {der3points[-2]:.2f}/EI \ rad $")
plt.xticks([])
plt.yticks([])
#print("index",der3.index(max(der3points)))
with open("paraboly.csv","w") as file:
    for i,j in zip(der4points,xpoints):
        file.write(f"{i,j}\n")

plt.show()


    #add option: add EI or keep it constant

#konzola
# opravit převislé konce na listy
# opravit i else: na listy
# přidat možnost cancellnutí/opakování při zadávání
# rename T_P to T_R