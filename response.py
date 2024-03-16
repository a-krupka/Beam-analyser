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
#Moment jen záporný OPRAVIT
#all_loads =[(24.0, -2.0, 'S', '', '', '', ''), (24.0, 0.0, 'S', '', '', '', ''), (-96.0000000000000, 2.0, 'R', 'a', '', '', ''), (24.0, 3.0, 'S', '', '', '', ''), (92.0, 7.0, 'D', '', 23.0, 5.0, 9.0), (-68.0000000000000, 7.0, 'R', 'b', '', '', ''),]#(0.1, 8.0, 'S', '', '', '', '')]
#NEFUNGUJE
#all_loads = [(8.0, 2.0, 'S', '', '', '', ''), (-30.7250000000000, 3.0, 'R', 'a', '', '', ''), (10.0, 11.0, 'S', '', '', '', ''), (37.5, 6.25, 'D', '', 15.0, 5.0, 7.5), (-24.7750000000000, 8.0, 'R', 'b', '', '', '')]
#Naděje - funguje
#all_loads=[(80.0, 2.0, 'D', '', 20.0, 0.0, 4.0), (80.0, 10.0, 'D', '', 20.0, 8.0, 12.0), (-80.0000000000000, 8.0, 'R', 'b', '', '', ''), (-80.0000000000000, 4.0, 'R', 'a', '', '', '')]
# Mohrovka, příklad C
#all_loads = [(-3.60000000000000, 0.0, 'R', 'a', '', '', ''), (2.0, 2.0, 'S', '', '', '', ''), (6.0, 3.0, 'S', '', '', '', ''), (-4.40000000000000, 5.0, 'R', 'b', '', '', '')]
#Mohrovka zprava
#all_loads =[(-1.50000000000000, 0.0, 'R', 'a', '', '', ''), (12.0, 3.5, 'D', '', 4.0, 2.0, 5.0), (-17.5000000000000, 5.0, 'R', 'b', '', '', ''), (7.0, 6.5, 'S', '', '', '', '')]
#Mohrovka příklad F
#all_loads = [(-35.9500000000000, 0.0, 'R', 'a', '', '', ''), (65.0, 4.5, 'D', '', 13.0, 2.0, 7.0), (2.0, 9.0, 'S', '', '', '', ''), (-31.0500000000000, 10.0, 'R', 'b', '', '', '')]
#Mohrovka zprava i zleva
#all_loads = [(10.0, -1.0, 'S', '', '', '', ''), (20.0, 0.0, 'S', '', '', '', ''), (-87.3333333333333, 0.0, 'R', 'a', '', '', ''), (25.0, 3.0, 'S', '', '', '', ''), (132.0, 6.0, 'D', '', 33.0, 4.0, 8.0), (-179.666666666667, 9.0, 'R', 'b', '', '', ''), (80.0, 9.5, 'S', '', '', '', '')]
#Mohrovka, příklad ?
#all_loads=[(-5.84615384615385, 0.0, 'R', 'a', '', '', ''), (2.0, 1.0, 'S', '', '', '', ''), (9.0, 3.5, 'D', '', 3.0, 2.0, 5.0), (-5.15384615384615, 6.5, 'R', 'b', '', '', '')]
#Mohrovka převislý konec zleva
all_loads = [(16.0, 2.0, 'D', '', 4.0, 0.0, 4.0), (-21.3333333333333, 4.0, 'R', 'a', '', '', ''), (5.33333333333333, 10.0, 'R', 'b', '', '', '')]
#Mohrovka převislý konec zprava
#all_loads = [(60.0, 7.0, 'D', '', 10.0, 4.0, 10.0), (-105.000000000000, 4.0, 'R', 'b', '', '', ''), (45.0000000000000, 0.0, 'R', 'a', '', '', '')]
#Mohrovka jednoduchý příklad
#all_loads = [(-12.0000000000000, 0.0, 'R', 'a', '', '', ''), (24.0, 5.0, 'D', '', 4.0, 2.0, 8.0), (-12.0000000000000, 10.0, 'R', 'b', '', '', '')]

positions = [i[1] if i[2] != 'D' else i[5] for i in all_loads] # gets the position of a load, in case of distributed load, outputs starting position
reaction_a = float("".join([str(i[1]) for i in all_loads if i[3] == 'a'])) #computing position of reaction a by list comprehension, then converting into single float
reaction_b = float("".join([str(i[1]) for i in all_loads if i[3] == 'b'])) #computing position of reaction a by list comprehension, then converting into single float
print(f"a = {reaction_a} b = {reaction_b}")
starting_pos = min(positions) # gets the starting (most left) position of force
t = starting_pos


print(positions,f"min pos is {min(positions)}")
all_loads= (sorted(all_loads,key=itemgetter(1)))
print(all_loads)
for i in all_loads:
    if i[2] == "D":
        if i[6] > max(positions):
            positions.append(i[6])
print(positions)
if t <= 0: #tady tweakuju
    max_pos = max(positions) + abs(t) + 0.001
    a= np.arange(0.001,max_pos,0.001)
    odchylka = 0
    inverse_odchylka = abs(t)
    posun = abs(t)
else:
    max_pos = max(positions) + 0.001
    a = np.arange(0.001, max_pos, 0.001)
    odchylka = abs(t)
    inverse_odchylka = 0
    posun = 0
Vy = [0.0 for i in a] #Posouvačky # making a list of len(a) of every value 0.0 to which we'll be adding results of for loop
Vx = [i for i in a]
Vnun=[0.0 for i in a] # auxillary temporary list which stores results of most recent iteration

### zkusit změnit pořadí ve for loopu

for j in range(len(all_loads)):
    Vy = list(map(add, Vy, Vnun)) #adds the temporary list of values Vnun to Vy ,add is a function imported from operators
    #print(Vy)
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

    ### potentional place for additional types of forces – "Triangular" "Parabolic" etc

### zkusit změnit pořadí ve for loopu ^^^^^^

Vy = list(map(add, Vy, Vnun)) # adding the last remaining values
Vy[0] = 0 # rewriting the first value on the y axis as 0 so it starts from zero

xvpoints = np.array(Vx) # making a numpy array from points on the x-axis of shear force diagram
yvpoints = np.array(Vy) # making a numpy array from points on the y-axis of shear force diagram
print(f"length of x {len(xvpoints)}, length of y {len(yvpoints)}")



### Moment diagram
y = [0.0 for i in a]
x = [i for i in a]

nun=[0.0 for i in a]
for j in range(len(all_loads)):
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
y = list(map(add, y, nun))

#for i,j in zip(x,y):
#    print(i,j)
xy = [(x[i],y[i]) for i in range(len(y))]

xpoints = np.array(x)
ypoints = np.array(y)
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
plt.xticks(np.arange(t, positions[-1]+abs(t), step=1),labels=[i-abs(t) for i in range(round(positions[0]),ceil(positions[-1]+abs(t)))])
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
        ax1.annotate(f"q = {i[4]} kN/m", [i[6] + abs(t), 0], (i[6] + abs(t), 0.22 * abs(Mmax)/2),)
        ax1.plot([i[5]+abs(t), i[6]+abs(t)], [0.22 * abs(Mmax)/2, 0.22 * abs(Mmax)/2], color="#4d004d", alpha=0.8)  # purple line graphing the D load
        for j in np.linspace(i[5],i[6],4,endpoint=True):
            ax1.annotate(f"", [j + abs(t), 0], (j + abs(t), 0.22 * abs(Mmax)/2),
            arrowprops=dict(facecolor='#4d004d', edgecolor='#4d004d', headlength=5, width=1, ))

### setting up the
ax2 = fig.add_subplot(gs[2, 0])
for i in all_loads:
    if i[3] == "a":
        ax2.plot(i[1]+abs(t),-abs(0.01*min(ypoints)),marker = "^", color = "r")
        ax2.annotate("a",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(ypoints))],color = "r")
    if i[3] == "b":
        ax2.plot(i[1] + abs(t), -abs(0.01*min(ypoints)), marker="^", color = "r")
        ax2.annotate("b",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(ypoints))],color = "r")

ax2.set_title(r"$Bending\ moments \ M_y$")
ax2.plot(xpoints, ypoints)

ax2.plot([0+abs(t),0+abs(t)],[min(ypoints-10),max(ypoints)+10],linestyle = "dotted",color = "#bfbfbf")
ax2.plot([min(xpoints)+odchylka,max(xpoints)+odchylka],[0,0],color = "black",alpha=0.8)   #black line on x-axis
ax2.plot(x[y.index(min(y))],min(ypoints),marker = "o")
ax2.plot(x[y.index(max(y))],max(ypoints),marker = "o")
plt.xticks(np.arange(odchylka, positions[-1]+int(inverse_odchylka), step=1),labels=[i-abs(t) for i in range(round(positions[0])+int(inverse_odchylka),ceil(positions[-1]+int(inverse_odchylka)))])
if Mmax == min(ypoints):
    ax2.annotate(f"$M_{{{'y,max'}}} = {-Mmax:.2f}$",[x[y.index(min(y))],min(ypoints)],
             (x[y.index(min(y))]+0.5,min(ypoints)-5))
else:
    ax2.annotate(f"$M_{{{'y,max'}}} = {-Mmax:.2f}$", [x[y.index(max(y))], max(ypoints)])
ax3= fig.add_subplot(gs[1, 0])
for i in all_loads:
    if i[3] == "a":
        ax3.plot(i[1]+abs(t),-abs(0.01*min(yvpoints)),marker = "^", color = "r")
        ax3.annotate("a",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(yvpoints))],color = "r")
    if i[3] == "b":
        ax3.plot(i[1] + abs(t), -abs(0.01*min(yvpoints)), marker="^", color = "r")
        ax3.annotate("b",[i[1]+abs(t),0],[i[1]+abs(t)+(0.01*all_loads[-1][1]),-abs(0.1*min(yvpoints))],color = "r")
ax3.set_title(r"$Shear\ forces\ V_x$")
ax3.plot(xvpoints,-yvpoints)
ax3.plot([0+abs(t),0+abs(t)],[min(ypoints-10),max(ypoints)+10],linestyle = "dotted",color = "#bfbfbf")
ax3.plot([min(xpoints)+odchylka,max(xpoints)+odchylka],[0,0],color = "black",alpha=0.8) # black line on x axis
plt.xticks(np.arange(odchylka, positions[-1]+int(inverse_odchylka), step=1),labels=[i-abs(t) for i in range(round(positions[0])+int(inverse_odchylka),ceil(positions[-1]+int(inverse_odchylka)))])
#ax5 = fig.add_subplot(gs[0:2, 1])
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
        if count == (reaction_a + inverse_odchylka) * 1000 and Zero_divide_check != 1:
            centroid_L = subcentroid / Area
            Area_L = Area
            subcentroid = 0
            Area = 0
            Zero_divide_check = 1
        if reaction_a *1000 < count_2 <= reaction_b*1000+1:
            if (ypoints[count_2] < 0 and ypoints[count] > 0) or (ypoints[count_2] > 0 and ypoints[count] < 0) or (count == reaction_b * 1000):
                # if theres a change of sign (+ to - or - to +) a new Area and centroid is created (principle of superposition)
                # when count is equal to the position of reaction b, the last Area is created (if the sign stays the same on the whole interval there will only be one Area and centroid)
                c = subcentroid / Area
                arm = Area
                centroids_in_M.extend([[c,arm]])
                Area =0
                subcentroid = 0
                Different_Areas_check = 1

        subcentroid += (xpoints[count] - 0.0005) * ypoints[count]
        Area += ypoints[count]
        count += 1
        count_2 += 1

    print("centroids in M",centroids_in_M)
    centroid_P = subcentroid / Area
    Area_P = Area


elif max(positions)>reaction_b:
    for i in a:
        if count == reaction_b * 1000:
            centroid_L = subcentroid / Area
            Area_L = Area
            subcentroid = 0
            Area = 0
        subcentroid += (xpoints[count] - 0.0005) * ypoints[count]
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
        if count == (reaction_a + inverse_odchylka) * 1000 and Zero_divide_check !=1:
            centroid_L = subcentroid / Area
            Area_L = Area
            subcentroid = 0
            Area = 0
            Zero_divide_check = 1
        subcentroid += (xpoints[count] - 0.0005) * ypoints[count]
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
        subcentroid += (xpoints[count]-0.0005)*ypoints[count]
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
        cantilever_R = -Area - ((-Area_L * (centroid_L - reaction_a)) / (reaction_b - reaction_a))
        cantilever_M = (-Area * centroid) - (((-Area_L * (centroid_L - reaction_a)) / (reaction_b - reaction_a)) * reaction_b)
        print("here")
    else:
        cantilever_R = -Area
        cantilever_M = -Area * centroid
    print(f"cantilever_R = {cantilever_R}, reaction_b = {((-Area_P * (centroid_P - reaction_a)) / (reaction_b - reaction_a))}")
    count = 0
    hodnota_pocatku = cantilever_R
    for i in a:
        hodnota_pocatku += ypoints[count]
        der3.append(hodnota_pocatku)
        count += 1
else:
    print("centroid =", centroid, "and", "area = ", Area)
    der3 = []
    pokus = int((-((reaction_b) - centroid)*Area/(reaction_b-reaction_a)))
    count = 0
    hodnota_pocatku = 0
    for i in a:
        hodnota_pocatku += ypoints[count]
        der3.append(hodnota_pocatku)
        if count * 0.001 == reaction_a:
            hodnota_pocatku += pokus
            pokus =0
        count += 1

#print("der3 =", der3)
der3[-1]=0 # Making Shear forces go back to zero
der3 = [i*0.001 for i in der3]
der3points = np.array(der3)
#for i,j in zip(xpoints,der3points):
#   print(i,j )

ax4 = fig.add_subplot(gs[0, 1])
ax4.set_title(r"$Slopes\ \phi \ [\frac{kNm}{EI}] $")
ax4.plot(xpoints,der3points)
ax4.plot([0+abs(t),0+abs(t)],[min(der3points-10),max(der3points)+10],linestyle = "dotted",color = "#bfbfbf")
ax4.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)
plt.xticks(np.arange(odchylka, positions[-1]+int(inverse_odchylka), step=1),labels=[i-abs(t) for i in range(round(positions[0])+int(inverse_odchylka),ceil(positions[-1]+int(inverse_odchylka)))])
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_title(r"$Deflections \ w$")
der4 = []
hokus=0
#hokus += -cantilever_M
count = 0
for i in a:
    hokus += der3points[count]*0.001
    der4.append(hokus)
    if t < reaction_a and max(positions) > reaction_b:
        hokus += -Mc*0.001
        Mc = 0
    elif t < reaction_a:
        hokus += -cantilever_M *0.001
        cantilever_M = 0
    count += 1
der4points = np.array(der4)

ax5.plot(xpoints,-der4points)
ax5.plot([0+abs(t),0+abs(t)],[min(der4points-10),max(der4points)+10],linestyle = "dotted",color = "#bfbfbf")
ax5.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)
plt.xticks(np.arange(t, positions[-1]+abs(t), step=1),labels=[i-abs(t) for i in range(round(positions[0]),ceil(positions[-1]+abs(t)))])
ax5.plot(x[der4.index(min(der4))],-min(der4points),marker = "o")
ax5.plot(x[der4.index(max(der4))],-max(der4points),marker = "o")
if abs(min(der4)) < abs(max(der4)):
    wmax = max(der4)
else:
    wmax= min(der4)
if wmax == min(der4):
    ax5.annotate(f"$w_{{{'max'}}} = {{{-wmax:.2f}}}$",[x[der4.index(min(der4))],min(der4points)],
             (x[der4.index(min(der4))],-min(der4points)))
else:
    ax5.annotate(rf"$w_{{{'max'}}} = {{{-wmax:.2f}}} / EI$", [x[der4.index(max(der4))],-max(der4points)])
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title(r"Results")
ax6.text(0.1,0.9,r"$w_{max,downward}$"+f" = {-wmax:.2f}") #if větší než 0.00 něco
if abs(min(der4)) > 1.5:
    ax6.text(0.1,0.8,r"$w_{max,upward}$"+f" = {-min(der4):.2f}")
plt.xticks([])
plt.yticks([])
plt.show()
    # zaměřit se na "D" loads, zautomatizovat výpočet, done
    #add option: add EI or keep it constant
    # možná opravit slopes, zkusit Mohorovu metodu
