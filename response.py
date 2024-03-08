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

#all_loads = [(12.0, 0.0, 'S', '', '', '', ''), (-83.8000000000000, 3.0, 'R', 'a', '', '', ''), (100.0, 4.5, 'D', '', 20.0, 2.0, 7.0), (-46.2000000000000, 8.0, 'R', 'b', '', '', ''), (18.0, 9.5, 'S', '', '', '', '')]
#all_loads =[(24.0, -2.0, 'S', '', '', '', ''), (24.0, 0.0, 'S', '', '', '', ''), (-96.0000000000000, 2.0, 'R', 'a', '', '', ''), (24.0, 3.0, 'S', '', '', '', ''), (92.0, 7.0, 'D', '', 23.0, 5.0, 9.0), (-68.0000000000000, 7.0, 'R', 'b', '', '', ''),]#(0.1, 8.0, 'S', '', '', '', '')]
#all_loads = [(8.0, 2.0, 'S', '', '', '', ''), (-30.7250000000000, 3.0, 'R', 'a', '', '', ''), (10.0, 11.0, 'S', '', '', '', ''), (37.5, 6.25, 'D', '', 15.0, 5.0, 7.5), (-24.7750000000000, 8.0, 'R', 'b', '', '', '')]


#all_loads = [(-3.60000000000000, 0.0, 'R', 'a', '', '', ''), (2.0, 2.0, 'S', '', '', '', ''), (6.0, 3.0, 'S', '', '', '', ''), (-4.40000000000000, 5.0, 'R', 'b', '', '', '')]

#all_loads =[(-1.50000000000000, 0.0, 'R', 'a', '', '', ''), (12.0, 3.5, 'D', '', 4.0, 2.0, 5.0), (-17.5000000000000, 5.0, 'R', 'b', '', '', ''), (7.0, 6.5, 'S', '', '', '', '')]
#all_loads = [(-35.9500000000000, 0.0, 'R', 'a', '', '', ''), (65.0, 4.5, 'D', '', 13.0, 2.0, 7.0), (2.0, 9.0, 'S', '', '', '', ''), (-31.0500000000000, 10.0, 'R', 'b', '', '', '')]
all_loads = [(10.0, -1.0, 'S', '', '', '', ''), (20.0, 0.0, 'S', '', '', '', ''), (-87.3333333333333, 0.0, 'R', 'a', '', '', ''), (25.0, 3.0, 'S', '', '', '', ''), (132.0, 6.0, 'D', '', 33.0, 4.0, 8.0), (-179.666666666667, 9.0, 'R', 'b', '', '', ''), (80.0, 9.5, 'S', '', '', '', '')]
#all_loads=[(-5.84615384615385, 0.0, 'R', 'a', '', '', ''), (2.0, 1.0, 'S', '', '', '', ''), (9.0, 3.5, 'D', '', 3.0, 2.0, 5.0), (-5.15384615384615, 6.5, 'R', 'b', '', '', '')]
all_loads = [(16.0, 2.0, 'D', '', 4.0, 0.0, 4.0), (-21.3333333333333, 4.0, 'R', 'a', '', '', ''), (5.33333333333333, 10.0, 'R', 'b', '', '', '')]
#all_loads = [(-12.0000000000000, 0.0, 'R', 'a', '', '', ''), (24.0, 5.0, 'D', '', 4.0, 2.0, 8.0), (-12.0000000000000, 10.0, 'R', 'b', '', '', '')]
positions = [i[1] for i in all_loads] # gets the position of a load
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
if t >= 0:
    a= np.arange(0.001,max(positions) + abs(t) + 0.001,0.001)
else:                                                               #tohle smazat asi?
    a = np.arange(0.001, max(positions) + abs(t) + 0.001, 0.001)
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

for i,j in zip(x,y):
    print(i,j)
xy = [(x[i],y[i]) for i in range(len(y))]

xpoints = np.array(x)
ypoints = np.array(y)
mpoints = np.array(xy)
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
ax1.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)   #black line on x-axis
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

ax2.set_title(r"$Bending\ moments$")
ax2.plot(xpoints, ypoints)
ax2.plot([0+abs(t),0+abs(t)],[min(ypoints-10),max(ypoints)+10],linestyle = "dotted",color = "#bfbfbf")
ax2.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)   #black line on x-axis
ax2.plot(x[y.index(min(y))],min(ypoints),marker = "o")
ax2.plot(x[y.index(max(y))],max(ypoints),marker = "o")
plt.xticks(np.arange(t, positions[-1]+abs(t), step=1),labels=[i-abs(t) for i in range(round(positions[0]),ceil(positions[-1]+abs(t)))])
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
ax3.set_title(r"$Shear\ forces$")
ax3.plot(xvpoints,-yvpoints)
ax3.plot([0+abs(t),0+abs(t)],[min(ypoints-10),max(ypoints)+10],linestyle = "dotted",color = "#bfbfbf")
ax3.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)
plt.xticks(np.arange(t, positions[-1]+abs(t), step=1),labels=[i-abs(t) for i in range(round(positions[0]),ceil(positions[-1]+abs(t)))])
#ax5 = fig.add_subplot(gs[0:2, 1])
subcentroid=0
Area=0
for i in range(0,int((max(positions) + abs(t))*1000)):   #V tomhle může být chyba?
    subcentroid += (xpoints[i]-0.0005)*ypoints[i]*0.001
    Area+=ypoints[i]*0.001
centroid = subcentroid/Area
#for i in np.arange(0.001,max(positions) + abs(t) + 0.001,0.001):   #V tomhle může být chyba?
    #index = int(i * 1000)-1
    #subcentroid += (xpoints[index]-0.0005)*ypoints[index]*0.001*0.001
    #Area+=ypoints[index]*0.001
#centroid = subcentroid/Area

print("centroid =", centroid, "and", "area = ", Area)
der3 = []
pokus = int((-((max(positions) + abs(t)) - centroid)*Area/(max(positions) + abs(t)))*1000)
print("pokus =",pokus)
for i in range(0,(int((max(positions) + abs(t))*1000))):
    pokus +=ypoints[i]
    der3.append(pokus)
#print("der3 =", der3)
der3points = np.array(der3)

ax4 = fig.add_subplot(gs[0, 1])
ax4.set_title(r"$Slopes\ \phi$")
ax4.plot(xpoints,der3points)
ax4.plot([0+abs(t),0+abs(t)],[min(der3points-10),max(der3points)+10],linestyle = "dotted",color = "#bfbfbf")
ax4.plot([min(xpoints),max(xpoints)],[0,0],color = "black",alpha=0.8)
plt.xticks(np.arange(t, positions[-1]+abs(t), step=1),labels=[i-abs(t) for i in range(round(positions[0]),ceil(positions[-1]+abs(t)))])
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_title(r"$Deflections$")
der4 = []
hokus=0
for i in range(0,(int((max(positions) + abs(t))*1000))):
    hokus += der3points[i]*0.001
    der4.append(hokus)
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
ax6.text(0.1,0.9,r"$w_{max}$"+f" = {-wmax:.2f}")
plt.show()
    # zaměřit se na "D" loads, zautomatizovat výpočet, done
    #add option: add EI or keep it constant
    # možná opravit slopes, zkusit Mohorovu metodu
