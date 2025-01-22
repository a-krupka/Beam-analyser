import threading
from math import sqrt,atan,pi,sin,cos
from enum import Enum
from ask import *
import numpy as np
import tkinter as tk
from subprocess import call




def sign_of_coord(num) -> str:
    if num > 0: return '+'
    elif num < 0: return '-'
    else: return '0'

def get_pismena(x):
    return x


Type_of_connection = {
    "VV" : 0,
    "VK" : 1,
    "KV" : 2,
    "KK" : 3
}

class Counter:
    def __init__(self, value):
        self.value = value

    def plus_one(self):
        self.value += 1
        return self.value

    def add(self, number):
        self.value += number
        return self.value


class Node:
    def __init__(self,x_cord: float,y_cord: float,num: int, okrajove_podminky: list[bool] = [True,True,True],hinge: bool = False):
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.num = num
        self.hinge = hinge
        self.okrajove_podminky = okrajove_podminky

    def __repr__(self):
        return f"Node({self.x_cord}, {self.y_cord}, {self.num}, {self.hinge})"

class Line:
    def __init__(self, node1: Node, node2: Node,connection_type: dict[str,int] = Type_of_connection["VV"], young: int = 24e6, area: float = 0.06, inertia: float = 4.5e-4):
        self.node1 = node1
        self.node2 = node2
        self.protilehla = self.node2.y_cord-self.node1.y_cord
        self.prilehla = self.node2.x_cord-self.node1.x_cord
        self.length = self.compute_len()
        self.half = self.coord_half_len()
        self.alpha = self.compute_alpha()
        self.gamma = self.compute_gamma()
        self.connection_type = connection_type
        self.young = young # [kPa]
        self.area = area # m2
        self.inertia = inertia # m 4

    def compute_len(self) -> float:
        return sqrt(abs(self.prilehla) **2 + abs(self.protilehla) **2)

    def compute_alpha(self) -> float:
        if self.prilehla != 0:
            return atan(abs(self.protilehla)/abs(self.prilehla))
        else:
            return pi/2

    def coord_half_len(self) -> tuple:
        return ((self.node1.x_cord + self.node2.x_cord) / 2, (self.node1.y_cord + self.node2.y_cord) / 2)

    def compute_gamma(self) -> float:
        match (sign_of_coord(self.prilehla),sign_of_coord(self.protilehla)):
            case ('+','-'):
                return 2*pi - self.alpha
            case ('+', '+'):
                return self.alpha
            case ('-', '+'):
                return pi - self.alpha
            case ('-', '-'):
                return pi + self.alpha
            case ('0', '-'):
                return 3/2 * pi
            case ('0', '+'):
                return pi/2
            case ('+', '0'):
                return 0
            case ('-', '0'):
                return pi
    def __repr__(self):
        return f"Line({self.node1}, {self.node2}, length {self.length}, gamma {self.gamma * 360 / (2* pi)}, alpha {self.alpha * 360 / (2* pi)}\n\t connection_type {self.connection_type} E= {self.young}, A = {self.area}, I = {self.inertia})"

class Load:
    def __init__(self,load, pos_cg, load_type, is_node, load_per_metre, start, fin, length,n):
        self.load = load
        self.pos_cg = pos_cg
        self.load_type = load_type
        self.is_node = is_node
        self.load_per_metre = load_per_metre
        self.start = start
        self.fin = fin
        self.length = length
        self.n = n
    def __repr__(self):
        return f"Load({self.load}, pos {self.pos_cg}, type {self.load_type})"

def is_support(arr: list[float],line: Line):
    x = line.node1.okrajove_podminky
    y = line.node2.okrajove_podminky
    print(f"Node L = {x} Node P = {y}"  )
    arr[0] = arr[0] * x[0]
    arr[1] = arr[1] * x[1]
    arr[2] = arr[2] * x[2]
    arr[3] = arr[3] * y[0]
    arr[4] = arr[4] * y[1]
    arr[5] = arr[5] * y[2]
    print("Array after transformation: ",arr)
    return arr

def T_local_to_global(arr: list[float], line: Line):
    """Transformační vektor z lokálních do globálních souřadnic"""
    gamma = line.gamma
    v_transformed = [arr[0]*cos(gamma) - arr[1] * sin(gamma),
                     arr[0] * sin(gamma) + arr[1] * cos(gamma),
                     arr[2],
                     arr[3] * cos(gamma) - arr[4] * sin(gamma),
                     arr[3] * sin(gamma) + arr[4] * cos(gamma),
                     arr[5]
                     ]
    for i in range(6):
        print(f"|{arr[i]:.2f} ----> {v_transformed[i]:.2f}|")
    return v_transformed

def T_global_to_local(arr: list[float], line: Line):
    """Transformační vektor z globálních do lokálních souřadnic"""
    gamma = line.gamma
    v_transformed = [arr[0]*cos(gamma) + arr[1] * sin(gamma),
                     -arr[0] * sin(gamma) + arr[1] * cos(gamma),
                     arr[2],
                     arr[3] * cos(gamma) + arr[4] * sin(gamma),
                     -arr[3] * sin(gamma) + arr[4] * cos(gamma),
                     arr[5]
                     ]
    #for i in range(6):
        #print(f"|{arr[i]:.2f} ----> {v_transformed[i]:.2f}|")
    return v_transformed


def R_lok_prim(line: Line, arr: list[Load]):
    R = []
    alpha = line.alpha
    if alpha in (0,pi/2,pi,3*pi/2):
        alpha = 0
    for i in arr:
        l = line.length
        if alpha in (0,pi/2,pi,3*pi/2):
            a = i.pos_cg
            b = l - a
            Fz = i.load
            Fx = 0 #TODO změnit / přidat možnost horizontální síly
        else:
            a = i.pos_cg/cos(alpha) # přepona z definice cosinu
            b = l - a
            Fz = i.load * cos(alpha)
            Fx = i.load * sin(alpha)
        if i.load_type == "S":
            vector = [
                (b/l) * Fx,
                -b**2 * ((3*l -2*b) / l**3) * Fz,
                ((a*b**2)/l**2) * Fz,
                (a/l) * Fx,
                -a**2 * ((3*l-2*a) / l**3) * Fz,
                -((a**2 * b)/l**2) * Fz
            ]
            #print(f"Singular = {vector}")
            R.append(vector)

        elif i.load_type == "D":
            q = i.load_per_metre * cos(alpha)
            if line.protilehla < 0:
                n = i.load_per_metre * sin(alpha)
            else:
                n = - i.load_per_metre * sin(alpha)
            b = 0
            if i.start != 0:
                b = i.start
                #print(f" b = {b} i.start != line.node1.x_cord")
            if alpha in (0, pi / 2, pi, 3 * pi / 2):
                a = (i.fin)
            else:
                a = (i.fin)/cos(alpha)
            #print(f"a = {a} ~ b = {b}")
            vector = [
                ((2*a*l-a**2 - (2*b*l-b**2) )/(2*l)) * n,
                ((-2*a*l**3 +2*a**3*l - a**4 + (2*b*l**3 - 2*b**3*l + b**4))/(2*l**3))*q,
                ((6 * a**2 * l**2 - 8*a**3*l + 3*a**4 - (6 * b**2 * l**2 - 8*b**3*l + 3*b**4)) / (12*l**2)) * q,
                ((a**2 - b**2)/(2*l))*n,
                ((-2 * a**3 * l + a ** 4 + (2 * b**3 * l - b ** 4)) / (2 * l ** 3)) * q,
                ((-4 * a ** 3 * l + 3 * a**4 + (4 * b ** 3 * l - 3 * b**4)) / (12 * l ** 2)) * q
            ]
            #print(f"Distributed = {[f'{x:.2f}' for x in vector]}")
            R.append(vector)

        else: print("error1")
    R = np.sum(R,axis=0).tolist()
    return R

def stiffness_matrix(line: Line, is_global: bool):
    """True for GLOBAL matrix, False for LOCAL matrix"""
    if is_global:
        a = line.alpha
    else: a = 0
    E = line.young #TODO: dodělat
    A = line.area #TODO: dodělat
    l = line.length
    I = line.inertia #TODO: dodělat
    if line.connection_type == 0:
        k = [
                [(E*A)/l * cos(a)**2  + (12*E*I)/l**3 * sin(a)**2, ((E*A)/l - (12*E*I)/l**3) * cos(a) * sin(a), (6 * E*I) / l**2 *sin(a),
                - (E*A)/l * cos(a)**2  + (12*E*I)/l**3 * sin(a)**2, - ((E*A)/l - (12*E*I)/l**3) * cos(a) * sin(a), (6 * E*I) / l**2 *sin(a)
                ], #u(a)
                [((E*A)/l - (12*E*I)/l**3) * cos(a) * sin(a), (E*A)/l * sin(a)**2  + (12*E*I)/l**3 * cos(a)**2, -(6 * E*I) / l**2 * cos(a),
                 -((E * A) / l - (12 * E * I) / l ** 3) * cos(a) * sin(a), -((E * A) / l * sin(a) ** 2 + (12 * E * I) / l ** 3 * cos(a) ** 2), -(6 * E * I) / l ** 2 * cos(a)
                ], #w(a)
                [
                     (6 * E * I) / l ** 2 * sin(a), - (6 * E * I) / l ** 2 * cos(a), (4*E*I)/l,
                    -(6 * E * I) / l ** 2 * sin(a),   (6 * E * I) / l ** 2 * cos(a), (2 * E * I) / l,
                ],#φ(a)
                [-(E * A) / l * cos(a) ** 2 + (12 * E * I) / l ** 3 * sin(a) ** 2,-((E * A) / l - (12 * E * I) / l ** 3) * cos(a) * sin(a), -(6 * E * I) / l ** 2 * sin(a),
                 (E * A) / l * cos(a) ** 2 + (12 * E * I) / l ** 3 * sin(a) ** 2, ((E * A) / l - (12 * E * I) / l ** 3) * cos(a) * sin(a), - (6 * E * I) / l ** 2 * sin(a)
                ], #u(b)
                [-((E * A) / l - (12 * E * I) / l ** 3) * cos(a) * sin(a),-((E * A) / l * sin(a) ** 2 + (12 * E * I) / l ** 3 * cos(a) ** 2), (6 * E * I) / l ** 2 * cos(a),
                 ((E * A) / l - (12 * E * I) / l ** 3) * cos(a) * sin(a),(E * A) / l * sin(a) ** 2 + (12 * E * I) / l ** 3 * cos(a) ** 2, (6 * E * I) / l ** 2 * cos(a)
                 ], #w(b)
                [
                    (6 * E * I) / l ** 2 * sin(a), - (6 * E * I) / l ** 2 * cos(a), (2 * E * I) / l,
                    -(6 * E * I) / l ** 2 * sin(a), (6 * E * I) / l ** 2 * cos(a), (4 * E * I) / l,
                ],  # φ(a)
            ]
        return k

import tkinter as tk
global count_nodes,count_lines,actibe_button

count_nodes = -1
count_lines = -1
nodes = {}
lines = {}
active_button = None

def on_click(event):
    global count_nodes
    x = round(event.x / 100)
    y = round(event.y / 100)
    print(f"Clicked at: ({x}, {y})")
    if active_button:
        podpory(event, x, y)
    else:
        count_nodes += 1
        canvas.create_rectangle(x*100 - 20, y*100 - 20, x*100 + 20, y*100 + 20, outline="red", width= 4)
        canvas.create_text(x * 100 + 30, y * 100, text=str(count_nodes), font=("Arial", 14), fill="blue")
        node = Node(x,y,count_nodes)
        nodes[count_nodes] = node
        print(f"Node {count_nodes}: {nodes[count_nodes]}")


def draw_grid():
    canvas.delete("grid_line")  # Clear previous grid
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    for i in range(0, geom_x + 1, 100):
        canvas.create_line([(i, 0), (i, geom_x)], fill='gray', tags='grid_line')
        canvas.create_line([(0, i), (geom_x, i)], fill='gray', tags='grid_line')

def connect_nodes():
    try:
        node1 = int(entry1.get())
        node2 = int(entry2.get())
        if node1 in nodes and node2 in nodes:
            x1, y1 = (nodes[node1].x_cord, nodes[node1].y_cord)
            x2, y2 = (nodes[node2].x_cord, nodes[node2].y_cord)
            canvas.create_line(x1 * 100, y1 * 100, x2 * 100, y2 * 100, fill="black", width=8)
            global count_lines
            count_lines += 1
            line = Line(nodes[node1],nodes[node2])
            lines[count_lines] = line
            print(f"Line {count_lines}: {lines[count_lines]}")
            canvas.create_text(line.half[0] * 100 + 30, line.half[1] * 100, text=str(count_lines), font=("Arial", 14), fill="green")
        else:
            print("Invalid node numbers")
    except ValueError:
        print("Please enter valid numbers")

# Create the main window
root = tk.Tk()
root.title("Click to Get Coordinates")
geom_x = 1500
geom_y = 1500
root.geometry(f"{geom_x}x{geom_y}")

# Create a resizable canvas
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create input fields and button for connecting nodes
frame = tk.Frame(root)
frame.pack()
entry1 = tk.Entry(frame, width=5)
frame.pack(side=tk.RIGHT, padx=20, pady=20)
entry1 = tk.Entry(frame, width=10, font=("Arial", 14))  # Bigger input field
entry1.pack()
tk.Label(frame, text="to", font=("Arial", 14)).pack()
entry2 = tk.Entry(frame, width=10, font=("Arial", 14))  # Bigger input field
entry2.pack()
connect_button = tk.Button(frame, text="Connect", command=connect_nodes, font=("Arial", 14))
connect_button.pack(pady=10)

#podpory
button_frame = tk.Frame(frame)
button_frame.pack(pady=10)  # Add padding for spacing below "Connect" button
buttons = {}
buttons["A"] = tk.Button(button_frame, text="A", width=5, command=lambda: toggle_button("A"))
buttons["A"].grid(row=0, column=0, padx=5)
buttons["B"] = tk.Button(button_frame, text="B", width=5, command=lambda: toggle_button("B"))
buttons["B"].grid(row=0, column=1, padx=5)
buttons["C"] = tk.Button(button_frame, text="C", width=5, command=lambda: toggle_button("C"))
buttons["C"].grid(row=0, column=2, padx=5)

# Bind mouse click event and resizing event
canvas.bind("<Button-1>", on_click)
canvas.bind("<Configure>", lambda event: draw_grid())

images = {}
supports = {}
deformations = {"A" : [False,False,True],
                "B" : [True,False,True],
                "C" : [False,False,False]
                }
def podpory(event, x, y):
    print(f"Podpory function triggered by Button {active_button} at: ({x}, {y})")
    #color = {"A": "green", "B": "yellow", "C": "blue"}.get(active_button, "black")
    #canvas.create_oval(x * 100 - 10, y * 100 - 10, x * 100 + 10, y * 100 + 10, fill=color)
    # Load custom images for the buttons
    # Map button IDs to specific image paths
    image_paths = {
        "A": "images/podpora_resized.png",  # Replace with the actual file path
        "B": "images/podpora_resized.png",
        "C": "images/podpora_resized.png",
    }

    # Get the image path for the current active button
    image_path = image_paths.get(active_button, None)


    # Load the image
    img = tk.PhotoImage(file=image_path)

    # Store the image in the global dictionary to avoid garbage collection
    images[(x, y)] = img
    supports[(x, y)] = active_button
    # Draw the image on the canvas
    canvas.create_image(x * 100, y * 100 + 75, image=img, anchor=tk.CENTER)

def toggle_button(button_id):
    global active_button
    if active_button == button_id:
        active_button = None
        update_button_visuals()
        print(f"Button {button_id} unpressed.")
    else:
        active_button = button_id
        update_button_visuals()
        print(f"Button {button_id} pressed.")


def update_button_visuals():
    # Update the visual state of the buttons
    for button_id, button in buttons.items():
        if active_button == button_id:
            button.config(relief=tk.SUNKEN, state=tk.ACTIVE)
        else:
            button.config(relief=tk.RAISED, state=tk.NORMAL)


def terminal_input():
    list_of_loads = []
    while True:
        while True:
            #for x in range(count_lines+1):
            print(f"Enter loads for Line ")
            a = ask()
            print(a[0])
            temp = []
            for i in range(len(a[0])):
                load = Load(*a[0][i])
                temp.append(load)
            list_of_loads.append(temp)
            print(temp)
            print(f"num of loads {len(list_of_loads)}",f"number of lines = {count_lines}")
            print(list_of_loads)
            if len(list_of_loads) == count_lines+1:
                break
        print(list_of_loads)
        lock = input("Start? Press Y: ").capitalize()
        if lock == "Y":
            call('cls', shell=True)
            break
    for i in range(count_nodes+1):
        #print(supports.keys())
        #print((nodes[i].x_cord,nodes[i].x_cord))
        #print(f"support_keys = {supports.keys()}")
        print((nodes[i].x_cord,nodes[i].y_cord))
        if (nodes[i].x_cord,nodes[i].y_cord) in supports.keys():
            nodes[i].okrajove_podminky = deformations[supports[(nodes[i].x_cord,nodes[i].y_cord)]]
            print(f"Node {i} = {nodes[i].okrajove_podminky}")
    R_net = {}
    """Globální matice tuhosti"""
    node_deformations = {}
    count = Counter(-1)
    pismena = {0:"u",1:"w",2:"φ"}
    seznam_pismen = []
    temp = []
    for i in range(count_lines + 1):
        if lines[i].node1.num not in node_deformations.keys():
            a = []
            for x in range(3):
                if lines[i].node1.okrajove_podminky[x]:
                    a.append(count.plus_one())
                    seznam_pismen.append([pismena[x], lines[i].node1.num])
            node_deformations[lines[i].node1.num] = a

        if lines[i].node2.num not in node_deformations.keys():
            a = []
            for x in range(3):
                if lines[i].node2.okrajove_podminky[x]:
                    a.append(count.plus_one())
                    seznam_pismen.append([pismena[x], lines[i].node2.num])
            node_deformations[lines[i].node2.num] = a


    k_global = np.zeros((count.value + 1, count.value + 1))
    R_global = np.zeros(count.value + 1)
    k_list = []
    print(f"%%%%%\nNode deformations: {node_deformations}\n%%%%%")
    print("Globální matice = ", k_global)
    print("Lokální vektor = ", R_global)
    def_list = []
    line_def = []
    def_non_zero = []
    list_of_global_vectors = []
    list_of_R_stars = []
    for i in range(count_lines + 1):
        def_list.append([*lines[i].node1.okrajove_podminky,*lines[i].node2.okrajove_podminky])
        print("deformace = ",def_list[i])
        lines[i].young = 2e7
        lines[i].area = 0.04
        lines[i].inertia = 1.2e-3
        print("_\nR*",R_lok_prim(lines[i],list_of_loads[i]))
        np.linalg.pinv(stiffness_matrix(lines[i], True))
        k = stiffness_matrix(lines[i], True)
        print(f"Stiffness matrix = {k}")
        R_G = T_local_to_global(R_lok_prim(lines[i],list_of_loads[i]),lines[i])
        R_origo = R_G.copy()
        R_G = is_support(R_G,lines[i])
        def_non_zero.append([j for j, x in enumerate(def_list[i]) if x])
        print(f"DEFORMATIONS NON ZERO = {def_non_zero[i]}")
        line_def.append([*node_deformations[lines[i].node1.num],*node_deformations[lines[i].node2.num]])
        print("Line def = ",line_def[i])
        d = {key : value for key,value in zip(range(len(def_non_zero[i])),line_def[i]) }
        print("d = ", d)
        k_reduced = []
        k_temp = []
        for x in range(len(def_non_zero[i])):
            k_temp.append(k[def_non_zero[i][x]])
            temp = []
            for j in range(len(def_non_zero[i])):
                k_global[d[x]][d[j]] += k_temp[x][def_non_zero[i][j]]
                temp.append(k_temp[x][def_non_zero[i][j]])
            k_reduced.append(temp)
        print("K redukovaná",k_reduced)
        k = np.linalg.pinv(k_reduced)
        list_of_global_vectors.append(np.array([[x] for x in R_G]))
        R_G = [[-x] for x in R_G]
        R_temp = []
        R_reduced = []
        for x in range(len(def_non_zero[i])):
            R_temp.append(R_G[def_non_zero[i][x]])
            temp = []
            R_global[d[x]] += R_temp[x][0]
            temp.append(R_temp[x])
            R_reduced.append(temp)
        print("inverze =", k)
        print("Primární vektor = ",R_G)


        print("Reduced R: ",R_reduced)
        k_list.append(k)
    print("%%%%%%%%%%%\nGLOBÁLNÍ MATICE = ", k_global)
    print("GLOBÁLNÍ VEKTOR\n%%%%%%%%%%% = ", R_global)
    r_glob = np.dot(np.linalg.pinv(k_global),R_global)
    print("Globální r = ",r_glob)
    for i in range(count.value + 1):
        if count.value == 1 or i == count.value // 2:
            print(f"|{r_glob[i] * 1000:.2f}| {seznam_pismen[i][0]}({seznam_pismen[i][1]}) 10^(-3) ")
        else:
            print(f"|{r_glob[i]*1000:.2f}| {seznam_pismen[i][0]}({seznam_pismen[i][1]})")
    #########
    for i in range(count_lines + 1):
        reversed_dict = {key : value for key,value in zip(def_non_zero[i],line_def[i])}
        order = 0
        temp = []
        for j in range(6):
            if def_list[i][j]:
                temp.append(r_glob[reversed_dict[def_non_zero[i][order]]])
                order += 1
            else:
                temp.append(0)
        print(f"passed {i} ")
        print("rGLOB =",temp)
        r_loc = T_global_to_local(np.array(temp),lines[i])
        k = stiffness_matrix(lines[i],False)
        print(k)
        R_secondary_local = np.dot(k,r_loc)
        print(f"^\nR*({lines[i].node1.num},{lines[i].node2.num}) = {R_secondary_local}")
        #print("R* = ",np.array([[x] for x in R_origo]) + R_secondary_local)
        list_of_R_stars.append(list_of_global_vectors[i] + R_secondary_local.reshape(6,1))
        print("Rstar =  ",list_of_R_stars[i])
        print(f"---------------\nR* ({lines[i].node1.num},{lines[i].node2.num}) =") #"uwφuwφ"
        for a,b,c in zip(range(6),"XZMXZM",[lines[i].node1.num,lines[i].node1.num,lines[i].node1.num,lines[i].node2.num,lines[i].node2.num,lines[i].node2.num]):
            print(f"{[f'{x:.2f}' for x in list_of_R_stars[i].tolist()[a]]} {b}({c})")
        print("---------------")
    for i in range(count_lines+1):
        if lines[i].node1.num not in R_net:
            R_net[lines[i].node1.num] = list_of_R_stars[i][0:3]
        else:
            R_net[lines[i].node1.num] = R_net[lines[i].node1.num] + list_of_R_stars[i][0:3]
        if lines[i].node2.num not in R_net:
            R_net[lines[i].node2.num] = list_of_R_stars[i][3:6]
        else:
            R_net[lines[i].node2.num] = list_of_R_stars[i][lines[i].node2.num] + list_of_R_stars[i][3:6]
    print(R_net)




# Start the input thread
thread = threading.Thread(target=terminal_input, daemon=True)
thread.start()


# Run the main event loop
root.mainloop()
