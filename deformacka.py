import threading
from math import sqrt,atan,pi,sin,cos
from ask import *
import numpy as np
import tkinter as tk
from subprocess import call
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from classes import *
from funkce import *
from myStrian import main


class Counter:
    def __init__(self, value):
        self.value = value

    def plus_one(self):
        self.value += 1
        return self.value

    def add(self, number):
        self.value += number
        return self.value


global count_nodes,count_lines,active_button

count_nodes = -1
count_lines = -1
nodes = {}
lines = {}
active_button = None

def on_click(event):
    global count_nodes
    x = round_to_nearest_half_or_one(event.x / 100)
    y = round_to_nearest_half_or_one(event.y / 100)
    print(f"Clicked at: ({x}, {y})")
    if active_button:
        podpory(event, x, y)
    else:
        count_nodes += 1
        canvas.create_rectangle(x*100 - 20, y*100 - 20, x*100 + 20, y*100 + 20, outline="red", width= 4)
        canvas.create_text(x * 100 , y * 100, text=str(count_nodes), font=("Arial", 20), fill="blue")

        node = Node(x,y,count_nodes)
        nodes[count_nodes] = node
        print(f"Node {count_nodes}: {nodes[count_nodes]}")


def draw_grid():
    canvas.delete("grid_line")  # Clear previous grid
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    for i in range(0, geom_x + 1, 50):
        if i % 100 == 0:
            canvas.create_line([(i, 0), (i, geom_x)], fill='gray', tags='grid_line')
            canvas.create_line([(0, i), (geom_x, i)], fill='gray', tags='grid_line')
        """else:
            
            canvas.create_line([(i, 0), (i, geom_x)], fill='blue', tags='grid_line')
            canvas.create_line([(0, i), (geom_x, i)], fill='blue', tags='grid_line')"""

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
button_images = {
    "A": tk.PhotoImage(file="images/button_neposuvna.png"),
    "B": tk.PhotoImage(file="images/button_kluzna.png"),
    "C": tk.PhotoImage(file="images/button_vetknuti.png"),
    "D": tk.PhotoImage(file="images/button_rotated.png"),  # Add new button image
}

# Create buttons with images
for idx, key in enumerate(button_images.keys()):  # Automatically iterate over all buttons
    buttons[key] = tk.Button(button_frame, image=button_images[key], command=lambda k=key: toggle_button(k))
    buttons[key].grid(row=0, column=idx, padx=5)

# Bind mouse click event and resizing event
canvas.bind("<Button-1>", on_click)
canvas.bind("<Configure>", lambda event: draw_grid())

images = {}
supports = {}
deformations = {"A" : [False,False,True],
                "B" : [True,False,True],
                "C" : [False,False,False],
                "D" : [False,True,True]
                }
def podpory(event, x, y):
    print(f"Podpory function triggered by Button {active_button} at: ({x}, {y})")
    #color = {"A": "green", "B": "yellow", "C": "blue"}.get(active_button, "black")
    #canvas.create_oval(x * 100 - 10, y * 100 - 10, x * 100 + 10, y * 100 + 10, fill=color)
    # Load custom images for the buttons
    # Map button IDs to specific image paths
    image_paths = {
        "A": "images/podpora_resized.png",  # Replace with the actual file path
        "B": "images/kluzna.png",
        "C": "images/vetknuti.png",
        "D": "images/rotated.png"
    }

    # Get the image path for the current active button
    image_path = image_paths.get(active_button, None)


    # Load the image
    img = tk.PhotoImage(file=image_path)

    # Store the image in the global dictionary to avoid garbage collection
    images[(x, y)] = img
    supports[(x, y)] = active_button
    # Draw the image on the canvas
    match active_button:
        case "A":
            canvas.create_image(x * 100, y * 100 + 65, image=img, anchor=tk.CENTER)
        case "B":
            canvas.create_image(x * 100, y * 100 + 60, image=img, anchor=tk.CENTER)
        case "C":
            canvas.create_image(x * 100 + 40, y * 100, image=img, anchor=tk.CENTER)
        case "D":
            canvas.create_image(x * 100 + 60, y * 100, image=img, anchor=tk.CENTER)

    #canvas.create_image(x * 100 + 40, y * 100 , image=img, anchor=tk.CENTER)

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

global points
points = []
def terminal_input(points):
    list_of_loads = []

    count = 0
    while True:
        global zero
        zero = True
        while True:
            #for x in range(count_lines+1):
            print(f"Enter loads for Line ")
            a = ask()
            print(a[0])
            if zero:
                list_form = [[] for x in range(count_lines + 1)]
            temp = []
            for i in range(len(a[0])):
                if lines[len(list_of_loads)].alpha in (0,pi/2,pi,3*pi/2):
                    list_form[count].append(a[0][i])
                else:
                    if a[0][i][2] == "S":
                        list_form[count].append(single_load(pos=a[0][i][1],load=a[0][i][0],alpha=lines[len(list_of_loads)].alpha,called=True ))
                    elif a[0][i][2] == "D":
                        list_form[count].append(distributed_load(pos=a[0][i][1],load=a[0][i][4],alpha=lines[len(list_of_loads)].alpha,called=True,start=a[0][i][5],fin = a[0][i][6]))
                    else:
                        raise ValueError("error")
                load = Load(*a[0][i])
                temp.append(load)
            count += 1
            list_of_loads.append(temp)
            print(temp)
            print(f"num of loads {len(list_of_loads)}",f"number of lines = {count_lines}")
            print(list_of_loads)
            zero = False
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
    Forces_nodes = {}
    """Globální matice tuhosti"""
    node_deformations = {} #Node : arr [deformation(Node.num)]
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
    list_of_local_vectors = []
    list_of_global_vectors = []
    list_of_R_stars = []
    list_of_forces_in_nodes = []
    for i in range(count_lines + 1):
        def_list.append([*lines[i].node1.okrajove_podminky,*lines[i].node2.okrajove_podminky])
        print("deformace = ",def_list[i])
        lines[i].young = 20e6
        lines[i].area = 0.06
        lines[i].inertia = 0.5e-3
        print("_\nR*",R_lok_prim(lines[i],list_of_loads[i]))
        k = stiffness_matrix(lines[i], True)
        print(f"Stiffness matrix = {k}")
        list_of_local_vectors.append(R_lok_prim(lines[i],list_of_loads[i]))
        print("Lokální primární vektor = ", list_of_local_vectors[i])
        R_G = T_local_to_global(R_lok_prim(lines[i],list_of_loads[i]),lines[i]) #tady smazáno .copy()
        R_origo = R_G.copy()
        list_of_global_vectors.append(np.array([[x] for x in R_origo]))
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
    print("GLOBÁLNÍ VEKTOR = ", R_global,"\n%%%%%%%%%%%")
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
        list_of_R_stars.append(T_global_to_local(list_of_global_vectors[i],lines[i]) + R_secondary_local.reshape(6,1))
        list_of_forces_in_nodes.append(T_local_to_global(list_of_R_stars[i],lines[i]))
        #list_of_R_stars.append(list_of_local_vectors[i] + R_secondary_local.reshape(6, 1))
        print("Rstar =  ",list_of_R_stars[i])
        print(f"---------------\nR* ({lines[i].node1.num},{lines[i].node2.num}) =") #"uwφuwφ"
        for a,b,c in zip(range(6),"XZMXZM",[lines[i].node1.num,lines[i].node1.num,lines[i].node1.num,lines[i].node2.num,lines[i].node2.num,lines[i].node2.num]):
            match a:
                case 1:
                    list_form[i].append(Load(*list_of_R_stars[i].tolist()[a],0,"R","a").return_load())
                case 2:
                    temp = list_of_R_stars[i].tolist()[a]
                    if abs(round(temp[0], 4)) != 0:
                        list_form[i].append(Load(*list_of_R_stars[i].tolist()[a], 0, "M").return_load())
                case 4:
                    list_form[i].append(Load(*list_of_R_stars[i].tolist()[a], lines[i].length, "R", "b").return_load())
                case 5:
                    temp =  list_of_R_stars[i].tolist()[a]
                    if abs(round(temp[0],4)) != 0:
                        list_form[i].append(Load(*temp, lines[i].length, "M").return_load())

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
    for i in range(len(R_net)):
        print(f"Node({i}): \n\t\t\tX({i}) = {R_net[i][0][0]}\n\t\t\tZ({i}) = {R_net[i][1][0]}\n\t\t\tM({i}) = {R_net[i][2][0]:.2f}")

        #list_of_loads[i].append(Load())#here
    print("Globální síly v uzlech: ")
    for i in range(count_lines+1):
        if lines[i].node1.num not in Forces_nodes:
            Forces_nodes[lines[i].node1.num] = list_of_forces_in_nodes[i][0:3]
        else:
            Forces_nodes[lines[i].node1.num] = Forces_nodes[lines[i].node1.num] + list_of_forces_in_nodes[i][0:3]
        if lines[i].node2.num not in Forces_nodes:
            Forces_nodes[lines[i].node2.num] = list_of_forces_in_nodes[i][3:6]
        else:
            Forces_nodes[lines[i].node2.num] = list_of_forces_in_nodes[i][lines[i].node2.num] + list_of_forces_in_nodes[i][3:6]
    for i in range(len(Forces_nodes)):
        print(f"Node({i}): \n\t\t\tX({i}) = {Forces_nodes[i][0][0]}\n\t\t\tZ({i}) = {Forces_nodes[i][1][0]}\n\t\t\tM({i}) = {Forces_nodes[i][2][0]:.2f}")
    print(f"List of loads: {list_form}")

    def plot_graph(a, b):
        plt.figure()
        plt.plot(a, b)
        plt.show()

    for i in range(count_lines + 1):
        a, b, c, d = main(list_form[i], True)
        if i == 0:
            with open("test.csv","w") as file:
                for i,j in zip(c,d):
                    file.write(f"{i},{j}\n")


        points.append([c,d])
        #draw_moments(canvas,c,d,lines[i])





# Start the input thread
thread = threading.Thread(target=terminal_input,args=(points,), daemon=True)
thread.start()
def check_points():
    """Check the queue and draw lines if data is available."""
    global points
    while points:
        magnitude = 20
        for i in range(count_lines+1):
            #print(points)
            draw_moments(canvas,points[i][0],points[i][1]/magnitude,lines[i])
        #canvas.delete('grid_line')
        points = []
    root.after(100, check_points)
check_points()
# Run the main event loop
root.mainloop()
