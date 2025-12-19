#=================================================================================
# Group 17
#===============================================================


#----------------Imports--------------------------------

from gurobipy import *
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('text', usetex=True)
#import os
#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'


#-----------------Model creation-------------------------


## Create optimization model
m = Model('TSPmodel')


#---------------Extraction of data---------------------------

# filename = "data_small.txt"
filename = r'C:\Users\annav\OneDrive\Documenten\Anna\Studie\Quantitative methods for logistics\Assignment_Q2\data_small.txt'


with open(filename, "r") as f:                  # Open Li & Lim PDPTW instance definitions
    data = f.readlines()                        # Extract instance definitions

TSP = []                                        # Create array for data related to nodes
i = 0                                           # Varible to keep track of lines in data file
for line in data:
    i = i+1
    words = line.split()
    words = [int(i) for i in words]             # Covert data from string to integer
    TSP.append(words)                           # Store node data
TSP = np.array(TSP)

# Dummy node creation for depot
depot_row = TSP[0, :].copy()
new_id = TSP[-1, 0] + 1 
depot_row[0] = new_id
TSP = np.vstack([TSP, depot_row])

# Charging period data

case = 0
if case == 1:
    Periods = np.array([[   0,    0,  350,    2],
                        [   1,  350, 1020,    2],
                        [   2, 1020, 1320,    2],
                        [   3, 1320, 1440,    2]])
else:
    period_filename = "data_periodsCharge.txt"
    
    with open(period_filename, "r") as f:
        period_data = f.readlines()
    
    Periods = []
    i = 0
    for line in period_data:
        i = i + 1
        words = line.split()
        words = [int(w) for w in words]
        Periods.append(words)
    Periods = np.array(Periods)


#---------------Parameters----------------

# Node specific (from data_small)
node_id = TSP[:,0]                              # List of id's of nodes i
pickup_volume = TSP[:,3]                        # Amount to be picked up at node i
earliest_pickup = TSP[:,4]                      # Opening time of node i
latest_pickup = TSP[:,5]                        # Closing time of node i
service_time = TSP[:,6]                         # Service time required in node i
allowed_charging = TSP[:,7]                     # Charging at node i - binary variable (0 for no charging; 1 for available charging)
xc = TSP[:,1]                                   # X-position of nodes i
yc = TSP[:,2]                                   # Y-position of nodes i



# Correcting Dummy Node specifics
pickup_volume[-1] = 0 
service_time[-1] = 0
allowed_charging[-1] = 0

# Charging period specific (from data_periodsCharge)
period_id = Periods[:, 0]                       # ID number of charging period p
period_start = Periods[:, 1]                    # Start time of charging period p
period_end = Periods[:, 2]                      # End time of charging period p
period_cost = Periods[:, 3]                     # Charging cost in period p (euro / time unit)


# Calculation of the euclidean distance between two nodes 
n = len(xc)
distance = np.zeros((n,n))                                                         # Create array for distance between nodes
for i in range(n):
    for j in range(n):
        if i != j:
            distance[i,j] = math.sqrt((xc[j] - xc[i])**2 + (yc[j] - yc[i])**2)
        else:
            distance[i, j] = 0    # Store distance between nodes

# Vehicle + Battery specific (choose)

# Travel cost from node i to node j

travel_time = 2 * distance


#Fixed cost per day
W = np.array([100, 100, 100, 120, 120, 120])
#Flexible cost per distance unit
Q = np.array([2, 2, 2, 1.25, 1.25, 1.25])
#Electric(0) or Diesel(1)
K = np.array([1, 1, 1, 0, 0, 0])
#Battery Capacity
maximum_battery = np.array([10e10, 10e10, 10e10, 90, 90, 90])
#Discharge rate
discharge = np.array([1, 1, 1, 0.7, 0.7, 0.7])
# Charging rate of the battery in vehicle
charge = np.array([0, 0, 0, 1.1, 1.1, 1.1])
# Maximum loading capacity of vehicle v (currently single type)
maximum_loading = np.array([100,100,100,100,100,100])


# Additional parameters
H = latest_pickup[-1] - earliest_pickup[0]      # Operation horizon (opening hours  of the depot node)
ML = maximum_loading                            # Big-M for pick up constraint
MC = H                                          # Big-M for charging time constraint
MB = 90 + discharge * H            # Big-M for battery constraint

MT = np.zeros((n,n))                            # Calculate big-M for each arc [i,j] for time evolution constraint
for i in node_id:
    for j in node_id:
        MT[i,j] = latest_pickup[j] + H + travel_time[i,j] - earliest_pickup[j]

#---------------Sets----------------------

V = range(len(K))                               # Set of vehicles
N = range(len(node_id))                                  # Set of all nodes
C = node_id[1:-1]                               # Set of all customers
start_node = node_id[0]
end_node = node_id[-1]
P = range(len(period_id))                                   # Set of all charging periods

#---------------------Variables-------------------------------

# Variable 1 - Binary variable for the route (i, j) being in the solution space for vehicle v 
x = {}
for i in N:
    for j in N:
        if i != j:
            for v in V:
                x[i,j,v] = m.addVar(vtype=GRB.BINARY, lb = 0, name="x_%s,%s,%s" %(i,j,v))

      
# Variable 3 - Arrival time of vehicle v at node i
a = {}
for i in N:
    for v in V:
        a[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="a_%s,%s" %(i,v))
        
# Variable 4 - Start time of service of vehicle v at node i
w = {}
for i in N:
    for v in V:
        w[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="w_%s,%s" %(i,v))

# Variable 5 - Total time spent at node i by vehicle v
st = {}
for i in N:
    for v in V:
        st[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="st_%s,%s" %(i,v))

# Variable 6 - Charging time at node i by vehicle v
ct = {}
for i in N:
    for v in V:
        for p in P:
            ct[i,v,p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="ct_%s,%s,%s" %(i,v,p))

# Variable 7 - Remaining battery level of vehicle v at node i
rb = {}
for i in N:
    for v in V:
        rb[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="cb_%s,%s" %(i,v))

# Variable 8 - Current load volume of vehicle v after visiting node i
cl = {}
for i in N:
    for v in V:
        cl[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="cl_%s,%s" %(i,v))

# Variable 9 - Available time window of vehicle v at node i during charging period p
tw = {}
for i in N:
    for v in V:
        for p in P:
            tw[i,v,p] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="tw_%s,%s,%s" %(i,v,p))


#-------------------Objective-------------------------------
# Constraint 1
routing_cost = quicksum(distance[i, j] * Q[v] * x[i, j, v] 
                        for i in N for j in N for v in V if i != j)
charging_cost = quicksum(period_cost[p] * ct[i, v, p]
                         for i in N for v in V for p in P)
daily_cost = quicksum( W[v] * quicksum(x[start_node,j,v] for j in C)  
                      for v in V)
m.setObjective(routing_cost + charging_cost + daily_cost, GRB.MINIMIZE)

#------------------Constraints-------------------------------  

## Constraints for tours:

# Constraint 2 - Each node only one outgoing arc and incoming arc
for v in V:
    for i in C:
        m.addConstr(quicksum(x[i,j,v] for j in N if j != i) == quicksum(x[j,i,v] for j in N if j != i), 
                    name=f"flow_out_{i}_{v}")

        
# Constraint 4 - Each customer is visited exactly one time
for i in C:
        m.addConstr(quicksum(x[j,i,v] for j in N if j != i for v in V) == 1)

# Constraint 5 - Depot outflow
for v in V:
    m.addConstr(
        quicksum(x[start_node,j,v] for j in C) <= 1,
        name=f"depot_start_{v}")

# Constraint 6 - Depot inflow
for v in V:
    m.addConstr(
        quicksum(x[i,end_node,v] for i in C) == quicksum(x[start_node,j,v] for j in C),
        name=f"depot_end_{v}")
    

## Constraints for time:
    
# Constraint 7 - Service start time should between lower and upper bound
for v in V:
    for i in C:
        m.addConstr(w[i,v] >= earliest_pickup[i] * quicksum(x[i,j,v] for j in N if j != i),
                    name=f"service_start_time_lb_{i}_{v}")
        m.addConstr(w[i,v] <= latest_pickup[i] * quicksum(x[i,j,v] for j in N if j != i), name=f"service_start_time_ub_{i}_{v}")

# Constraint 8 - Service cannot start before arrival
for v in V:
    for i in C:
        m.addConstr(a[i,v] <= w[i,v],
                    name=f"service_start_after_arrival_{i}_{v}")

# Constraint 9 - Spent time at node cannot be lower than waiting time plus service time
for v in V:
    for i in C:
        m.addConstr(w[i,v] + service_time[i] * quicksum(x[i,j,v] for j in N if j != i) <= a[i,v] + st[i,v],
                    name=f"service_within_stay_{i}_{v}")

# Constraint 10 - Spent time at node cannot be lower than charging time
for v in V:
    for i in N:
        m.addConstr(st[i,v] >= quicksum(ct[i,v,p] for p in P),
                    name=f"charge_within_stay_{i}_{v}")

# Constraint 11 - The vehicle can only be charged if it visits to that node.
for v in V:
    for i in N:
        m.addConstr(quicksum(ct[i,v,p] for p in P) <= MC * quicksum(x[i,j,v] for j in N if j != i),
                    name=f"charge_only_if_visited_{i}_{v}")

# Constraint 12 - The vehicle can only be charged if charging is allowed at that node
for v in V:
    for i in N:
        m.addConstr(quicksum(ct[i,v,p] for p in P) <= MC * allowed_charging[i],
                    name=f"charge_allowed_{i}_{v}")

# Constraint 13 - Time consistency along the route
for v in V:
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(
                    a[j,v] >= a[i,v] + st[i,v] + travel_time[i,j]
                              - MT[i,j] * (1 - x[i,j,v]),
                    name=f"time_flow_{i}_{j}_{v}")
                

## Battery constraints:

# Constraint 14 - Vehicles start at the depot with a full battery
for v in V:
    m.addConstr(rb[start_node,v] == maximum_battery[v],
                name=f"battery_start_full_{v}")
    
# Constraint 15 - Battery level must stay within lcapacity imits
for v in V:
    for i in N:
        m.addConstr(rb[i,v] >= 0,
                    name=f"battery_lb_{i}_{v}")
        m.addConstr(rb[i,v] <= maximum_battery[v],
                    name=f"battery_ub_{i}_{v}")

# Constraint 16 - Battery lower bound along used arcs
for v in V:
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(rb[j,v] >= rb[i,v] + charge[v] * quicksum(ct[i,v,p] for p in P) - discharge[v] * travel_time[i,j] - MB[v] * (1 - x[i,j,v]),
                    name=f"battery_flow_lb_{i}_{j}_{v}")

# Constraint 17 - Battery upper bound along used arcs
for v in V:
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(rb[j,v] <= rb[i,v] + charge[v] * quicksum(ct[i,v,p] for p in P) - discharge[v] * travel_time[i,j] + MB[v] * (1 - x[i,j,v]),
                    name=f"battery_flow_ub_{i}_{j}_{v}")

# Constraint 18 - Remaining battery plus charging cannot exceed battery capacity
for v in V:
    for i in N:
        m.addConstr(rb[i,v] + float(charge[v]) * quicksum(ct[i,v,p] for p in P) <= maximum_battery[v] ,
                    name=f"battery_charge_capacity_{i}_{v}")


## Constraints for vehicle picking up loads:

# Constraint 19 - Vehicles start empty at the depot
for v in V:
    m.addConstr(cl[start_node,v] == 0,
                name=f"load_start_zero_{v}")
    
# Constraint 20 - Vehicle load must stay within capacity bounds
for v in V:
    for i in N:
        m.addConstr(cl[i,v] >= 0, 
                    name=f"load_lb_{i}_{v}")  #this is already defined in the variable itself we can remove
        m.addConstr(cl[i,v] <= maximum_loading[v],
                    name=f"load_ub_{i}_{v}")

# Constraint 21 -  Load flow lower bound when an arc is used
for v in V:
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(
                    cl[j,v] >= cl[i,v] + pickup_volume[j]
                               - ML[v] * (1 - x[i,j,v]),
                    name=f"load_flow_lb_{i}_{j}_{v}")

# Constraint 22 - Load flow upper bound when an arc is used
for v in V:
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(
                    cl[j,v] <= cl[i,v] + pickup_volume[j]
                               + ML[v] * (1 - x[i,j,v]),
                    name=f"load_flow_ub_{i}_{j}_{v}")
                
                
## Constraints for period-based charging:

# Constraint 23 - charging time allocation across periods
for i in N:
    for v in V:
        m.addConstr(
            quicksum(tw[i, v, p] for p in P) == st[i, v],
            name=f"tw_allocation_{i}_{v}")

# Constraint 24 - charging within a period cannot exceed time availability
for i in N:
    for v in V:
        for p in P:
            m.addConstr(
                ct[i, v, p] <= tw[i, v, p],
                name=f"charging_within_tw_{i}_{v}_{p}")



# ----- Solve ------

m.update()
m.Params.timeLimit = 3600
m.optimize()

# Solution status check
if m.status == GRB.OPTIMAL:
    m.write('TSPmodel.sol')
    print("\nOptimal solution found.")

elif m.status == GRB.INFEASIBLE:
    print("\nModel is infeasible.")
    m.computeIIS()
    m.write("model.ilp")
    print("IIS written to model.ilp")
    pass 

else:
    print(f"\nSolver ended with status {m.status}")
    
# ----- Print results ----
if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):

    print("\n--- Results Summary ---")
    print(f"Total Distance Travelled (Objective Value): {m.ObjVal:.2f}")

    print("\n--- Detailed Vehicle Routes ---")

    vals_x = m.getAttr('X', x)
    routes_data = {}

    for v in V:

        current_node = start_node
        route_nodes = [current_node]
        route_details = []
        visited = set([current_node])

        # --- safe route construction ---
        while current_node != end_node:

            next_node = None
            for j in N:
                if j != current_node and vals_x[current_node, j, v] > 0.5:
                    next_node = j
                    break

            if next_node is None:
                break

            #  loop protection
            if next_node in visited:
                print(f"Vehicle {v+1}: loop detected at node {next_node}, stopping route trace.")
                break

            visited.add(next_node)
            route_nodes.append(next_node)
            current_node = next_node

        if len(route_nodes) <= 1:
            continue

        print(f"\nVehicle {v+1} Route:")

        for i in route_nodes:
            arr_time = a[i,v].X
            load_level = cl[i,v].X
            batt_level = rb[i,v].X
            charge_time = sum(ct[i,v,p].X for p in P)
            charged_amount = charge[v] * charge_time

            print(
                f" -> Node {i} "
                f"(Time: {arr_time:.1f}, "
                f"Load: {load_level:.1f}, "
                f"Battery: {batt_level:.1f}, "
                f"Charge Time: {charge_time:.1f}, "
                f"Charged: {charged_amount:.1f})"
            )

            route_details.append({
                "Node": i,
                "Arrival": arr_time,
                "Load": load_level,
                "Battery": batt_level,
                "ChargeTime": charge_time,
                "Charged": charged_amount,
                "x": xc[i],
                "y": yc[i]
            })

        routes_data[v] = pd.DataFrame(route_details)

if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    print("No feasible solution found. Skipping plots.")
    raise SystemExit


#%%
# ----- Plotting routes -----    
    
# --- Route Map (Color coded by Battery Level) ---
plt.figure(figsize=(12, 12))
    
# Colors for vehicles
vehicle_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange']

# --- Plot Depot ---

plt.scatter(
    xc[start_node], yc[start_node],
    c='red', s=200, marker='s',
    edgecolors='black', linewidth=1.5,
    zorder=4, label='Depot'
)

# --- Plot Nodes ---
customers_labeled = False
charging_used_labeled = False
charging_station_labeled = False

for i in N:
    if i == start_node or i == end_node:
        continue 
    
    # checking if charging happened at node i
    charged_here = sum(ct[i, v, p].X for v in V for p in P) > 1e-6

    if allowed_charging[i] == 1:
        plt.scatter(
            xc[i], yc[i],
            c='gold', marker='^', s=140,
            edgecolors='black', zorder=3,
            label='Charging Stations' if not charging_station_labeled else ""
        )
        charging_station_labeled = True

        # overlay marker if charging actually happened
    if charged_here:
        plt.scatter(
            xc[i], yc[i],
            c='red', marker='*', s=220,
            edgecolors='black', zorder=4,
            label='Charging Used' if not charging_used_labeled else ""
        )
        charging_used_labeled = True

    else:
        plt.scatter(
            xc[i], yc[i],
            c='lightblue', marker='o', s=80,
            edgecolors='black', zorder=3,
            label='Customers' if not customers_labeled else ""
        )
        customers_labeled = True

    plt.annotate(str(i), (xc[i] + 0.5, yc[i] + 0.5), fontsize=15)

# --- Plot Routes ---

for v, df in routes_data.items():

    color = vehicle_colors[v % len(vehicle_colors)]
    xs = df["x"].values
    ys = df["y"].values

    for i in range(len(xs) - 1):
        dx = xs[i+1] - xs[i]
        dy = ys[i+1] - ys[i]

        plt.arrow(
            xs[i], ys[i],
            dx, dy,
            length_includes_head=True,
            head_width=1.2,
            head_length=1.8,
            color=color,
            linewidth=2.2,
            alpha=0.85,
            zorder=2
        )

    plt.plot(xs, ys, color=color, linewidth=1.8, alpha=0.6, label=f'Vehicle {v+1}')

# --- Layout ---

plt.title("EVRPTW Routes and Charging Stations")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("QML_f_routes.png")
plt.show()

print(f"Fixed cost: {daily_cost.getValue()}")
print(f"Charging cost: {charging_cost.getValue()}")
print(f"Variable cost: {routing_cost.getValue()}")
if m.status == GRB.OPTIMAL:
    print("Total cost:", m.objVal)
    print("Optimality gap:", m.MIPGap)
    print("Computation time [seconds]:", m.Runtime)
    
#%%
# --------- Plotting battery evolution % ----------------


charge_list = [
    # Vehicle 1
    [],
    # Vehicle 3
    [],
    # Vehicle 2
    [],
    # Vehicle 4
    [903.3,0]]

charge_increase = [
    # Vehicle 1
    [],
    # Vehicle 3
    [],
    # Vehicle 2
    [],
    # Vehicle 4
    [3]]

y_battery = [
    # Vehicle 1
    [],
    # Vehicle 3
    [],
    # Vehicle 2
    [],
    # Vehicle 4
    [90, 46.9, 35.9, 26, 21.8, 13.3,0]]

x_time = [
    # Vehicle 1
    [], #0.0, 30.3, 176.8, 277.6, 378.4, 627.1, 1600
    # Vehicle 3
    [], # 0.0, 20.0, 168.3, 842, 946.1, 1119, 1600
    # Vehicle 2
    [], # 0.0, 88.9, 213, 320.2, 1073, 1171.9, 1600
    # Vehicle 4
    [0.0, 405.0, 510.6, 705.1, 801.1, 903.3,1600]]



def plotting_battery(x_time, y_battery, charge_list, charge_increase):

    plt.figure(figsize=(8, 5),dpi=300)
    labels = ['Vehicle 1', 'Vehicle 2', 'Vehicle 3', 'Vehicle 4']
    lines = []

    # Plot battery curves
    for i in range(len(y_battery)):
        line, = plt.plot(
            x_time[i],
            y_battery[i],
            marker='o',
            label=labels[i]
        )
        lines.append(line)

    # Plot charge annotations
    for v in range(len(charge_list)):

        if not charge_list[v] or not charge_increase[v]:
            continue

        color = lines[v].get_color()

        # Loop through charge points, only as many as charge_increase values
        for j, inc in enumerate(charge_increase[v]):
            x_charge = charge_list[v][2*j]

            # Find closest y value on battery curve
            x_arr = x_time[v]
            y_arr = y_battery[v]
            idx = min(range(len(x_arr)), key=lambda k: abs(x_arr[k] - x_charge))
            y_charge = y_arr[idx]

            # New dot at charged value
            y_new = y_charge + inc
            plt.plot(x_charge, y_new, marker='o', color=color, markersize=8)

            # Vertical line from original point to new dot
            plt.vlines(
                x_charge,
                y_charge,
                y_new,
                linestyles='-.',
                colors=color,
                linewidth=2
            )

            # Annotation pointing at the new dot
            plt.annotate(
                f"+{inc:.1f}",
                xy=(x_charge, y_new),
                xytext=(x_charge + 80, y_new + 5),
                arrowprops=dict(arrowstyle="->", color=color),
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9)
            )

    plt.xlabel("Time unit")
    plt.ylabel("Battery level")
    plt.title("Battery level throughout time")
    plt.legend()
    plt.grid(True)
    plt.savefig("part_f_battery.png", dpi=300, bbox_inches="tight")
    plt.show()




# =========================
# RUN
# =========================

plotting_battery(x_time, y_battery, charge_list, charge_increase)


# --------- Plotting volume evolution % ----------------

y_load = [
    # Vehicle 1
    [0, 10, 20, 60, 80, 100, 100],
    # Vehicle 3
    [0, 10, 30, 40, 80, 100, 100],
    # Vehicle 2
    [0, 10, 40, 50, 70, 80, 80],
    # Vehicle 4
    [0, 30, 40, 50, 60, 80,80]]

x_time = [
    # Vehicle 1
    [0.0, 30.3, 176.8, 277.6, 378.4, 627.1, 1600],
    # Vehicle 3
    [0.0, 20.0, 168.3, 842, 946.1, 1119, 1600],
    # Vehicle 2
    [0.0, 88.9, 213, 320.2, 1073, 1171.9, 1600],
    # Vehicle 4
    [0.0, 405.0, 510.6, 705.1, 801.1, 903.3,1600]]


def plotting_load(x_time, y_load):
    plt.figure(figsize=(8, 5))
    labels = ['Vehicle 1', 'Vehicle 2', 'Vehicle 3', 'Vehicle 4']

    for i in range(len(y_load)):
        plt.plot(
            x_time[i],
            y_load[i],
            marker='o',
            label=labels[i]
        )

    plt.xlabel("Time unit")
    plt.ylabel("Volume in vehicle")
    plt.title("Volume in vehicle throughout time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"part_f_volume.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# RUN
# =========================

plotting_load(x_time, y_load)




