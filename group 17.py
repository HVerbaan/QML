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


## Create optimization model
m = Model('TSPmodel')


#---------------Extraction of data---------------------------

with open("data_small.txt", "r") as f:          # Open Li & Lim PDPTW instance definitions
    data = f.readlines()                        # Extract instance definitions

TSP = []                                        # Create array for data related to nodes
i = 0                                           # Varible to keep track of lines in data file
for line in data:
    i = i+1
    words = line.split()
    words = [int(i) for i in words]             # Covert data from string to integer
    TSP.append(words)                           # Store node data
TSP = np.array(TSP)


#---------------Sets----------------------

V = 4                                           # Number of vehicles
H = TSP[:,0]                                    # List of all nodes
N = len(TSP[:,0])                               # Number of nodes


#---------------Parameters----------------

# Node specific
xc = TSP[:,1]                                   # X-position of nodes i
yc = TSP[:,2]                                   # Y-position of nodes i
pickup_demand = TSP[:,3]                        # Demand at the node
earliest_pickup = TSP[:,4]                      # Opening time of node i
latest_pickup = TSP[:,5]                        # Closing time of node i
service_time = TSP[:,6]                         # Service time required in node i
allowed_charging = TSP[:,7]                     # Charging at node i - binary variable (0 for no charging; 1 for available charging)


# Calculation of the euclidean distance per node 
distance=np.zeros((n,n))                                                         # Create array for distance between nodes
for i in H:
    for j in H:
        distance[i][j] = math.sqrt((xc[j] - xc[i])**2 + (yc[j] - yc[i])**2)       # Store distance between nodes


# Vehicle + Battery specific
charge = 1                                       # Charging rate of the battery in vehicle
discharge = 1                                    # Discharging rate of the battery in vehicle
travel_time = distance * 2                       # Travel cost from node i to node j
maximum_loading = 120                            # Maximum loading capacity of vehicle v (currently single type)
maximum_battery = 110                            # Maximum battery capacity of vehicle v
amount_vehicles = V


# Additional parameters
big_m = 1000                                    # Used for the Big M method


#---------------------Variables-------------------------------

# Variable 1 - Binary variable for the route (i, j) being in the solution space for vehicle v 
x = {}
for i in H:
    for j in H:
        for v in V:
            x[i,j,v] = m.addVar(vtype=GRB.BINARY, lb = 0, name="X_%s,%s,%s" %(i,j,v))


# Variable 2 - Order of the nodes in solution space of the route per vehicle
u = {}
for i in H:
    for v in V:
        u[i,v] = m.addVar(vtype=GRB.BINARY, lb = 0, name="U_%s_%s" %(i.v))

# Variable 3 - Binary variable for if the node is visited by a vehicle or not 
z = {}
for i in H:
    for v in V:
        z[i,v] = m.addVar(vtype=GRB.BINARY, lb=0, name="Z_%s,%s" %(i,v))
      
# Variable 4 - Arrival time of vehicle v at node i
w = {}
for i in H:
    for v in V:
        w[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="W_%s,%s" %(i,v))

# Variable 5 - Amount of energy charged by vehicle v at node i
q = {}
for i in H:
    for v in V:
        q[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Q_%s,%s" %(i,v))

# Variable 6 - Battery level of vehicle v at node i
cb = {}
for i in H:
    for v in V:
        cb[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cb_%s,%s" %(i,v))

# Variable 7 - Load volume of vehicle v after visiting node i
cl = {}
for i in H:
    for v in V:
        cl[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cl_%s,%s" %(i,v))
  

#-------------------Objective-------------------------------

obj = (quicksum(d[i,j]*x[i,j,v] for i in H for j in H for v in V))
m.setObjective(obj, GRB.MINIMIZE)


#------------------Constraints-------------------------------  

# Constraint 1 - All locations should be visited once
for i in N:
    m.addConstr(quicksum(x[i,j] for j in N) == 1, name='Visit_%s' % (i))
for j in N:
    m.addConstr(quicksum(x[i,j] for i in N) == 1, name='Visit_%s' % (i))

# Constraint 2 - Each customer node is visited by one vehicle
for i in N:
    for v in V:
        m.addConstr(quicksum(x[i,v]) == 1, name='Visit_%s' % (i))


#======================Continue Here!!!!!!!=============

       
## Objective
obj = (quicksum(t[i,j]*x[i,j] for i in V for j in V))
m.setObjective(obj, GRB.MINIMIZE)

#---------------------Constraints------------------------------ 
# Constraint 1 - All locations should be visited
for i in V:
    m.addConstr(quicksum(x[i,j] for j in V) == 1, name='Visit_%s' % (i))
for j in V:
    m.addConstr(quicksum(x[i,j] for i in V) == 1, name='Visit_%s' % (i))
       

# #SUBTOUR ELIMINATION     
# ##maximum and minimum values for each node except the start  
m.addConstr(u[0] == 1) 
for i in V: 
    if i != 0:
        m.addConstr(u[i] >= 2)
        m.addConstr(u[i] <= len(V))
#the order of nodes based on the decision variables  
for i in V:
    for j in V:
        if j!=0:
            m.addConstr(u[i] - u[j] + (len(V)) * x[i,j] <= len(V)-1)       
    
# # No connection to the node itself
# for i in V:
#     m.addConstr(x[i,i],GRB.EQUAL,0,name='NoC_%s' % (i)) 
    
 
m.update()
m.write('TSPmodel.lp')
m.Params.timeLimit = 3600
m.optimize()
m.write('TSPmodel.sol')

# Plot the routes that are decided to be traversed 
arc_solution = m.getAttr('x', x)
#
fig= plt.figure(figsize=(15,15))
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.scatter(xc[1:n-1],yc[1:n-1])
for i in range(1,n-1):
    plt.annotate(str(i),(xc[i],yc[i]))
plt.plot(xc[0],yc[0],c='g',marker='s')
#
for i in range(n):
    for j in range(n):
        if arc_solution[i,j] > 0.99:
            plt.plot([xc[i], xc[j]], [yc[i], yc[j]],'r--')
plt.show()          
##YOU CAN SAVE YOUR PLOTS SOMEWHERE IF YOU LIKE
##plt.savefig('Plots/TSP.png',bbox_inches='tight')   
#

for i in V:
    print("ORDER OF NODE " , i, " IS " , u[i].X)
                           

print('Obj: %g' % m.objVal)

Totaldistance = sum(t[i,j]*x[i,j].X for i in V for j in V)

print('Total distance traveled: ', Totaldistance)


