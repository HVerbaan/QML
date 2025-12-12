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

filename = "data_small.txt"
#filename = r"C:\Users\annav\OneDrive\Documenten\Anna\Studie\Quantitative methods for logistics\Assignment_Q2\data_small.txt"

with open(filename, "r") as f:          # Open Li & Lim PDPTW instance definitions
    data = f.readlines()                        # Extract instance definitions

TSP = []                                        # Create array for data related to nodes
i = 0                                           # Varible to keep track of lines in data file
for line in data:
    i = i+1
    words = line.split()
    words = [int(i) for i in words]             # Covert data from string to integer
    TSP.append(words)                           # Store node data
TSP = np.array(TSP)



#%%

#---------------Parameters----------------

# Node specific (from datafile)
node_id = TSP[:,0]                              # List of id's of nodes i
xc = TSP[:,1]                                   # X-position of nodes i
yc = TSP[:,2]                                   # Y-position of nodes i
pickup_volume = TSP[:,3]                        # Amount to be picked up at node i
earliest_pickup = TSP[:,4]                      # Opening time of node i
latest_pickup = TSP[:,5]                        # Closing time of node i
service_time = TSP[:,6]                         # Service time required in node i
allowed_charging = TSP[:,7]                     # Charging at node i - binary variable (0 for no charging; 1 for available charging)


# Calculation of the euclidean distance between two nodes 
n = len(xc)
distance=np.zeros((n,n))                                                         # Create array for distance between nodes
for i in node_id:
    for j in node_id:
        distance[i][j] = math.sqrt((xc[j] - xc[i])**2 + (yc[j] - yc[i])**2)       # Store distance between nodes


# Vehicle + Battery specific (choose)
charge = 1                                       # Charging rate of the battery in vehicle
discharge = 1                                    # Discharging rate of the battery in vehicle
travel_time = distance * 2                       # Travel cost from node i to node j
maximum_loading = 120                            # Maximum loading capacity of vehicle v (currently single type)
maximum_battery = 110                            # Maximum battery capacity of vehicle v
K = 4                                            # Amount of vehicles in fleet


# Additional parameters
H = latest_pickup[0] - earliest_pickup[0]       # Operation horizon (opening hours  of the depot node)
ML = maximum_loading                            # Big-M for pick up constraint
MC = H                                          # Big-M for charging time constraint
MB = maximum_battery + discharge*H              # Big-M for battery constraint

MT = np.zeros((n,n))                            # Calculate big-M for each arc [i,j] for time evolution constraint
for i in node_id:
    for j in node_id:
        MT[i,j] = latest_pickup[j] + H + travel_time[i][j] - earliest_pickup[j]     


#---------------Sets----------------------

V = range(K)                                    # Set of vehicles
N = node_id                                     # Set of all nodes


#---------------------Variables-------------------------------

# Variable 1 - Binary variable for the route (i, j) being in the solution space for vehicle v 
x = {}
for i in N:
    for j in N:
        for v in V:
            x[i,j,v] = m.addVar(vtype=GRB.BINARY, lb = 0, name="X_%s,%s,%s" %(i,j,v))


# Variable 2 - Binary variable for if node i is visited by a vehicle or not 
z = {}
for i in N:
    for v in V:
        z[i,v] = m.addVar(vtype=GRB.BINARY, lb=0, name="Z_%s,%s" %(i,v))
      
# Variable 3 - Arrival time of vehicle v at node i
a = {}
for i in N:
    for v in V:
        a[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="W_%s,%s" %(i,v))
        
# Variable 4 - Start time of service of vehicle v at node i
w = {}
for i in N:
    for v in V:
        w[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=earliest_pickup[i],
                          ub=latest_pickup[i], name="W_%s,%s" %(i,v))

# Variable 5 - Total time spent at node i by vehicle v
alpha = {}
for i in N:
    for v in V:
        alpha[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Q_%s,%s" %(i,v))

# Variable 6 - Time spent charging at node i by vehicle v
beta = {}
for i in N:
    for v in V:
        beta[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=maximum_battery, name="Q_%s,%s" %(i,v))

# Variable 7 - Battery level of vehicle v at node i
cb = {}
for i in N:
    for v in V:
        cb[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cb_%s,%s" %(i,v))

# Variable 8 - Load volume of vehicle v after visiting node i
cl = {}
for i in N:
    for v in V:
        cl[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=maximum_loading, name="Cl_%s,%s" %(i,v))
  

#-------------------Objective-------------------------------

obj = (quicksum(distance[i,j]*x[i,j,v] for i in N for j in N for v in V))
m.setObjective(obj, GRB.MINIMIZE)




#------------------Constraints-------------------------------  

## Constraints for tours:

# Constraint 1 - Each node only one outgoing arc
for v in V:
    for i in N:
        m.addConstr(quicksum(x[i,j,v] for j in N if j != i) == z[i,v], 
                    name=f"flow_out_{i}_{v}")

# Constraint 2 - Each node only one incoming arc
for v in V:
    for j in N:
        m.addConstr(
            quicksum(x[i,j,v] for i in N if i != j) == z[j,v],
            name=f"flow_in_{j}_{v}")
        
# Constraint 3 - Visit all nodes except the depot exactly one time
for i in N:
    if i != 0:
        m.addConstr(
            quicksum(z[i,v] for v in V) == 1,
            name=f"visit_once_{i}")
        
# Constraint 4 - N number of vehicles are allowed to leave the depot
for v in V:
    m.addConstr(
        quicksum(z[0,v] for v in V) == K,
        name=f"depot_out_{v}"
    )
    


## Constraints for time:
    
# Constraint 5 - Service time starting after arrival time at node
for v in V:
    for i in N:
        m.addConstr(a[i,v] <= w[i,v], name=f"service_time_{v}_at_{i}")

# Constraint 6 - The vehicle doesn't stay for longer than the total time spent at the node
for v in V:
    for i in N:
        m.addConstr(
            w[i,v] + service_time[i]*z[i,v] <= a[i,v] + alpha[i,v], 
            name=f"time_at_node_{i}_by_{v}")
        
# Constraint 7 - Charging time is not longer than the total time spent at the node
for v in V:
    for i in N:
        m.addConstr(
            beta[i,v] <= alpha[i,v], 
            name=f"chargetime_at_{i}_by_{v}")

# Constraint 8 - Charging time is positive only at nodes that are visited
for v in V:
    for i in N:
        m.addConstr(
            beta[i,v] <= MC * z[i,v], 
            name=f"chargetime_no_node_{i}_by_{v}")

# Constraint 9 - Charging time is positive only at nodes that allow charging
for v in V:
    for i in N:
        m.addConstr(
            beta[i,v] <= MC * allowed_charging[i], 
            name=f"chargestation_at_{i}_by_{v}")

# Constraint 10 - Arrival time at node j cannot be earlier than the sum of arrival time at previous node and time spent there
for v in V:
    for i in N:
        if i != j:
            m.addConstr(
                a[j,v] >= a[i,v] + alpha[i,v] + travel_time[i,j] - MT[i,j] * (1 - x[i,j,v]), 
                name=f"chargestation_at_{i}_by_{v}")
        
        
        
## Constraints for battery        
        
# Constraint 11 - Full battery charge when starting at the depot
for v in V:
    m.addConstr(
        cb[0,v] == maximum_battery, 
        name=f"chargestation_at_{i}_by_{v}")

# Constraint 12 - The vehicle needs enough charge to reach the next node before departure
for v in V:
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(
                    cb[j,v] >= cb[i,v] - discharge * travel_time[i,j] + charge * beta[j,v] - MB * (1-x[i,j,v]), 
                    name=f"chargestation_at_{i}_by_{v}")

# Constraint 13 - The vehicle needs enough charge to reach the next node before departure
for v in V:
    for i in N:
        for j in N:
            if i != j:
                m.addConstr(
                    cb[j,v] <= cb[i,v] - discharge * travel_time[i,j] + charge * beta[j,v] + MB * (1-x[i,j,v]), 
                    name=f"chargestation_at_{i}_by_{v}")
        
# Constraint 14 - Constraint for charging with respect to the maximum battery capacity
for v in V:
    for i in N:
        m.addConstr(
            cb[j,v] + discharge * travel_time[i,j] <= maximum_battery, 
            name=f"chargestation_at_{i}_by_{v}")

# Constraint 15 - Charge in the battery is sufficient to reach the next node
for v in V:
    for i in N:
        if i != j:
            m.addConstr(
                cb[j,v] >= discharge * travel_time[i,j] * x[i,j,v], 
                name=f"chargestation_at_{i}_by_{v}")


## Constraints for vehicle

# Constraint 16 - Load volume is 0 at depot 
for v in V:
    m.addConstr(
        cl[0,v] == 0, 
        name=f"volume_at_depot_by_{v}")
    
# Constraint 17 - The available storage in vehicle should be sufficient to pick up the load at node j
for v in V:
    for j in N:
        for i in N:
            if i != j:
                m.addConstr(
                    cl[j,v] >= cl[i,v] + pickup_volume[j] - ML * (1 - x[i,j,v]), 
                    name=f"chargestation_at_{i}_by_{v}")   

# Constraint 18 - The available storage in vehicle should be sufficient to pick up the load at node j
for v in V:
    for j in N:
        for i in N:
            if i != j:
                m.addConstr(
                    cl[j,v] <= cl[i,v] + pickup_volume[j] + ML * (1 - x[i,j,v]), 
                    name=f"chargestation_at_{i}_by_{v}")  
            
# Constraint 19 - Volume after pickup does not exceed capacity
for v in V:
        for i in N:
            m.addConstr(
                quicksum(pickup_volume[i] * z[i,v] for i in N) <= maximum_loading,
                name=f"chargestation_at_{i}_by_{v}")   
    
    
#------------------Running model-------------------------------   

m.update()
m.write('TSPmodel.lp')
m.Params.timeLimit = 3600
m.optimize()
m.write('TSPmodel.sol')



###OLD CODE FROM EXAMPLE: ###
#======================Continue Here!!!!!!!=============

       
## Objective
#obj = (quicksum(t[i,j]*x[i,j] for i in V for j in V))
#m.setObjective(obj, GRB.MINIMIZE)

#---------------------Constraints------------------------------ 
# Constraint 1 - All locations should be visited
#for i in V:
#    m.addConstr(quicksum(x[i,j] for j in V) == 1, name='Visit_%s' % (i))
#for j in V:
#    m.addConstr(quicksum(x[i,j] for i in V) == 1, name='Visit_%s' % (i))
       

# #SUBTOUR ELIMINATION     
# ##maximum and minimum values for each node except the start  
#m.addConstr(u[0] == 1) 
#for i in V: 
#    if i != 0:
#        m.addConstr(u[i] >= 2)
#        m.addConstr(u[i] <= len(V))
#the order of nodes based on the decision variables  
#for i in V:
#    for j in V:
#        if j!=0:
#            m.addConstr(u[i] - u[j] + (len(V)) * x[i,j] <= len(V)-1)       
    
# # No connection to the node itself
# for i in V:
#     m.addConstr(x[i,i],GRB.EQUAL,0,name='NoC_%s' % (i)) 
    
 
#m.update()
#m.write('TSPmodel.lp')
#m.Params.timeLimit = 3600
#m.optimize()
#m.write('TSPmodel.sol')

# Plot the routes that are decided to be traversed 
#arc_solution = m.getAttr('x', x)
#
#fig= plt.figure(figsize=(15,15))
#plt.xlabel('x-coordinate')
#plt.ylabel('y-coordinate')
#plt.scatter(xc[1:n-1],yc[1:n-1])
#for i in range(1,n-1):
#    plt.annotate(str(i),(xc[i],yc[i]))
#plt.plot(xc[0],yc[0],c='g',marker='s')
#
#for i in range(n):
#    for j in range(n):
#        if arc_solution[i,j] > 0.99:
#            plt.plot([xc[i], xc[j]], [yc[i], yc[j]],'r--')
#plt.show()          
##YOU CAN SAVE YOUR PLOTS SOMEWHERE IF YOU LIKE
##plt.savefig('Plots/TSP.png',bbox_inches='tight')   
#

#for i in V:
#    print("ORDER OF NODE " , i, " IS " , u[i].X)
                           

#print('Obj: %g' % m.objVal)

#Totaldistance = sum(t[i,j]*x[i,j].X for i in V for j in V)

#print('Total distance traveled: ', Totaldistance)


