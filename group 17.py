
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

#---Extraction of data---
with open("data_small.txt", "r") as f:          # Open Li & Lim PDPTW instance definitions
    data = f.readlines()                        # Extract instance definitions

TSP = []                                        # Create array for data related to nodes
i=0                                             # Varible to keep track of lines in data file
for line in data:
    i=i+1
    words = line.split()
    words=[int(i) for i in words]               # Covert data from string to integer
    TSP.append(words)                           # Store node data
TSP = np.array(TSP)

#---Sets---
V= 4                                            # Vehicles
H=TSP[:,0]                                      # Nodes
N=len(TSP[:,0])                                        # Number of nodes
#---Parameters---

xc=TSP[:,1]                                     # X-position of nodes i
yc=TSP[:,2]                                     # Y-position of nodes i
pickup_demand=TSP[:,3] #something
earliest_pickup=TSP[:,4] #time
latest_pickup=TSP[:,5] #time
service_time=TSP[:,6] #time
allowed_charging=TSP[:,7] #binary

# Calculation of the euclidean distance
distance=np.zeros((n,n))                               # Create array for distance between nodes
for i in H:
    for j in H:
        distance[i][j]=math.sqrt((xc[j] - xc[i])**2 + (yc[j] - yc[i])**2) # Store distance between nodes

charge= 1
discharge= 1
travel_time= distance * 2 
maximum_loading=120
maximum_battery=110
amount_vehicles=4
big_m=1000


#---Variables---
#Variable 1: binary variable if it in the solution space
x = {}
for i in H:
    for j in H:
        for v in V:
            x[i,j,v] = m.addVar(vtype=GRB.BINARY, lb = 0, name="X_%s,%s,%s" %(i,j,v))

u = {}
for i in H:
        u[i] = m.addVar(vtype=GRB.BINARY, lb = 0, name="U_%s" %(i))

z = {}
for i in H:
    for v in V:
        z[i,v] = m.addVar(vtype=GRB.BINARY, lb=0, name="Z_%s,%s" %(i,v))
      
w = {}
for i in H:
    for v in V:
        w[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="W_%s,%s" %(i,v))

q = {}
for i in H:
    for v in V:
        q[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Q_%s,%s" %(i,v))


cb = {}
for i in H:
    for v in V:
        cb[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cb_%s,%s" %(i,v))

cl = {}
for i in H:
    for v in V:
        cl[i,v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="Cl_%s,%s" %(i,v))
  

#---Objective---
obj = (quicksum(d[i,j]*x[i,j,v] for i in H for j in H for v in V))
m.setObjective(obj, GRB.MINIMIZE)



#======================Continue Here!!!!!!!=============

#---Constraints---  
# All locations should be visited
for i in V:
    m.addConstr(quicksum(x[i,j] for j in V) == 1, name='Visit_%s' % (i))
for j in V:
    m.addConstr(quicksum(x[i,j] for i in V) == 1, name='Visit_%s' % (i))
       
#Objective 


print(H)
## Objective
obj = (quicksum(t[i,j]*x[i,j] for i in V for j in V))
m.setObjective(obj, GRB.MINIMIZE)

# Constraints    
# All locations should be visited
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


