# -*- coding: utf-8 -*-
"""
@author: lucas.moorlag
"""
import gurobipy as gb
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from print_img import transform_point, draw_point, draw_line

""""
!!! Press 'q' to clear image after running !!!
"""

full_output = True
plot_network = True

# Read data to dictionary
file_name = 'GreenLightModelCompressed'

#-------------------------------------------------------- Set Parameters ------------------------------------------
Years = 1
T = range(Years)

P  = 5          #Max USV facilities

V0 = 5          #Max vessels type 0                       
V1 = 5          #Max vessels type small USV               
V2 = 5          #Max vessels type large USV               

V =    range (V0+V1+V2)
VR0 =  range (0, V0)
VR1 =  range (V0, V1+V0)
VR2 =  range (V1+V0,V2+V1+V0)
VR12 = range (V0,V2+V1+V0)

R0C = 75         #Range from centroid
R1  = 120        #Range USV 1 (maximum 1 day of sailing)        
R2  = 192        #Range USV 2 (maximum 2 days of sailing)      

TS0 = 10         #Travel speed Traditional Vessel
TS1 = 5          #Travel speed 12-meter USV
TS2 = 8          #Travel speed 18-meter USV
                    
HC0  = 22000     #Harbor Cost Crew Change
HC12 = 2*10000   #Harbor Cost USVs per year (considering 2 years)

VC0 = 10000      #Vessel cost Traditional Vessel (daily)
VC1 = 5000       #Vessel cost small USV (daily)
VC2 = 6000       #Vessel cost large USV (daily)

AVC0 = 10000     #Vessel cost Traditional Vessel (daily)
AVC1 = 5000      #Vessel cost small USV (daily)
AVC2 = 6000      #Vessel cost large USV (daily)

ST0 = 20         #Service Time at asset for traditional vessel
ST1 = 26         #Service Time at asset for 12-meter USV
ST2 = 22         #Service Time at asset for 18-meter USV

O0 = 2*280       #Operational for traditional vessel (yearly)
O1 = 2*150       #Operational for 12-meter USV (yearly)
O2 = 2*200       #Operational for 18-meter USV (yearly)

RC  = 50000      #Costs for changing a Robodock
RIC = 1000000    #Costs for constructing a Robodock 

M = 1000         #Artbitrarily large number

#-------------------------------------------------------- Start Script ------------------------------------------
#Define subset of hubs and locations as dataframe
#Read data to dictionary
df_per_year = {}
AJ = {}
IJ = {}
activelocations = {}
inactivelocations = {}
activeWindlocations = {}
inactiveWindlocations = {}
activeOGlocations = {}
inactiveOGlocations = {}
harbors={}
harborsu={}
inactiveharbors ={}
D={}
DU={}
DIU={}
for t in T:
    data_dict = {}  
    
    #IJ = {}
    with open(f"{file_name}.txt") as f:
        for line in f:
            data_line = line.strip("\n").split("\t")
            data_line = [int(x) for x in data_line]
            data_dict[data_line[0]] = {}
            data_dict[data_line[0]]['nodenumber'] = int(data_line[0])  
            data_dict[data_line[0]]['x_coord'] = int(data_line[2])
            data_dict[data_line[0]]['y_coord'] = int(data_line[1])
            data_dict[data_line[0]]['node_type'] = int(data_line[3+t])    #deze voor mastersheet 6
            data_dict[data_line[0]]['demand'] = int(data_line[7+t])     #deze voor mastersheet 6
        df_per_year[t] = pd.DataFrame.from_dict(data_dict, orient='index')

        harbors[t] = df_per_year[t][df_per_year[t]['node_type'] <= 1]
        harborst = df_per_year[t][df_per_year[t]['node_type'] == 0]
        harborsu[t] = df_per_year[t][df_per_year[t]['node_type'] == 1]
        centroids= df_per_year[t][df_per_year[t]['node_type'] == 3]
        inactiveharbors[t]= df_per_year[t][df_per_year[t]['node_type'] == 4]
        locations = df_per_year[t][df_per_year[t]['demand'] != 0]
        harborsandcentroids = df_per_year[t][df_per_year[t]['demand'] == 0]
        activelocations[t] = df_per_year[t][df_per_year[t]['demand'] > 1]
        inactivelocations[t] = df_per_year[t][df_per_year[t]['demand'] < 0]
        activeWindlocations[t] = df_per_year[t][df_per_year[t]['demand'] == 750000]
        inactiveWindlocations[t] = df_per_year[t][df_per_year[t]['demand'] == -1]
        activeOGlocations[t] = df_per_year[t][df_per_year[t]['demand'] == 1000000]
        inactiveOGlocations[t] = df_per_year[t][df_per_year[t]['demand'] == -2]
           
        harbors_dict = harbors[t].to_dict()
        harborst_dict = harborst.to_dict()
        harborsu_dict = harborsu[t].to_dict()
        locations_dict = locations.to_dict()
        harborsandcentroids_dict=harborsandcentroids.to_dict()
        inactiveharbors_dict=inactiveharbors[t].to_dict()
        activelocations_dict = activelocations[t].to_dict()
        inactivelocations_dict=inactivelocations[t].to_dict()
        activeWindlocations_dict = activeWindlocations[t].to_dict()
        inactiveWindlocations_dict = inactiveWindlocations[t].to_dict()
        activeOGlocations_dict = activeOGlocations[t].to_dict()
        inactiveOGlocations_dict = inactiveOGlocations[t].to_dict()
        centroids_dict = centroids.to_dict()
        all_nodes = df_per_year[t].to_dict()

        J = locations.index
        H = harborsandcentroids.index
        AJ[t] = activelocations[t].index
        IJ[t] = inactivelocations[t].index
        D[t] = harbors[t].index
        DT = harborst.index
        DU[t] = harborsu[t].index
        DIU[t] = inactiveharbors[t].index
        C = centroids.index
        N = df_per_year[t].index

    # Distance_Calculation
    distance_list = []

    for i in N:
        distance_list.append([])
        for j in N:
            x_difference = all_nodes['x_coord'][j] - all_nodes['x_coord'][i]
            y_difference = all_nodes['y_coord'][j] - all_nodes['y_coord'][i]
            t_travel = (1/2.5)* math.sqrt(x_difference ** 2 + y_difference ** 2)    #1/2.5 makes it nautical miles
            distance_list[i].append(t_travel)    

# print(distance_list[108][75]) #To print distance in knots

#-------------------------------------------------------- Define Gurobi Files------------------------------------------

m = gb.Model('FLP')

#-------------------------------------------------------- Decision Variables------------------------------------------
x = {}
for i in N:
    for j in N:
        for v in V:
            for t in T:
                x[i, j, v, t] = m.addVar(vtype=gb.GRB.BINARY, name=f'x[{i, j, v, t}]')

y = {}
for i in N:
    for t in T:
        y[i, t] = m.addVar(vtype=gb.GRB.BINARY, name=f'y[{i, t}]')

z= {}
for i in N:
    for v in V:
        for t in T:
            z[i, v, t] = m.addVar(vtype=gb.GRB.BINARY, name=f'z[{i, v, t}]')

a={}
for i in N:
    for v in V:
        for t in T:
            a[i,v,t] = m.addVar(vtype=gb.GRB.BINARY, name=f'a[{i,v, t}]')
           
#-------------------------------------------------------- Objective Function------------------------------------------
ob_f  = gb.quicksum(df_per_year[t]['demand'][j] * x[i, j, v, t]  for i in DU[t] for j in J for v in VR1 for t in T)
ob_f += gb.quicksum(df_per_year[t]['demand'][j] * x[i, j, v, t]  for i in DU[t] for j in J for v in VR2 for t in T)
ob_f += gb.quicksum(df_per_year[t]['demand'][j] * x[i, j, v, t]  for i in C for j in J for v in VR0 for t in T)

ob_f -= gb.quicksum(VC2*2*distance_list[i][j]*x[i, j, v, t]/(TS2*24)  for i in DU[t] for j in J for v in VR2 for t in T)
ob_f -= gb.quicksum(VC1*4*distance_list[i][j]*x[i, j, v, t]/(TS1*24)  for i in DU[t] for j in J for v in VR1 for t in T)
ob_f -= gb.quicksum(VC0*2*distance_list[i][j]*x[i, j, v, t]/(TS0*24) for i in C for j in J for v in VR0 for t in T)
ob_f -= gb.quicksum(VC0*2*distance_list[i][j]*x[i, j, v, t]/(TS0*24) for i in DT for j in C for v in VR0 for t in T)

ob_f -= gb.quicksum(2*365 * AVC0 * z[i,v,t]   for i in DT for v in VR0 for t in T)
ob_f -= gb.quicksum(2*365 * AVC1 * z[i,v,t]   for i in DU[t] for v in VR1 for t in T)
ob_f -= gb.quicksum(2*365 * AVC2 * z[i,v,t]   for i in DU[t] for v in VR2 for t in T)

ob_f -= gb.quicksum(HC12* y[i, t] for i in DU[t] for t in T)
ob_f -= gb.quicksum(HC0 * x[i,j,v,t] for i in DT for j in C for v in VR0 for t in T)

ob_f -= gb.quicksum(RC*a[i,v,t]  for i in D[t] for v in VR12 for t in T)

ob_f -= gb.quicksum(RIC * y[i, max(T)] for i in DU[t])


m.setObjective(ob_f, gb.GRB.MAXIMIZE)                   
   
#-------------------------------------------------------- Giving Conditions ------------------------------------------
# # There is 1 12-meter USV in year 0
giv0 = {}
giv0[j,v,t] = m.addConstr(gb.quicksum(z[i,v,0] for i in D[0] for v in VR0) == 1, name=f'giv0[{j,v,t}]')

# # # There are no 18-meter USVs in year 0
# giv1 = {}
# giv1[j,v,t] = m.addConstr(gb.quicksum(z[i,v,0] for i in D[0] for v in VR2) == 0, name=f'giv1[{j,v,t}]')

# # # There are no traditional vessels in year 3
# giv2 = {}
# giv2[j,v,t] = m.addConstr(gb.quicksum(z[i,v,2] for i in D[2] for v in VR0) == 0, name=f'giv2[{j,v,t}]')

# # # There are no harbors at harbor 444 in year 0
# giv3 = {}
# giv3[j,v,t] = m.addConstr(gb.quicksum(z[444,v,0]  for v in VR12) == 0, name=f'giv3[{j,v,t}]')


#-------------------------------------------------------- Constraints ------------------------------------------
# Constraint 1: There is no travel from inactive harbors
con1 = {}
for t in T:
    for i in DIU[t]:
        con1[i,j,v] =  m.addConstr(gb.quicksum(x[i, j, v, t] for j in J for v in V) == 0 , name=f'con1[{i,j,v}]')

# Constraint 2: Each demand location is covered once
con2 = {}
for t in T:
    for j in AJ[t]:
        con2[j] = m.addConstr(  gb.quicksum(x[i, j, v, t] for i in D[t] for v in VR12) 
                              + gb.quicksum(x[i, j, v, t] for i in C for v in VR0) == 1, name=f'con2[{j}]')

# #Constraint 3: Traditional Vessels do not go directly to jobs
con3 ={}
for i in DT:
    for j in J:
        for v in VR0:
            for t in T:
                con3[i,j,v, t] = m.addConstr(x[i, j, v, t] == 0, name=f'con3[{i,j,v}]') 
        

# #Constraint 4: There is no travel in between demand nodes
con4 ={}
for t in T:
    for i in AJ[t]:
        for v in V:
            con4[i,j,v, t] = m.addConstr(gb.quicksum(x[i, j, v, t] for j in AJ[t]) == 0, name=f'con4[{i,j,v}]') 
          

#Constraint 5: USV's do not go directly to centroids
con5 = {}
for t in T: 
    for i in DU[t]:
        for j in C:
            for v in VR12:           
                    con5[i,j,v] = m.addConstr(x[i, j, v, t]  == 0, name=f'con5[{i,j,v}]') 

#Constraint 6: If vessel v goes from centroid to demand node, vessel v has to go from harbor to centroid
con6 = {}
for i in C:
    for v in VR0:
        for t in T: 
            con6[i,v] = m.addConstr(gb.quicksum(x[i, j, v, t] for j in J) <= 
                                    M * gb.quicksum(x[j, i, v, t] for j in DT), name=f'con6[{i}]')


# Constraint 7: Limits number of USV facilities that can be build
con7 = {}
for t in T: 
    con7[j] = m.addConstr(gb.quicksum(y[i, t] for i in DU[t]) <= P , name=f'con7[{j}]')

# Constraint 8: Vessels can only be assigned to located facilities
con8 = {}
for t in T: 
    for v in V:
        for i in D[t]:
                con8[j,v]= m.addConstr(z[i,v,t] <= y[i,t] , name=f'con8[{j}]')

# Constraint 88: Geen facilities als er geen schepen zijn
con88 = {}
for t in T: 
        for i in D[t]:
                con88[j,v]= m.addConstr(gb.quicksum(z[i,v,t] for v in V) - y[i,t] >= 0 
                                       , name=f'con88[{j}]')

# # Constraint 9: If vessel goes from location D to J, a facility has te be there 
con9 = {}
for t in T: 
    for i in D[t]:
        for v in V:
                con9 =  m.addConstr(gb.quicksum(x[i, j, v, t] for j in J) <= M * z[i,v,t], name=f'con9[{i,v}]')

# Constraint 11: Traditional Vessels can only be assigned to one Traditional Vessel Facility
con11 = {}
for v in VR0:
    for t in T: 
        con11[v] = m.addConstr(gb.quicksum(z[i,v, t] for i in DT) <= 1, name=f'con11[{v}]')

# Constraint 12: Traditional Vessels cannot be assigned to USV harbor
con12 = {}
for v in VR0:
    for t in T: 
        con12[v] = m.addConstr(gb.quicksum(z[i,v,t] for i in DU[t]) == 0, name=f'con12[{v}]')

# Constraint 13: USVs can only be assigned to one facility
con13 = {}
for v in VR12:
    for t in T: 
        con13[v] = m.addConstr(gb.quicksum(z[i,v,t] for i in DU[t]) <= 1, name=f'con13[{v}]')

# Constraint 14: USVs can not be assigned to Traditional Vessel Harbor
con14 = {}
for v in VR12:
    for t in T: 
        con14[v] = m.addConstr(gb.quicksum(z[i,v,t] for i in DT) == 0, name=f'con14[{v}]')

# # Constraint 15: Enforces range constraint on USV1
con15 = {}
for t in T: 
    for i in DU[t]:
        for j in J:
            for v in VR1:       
                    con15[i,j,v] = m.addConstr(distance_list[i][j]*x[i,j,v,t] <=  R1*z[i,v,t], name=f'con15[{i,j,v}]')

# # Constraint 16: Enforces range constraint on USV2 
con16 = {}
for t in T: 
    for i in DU[t]:
        for j in J:
            for v in VR2:       
                    con16[i,j,v] = m.addConstr(distance_list[i][j]*x[i,j,v,t] <=  R2*z[i,v,t], name=f'con16[{i,j,v}]')

# Constraint 17: If there is travel between a harbor and a centroid, there needs to be a harbor location             
con17 = {}
for i in DT:
    for j in C:
        for v in VR0:   
            for t in T:
                con17[i,j,v] = m.addConstr(x[i,j,v,t] <=  M*z[i,v,t], name=f'con17[{i,j,v}]')
                
# Constraint 18: Enforces range constraint on Traditional vessels between centroids and demand nodes
con18 = {}
for i in C:
    for j in J:
        for v in VR0:   
            for t in T: 
                con18[i,j,v,t] = m.addConstr(distance_list[i][j]*x[i,j,v,t] <=  R0C*z[i,v,t], name=f'con18[{i,j,v,t}]')
    
# Constraint 19: Forces collected demand to be less or equal to the operational limit of a vessel
con19 = {}
for v in VR0:
    for t in T: 
        con19[v] = m.addConstr(gb.quicksum(ST0*x[i,j,v,t] for i in C for j in J) 
                               + gb.quicksum(2*distance_list[i][j]*x[i,j,v,t] for i in C for j in J)  /    (TS0*24) 
                               + gb.quicksum(2*distance_list[i][j]*x[i,j,v,t] for i in D[t] for j in C)  / (TS0*24) <= O0,
                      name=f'con19[{v}]')

# Constraint 20: Forces collected demand to be less or equal to the operational limit of a vessel
con20 = {}
for v in VR1:
    for t in T: 
        con20[v] = m.addConstr(gb.quicksum(ST1*x[i, j, v, t] for i in DU[t] for j in J) 
                               + gb.quicksum(4*distance_list[i][j]*x[i,j,v, t] for i in DU[t] for j in J)  /  (TS1*24) <= O1,
                      name=f'con20[{v}]')    

# Constraint 21: Forces collected demand to be less or equal to the operational limit of a vessel
con21 = {}
for v in VR2:
    for t in T: 
        con21[v] = m.addConstr(gb.quicksum(ST2*x[i, j, v, t] for i in DU[t] for j in J) 
                               + gb.quicksum(2*distance_list[i][j]*x[i,j,v, t] for i in DU[t] for j in J)  /  (TS2*24) <= O2,
                      name=f'con21[{v}]')    


# Constraint 22: Ensures there will be the same amount of 12-meter USVs in subsequential time indices
con22 = {}
for t in T:
    for v in VR1:
        if t > 0:
            con22[i,t] = m.addConstr(gb.quicksum(z[i,v,t-1] for i in DU[t]) 
                                                  <= gb.quicksum(z[i,v,t]for i in DU[t]), name=f'con22[{i,v,t}]')

# Constraint 22: Ensures there will be the same amount of 18-meter USVs in subsequential time indices
con23 = {}
for t in T:
    for v in VR2: 
        if t > 0:
                con23[i,t] = m.addConstr(gb.quicksum(z[i,v,t-1] for i in DU[t]) 
                                                  <= gb.quicksum(z[i,v,t]for i in DU[t]), name=f'con23[{i,v,t}]')

# #Constraint 23: Enforces fine when robodock moves not throughout time indices
con24 = {}
for t in T:
    for v in VR12:
        for i in DU[t]:
            if t > 0:
                con24[i,t] = m.addConstr(z[i,v,t-1] <= z[i,v,t] + a[i,v,t], name=f'con24[{i,v,t}]')
                
# #Constraint 23: Limits total inspection jobs that can be visited per vehicle per centroid
con100 = {}
for j in C:
    for v in VR0:
        for t in T:
            con100[j,v] = m.addConstr(gb.quicksum(x[j, i, v, t] for i in J) <= 4, name=f'con100[{j,v}]')  
            
#-------------------------------------------------------- Output Variables ------------------------------------------
m.update()
print('')
print("Start optimizing:")
print("Set Time limit")
m.setParam('Presparsify', 1)
m.setParam('Presolve', 2)
m.setParam('MIPGap', 0.025)
m.setParam('TimeLimit', 600)
print('')
#m.setParam('LogFile', 'output/output.log')
m.optimize()

m.write("LP3.lp")

found_solution = False 

if m.status == gb.GRB.Status.OPTIMAL:
    found_solution = True
    
    print('Finished')
    print('')
    print(f"Optimal Value found is: {m.ObjVal}")
    print('Results:')

    # #Print arcs
    # for t in T:
    #     for i in H:
    #         for j in N:
    #             for v in V:
    #                 if x[i, j, v, t].X == 1:
    #                     print(x[i,j,v,t])
    
    #Print harbors used
    print('')
    for t in T:
        for i in D[t]:
            if y[i,t].X == 1:
                print (y[i,t])

    #Print vessels at harbors
    print('')
    for t in T:
        for i in D[t]:
            for v in V: 
                    if z[i, v, t].X == 1:
                        print (z[i,v,t])
                        
    #Print harbors used
    print('')
    for t in T:
        for i in DU[t]:
            if y[i,t].X == 1:
                print (y[i,t])
    
    
    #Print when robodock change occurs
    print('')
    for t in T:
        for i in D[t]:
            for v in V: 
                if a[i,v,t].X == 1:
                    print (a[i,v,t])
     
    TOTALV0 = []  
    TOTALV1 = []  
    TOTALV2 = []  
    for t in T:
        V0count = 0
        V1count = 0
        V2count = 0
        TOTALV0.append([])
        TOTALV1.append([])
        TOTALV2.append([])
        for v in VR0:
            for i in DT:
                if z[i,v,t].X == 1:
                    V0count += 1
                TOTALV0[t] = V0count
        for v in VR1:
            for i in DU[t]:
                if z[i,v,t].X == 1:
                    V1count += 1
                TOTALV1[t] = V1count
        for v in VR2:
            for i in DU[t]:
                if z[i,v,t].X == 1:
                    V2count += 1
                TOTALV2[t] = V2count
    print('') 
    print('Number of VR0s in year t', TOTALV0)
    print('Number of 12m-USVs in year t', TOTALV1)
    print('Number of 18m-USVs in year t', TOTALV2)            
    
    NODESV0 = []  
    NODESV1 = []  
    NODESV2 = []  
    for t in T:
        nodesV0count = 0
        nodesV1count = 0
        nodesV2count = 0
        NODESV0.append([])
        NODESV1.append([])
        NODESV2.append([])
        for v in VR0:
            for i in C: 
                for j in J:
                    if x[i,j,v,t].X == 1:
                        nodesV0count += 1
                    NODESV0[t] = nodesV0count
        for v in VR1:
            for i in DU[t]: 
                for j in J:
                    if x[i,j,v,t].X == 1:
                        nodesV1count += 1
                    NODESV1[t] = nodesV1count
        for v in VR2:
            for i in DU[t]: 
                for j in J:
                    if x[i,j,v,t].X == 1:
                        nodesV2count += 1
                    NODESV2[t] = nodesV2count
    print('')                     
    print('Nodes visited by VR0s in year t', NODESV0)
    print('Nodes visited by 12m-USVs in year t', NODESV1)
    print('Nodes visited by 18m-USVs in year t', NODESV2)    
    
    DISTEV0 = 0
    EMV01 = []
    EMV02 = []
    EMV1 = []
    EMV2 = []
    for t in T:
        EMV0 = []
        EMV0Count= 0
        EMV0Count1= 0
        EMV0Count2= 0
        EMV1Count= 0
        EMV2Count= 0
        EMV0.append([])
        EMV01.append([])
        EMV02.append([])
        EMV1.append([])
        EMV2.append([])
        for v in VR0:
            EV0_vt = 0
            for j in C: 
                for i in H:
                    if x[i,j,v,t].X == 1:
                        EV0_vt += 2*distance_list[i][j]
                        DISTEV0 += 2*distance_list[i][j]
                        EMV0Count1 += 0.001*(2*distance_list[i][j]/(TS0*24))*6000
                    EMV01[t]=round(EMV0Count1)  
                    
                for i in J:
                    if x[j,i,v,t].X == 1:
                        EV0_vt += 2*distance_list[i][j]                                                 
                        DISTEV0+= 2*distance_list[i][j]
                        EMV0Count2 += 0.001*(2*distance_list[i][j]/(TS0*24))*6000
                        EMV0Count2 += 0.001*ST0*6000
                    EMV02[t]=round(EMV0Count2)  
                    
        for v in VR1:    
            for i in H:
                for j in J:
                    if x[i,j,v,t].X == 1:
                        EMV1Count += 0.001*(2*distance_list[i][j]/(TS0*24))*300  
                        EMV1Count += 0.001*ST1*300   
                    EMV1[t]=round(EMV1Count)
                    
        for v in VR2:    
            for i in H:
                for j in J:
                    if x[i,j,v,t].X == 1:
                        EMV2Count += 0.001*(2*distance_list[i][j]/(TS0*24))*325  
                        EMV2Count += 0.001* ST2*325
                    EMV2[t]=round(EMV2Count)
    
    print('')                     
    print('Emissions by VR0s transit in year t', EMV01)
    print('Emissions by VR0s operations in year t', EMV02)
    print('Emissions by 12m-USVs in year t', EMV1)
    print('Emissions by 18m-USVs in year t', EMV2)           
        
    print('')
    print(f"Optimal Value found is: {round(m.ObjVal,2)}")
    print('')
    print('Draw Graph')
#-------------------------------------------------------- Draw Graph ------------------------------------------
    #Plot Figure
    plt.figure(figsize=(20,15))
    
    for t in T:
            plt.scatter(harborst['x_coord'][:], harborst['y_coord'][:], c='darkblue')  # Location of depot 2
            plt.scatter(harborsu[t]['x_coord'][:], harborsu[t]['y_coord'][:], c='blueviolet')  # Location of depot 2
            plt.scatter(activeWindlocations[t]['x_coord'][:], activeWindlocations[t]['y_coord'][:], c='green')  # ActiveLocation of clients
            plt.scatter(inactiveWindlocations[t]['x_coord'][:], inactiveWindlocations[t]['y_coord'][:], c='gold')  # InactiveLocation of clients
            plt.scatter(activeOGlocations[t]['x_coord'][:], activeOGlocations[t]['y_coord'][:], c='red')  # ActiveLocation of clients
            plt.scatter(inactiveOGlocations[t]['x_coord'][:], inactiveOGlocations[t]['y_coord'][:], c='orange')  # InactiveLocation of clients
            plt.scatter(centroids['x_coord'][:], centroids['y_coord'][:], c='gray')  # Location of artificial       
    
    # Plot Arcs
    coloursV0 = ['red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                 'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                 'brown', 'sienna', 'salmon', 'coral' , 'darkorange']
    coloursV1 = ['cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                 'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                 'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                 'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                 'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                 'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                 'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                 'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                 'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                 'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                 'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                 'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',]
    
    coloursV2 = ['forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                 'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',]

                    
    for t in T:
            for idx, row in df_per_year[t].iterrows():
            #    label = f"{idx};"
                label = f"{row['nodenumber']};"  
                plt.annotate(label,  # this is the text
                             (row['x_coord'], row['y_coord']),  # these are the coordinates to position the label
                             textcoords="offset points",  # how to position the text
                             xytext=(-10, 8),  # distance from text to points (x,y)
                             ha='center')
    
            for v in VR2:
                current_colour = coloursV2[v]
                for i in DU[t]:
                    for j in J:
                            if x[i, j, v, t].X == 1.0:
                                plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
                                
            for v in VR1:
                current_colour = coloursV1[v]
                for i in DU[t]:
                    for j in J:
                            if x[i, j, v, t].X == 1.0:
                                plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
                        
            for v in VR0:
                current_colour = coloursV0[v]
                for i in DT:
                    for j in C:
                            if x[i, j, v, t].X == 1.0:
                                plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
                        
            for v in VR0:
                current_colour = coloursV0[v]
                for i in C:
                    for j in J:
                            if x[i, j, v, t].X == 1.0:
                               plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
        
    plt.show
       
    #Plot Figure for T's
    for t in T:  
            plt.figure(figsize=(20,15))
        
            plt.scatter(harborst['x_coord'][:], harborst['y_coord'][:], c='darkblue')  # Location of depot 2
            plt.scatter(harborsu[t]['x_coord'][:], harborsu[t]['y_coord'][:], c='blueviolet')  # Location of depot 2
            plt.scatter(activeWindlocations[t]['x_coord'][:], activeWindlocations[t]['y_coord'][:], c='green')  # ActiveLocation of clients
            plt.scatter(inactiveWindlocations[t]['x_coord'][:], inactiveWindlocations[t]['y_coord'][:], c='gold')  # InactiveLocation of clients
            plt.scatter(activeOGlocations[t]['x_coord'][:], activeOGlocations[t]['y_coord'][:], c='red')  # ActiveLocation of clients
            plt.scatter(inactiveOGlocations[t]['x_coord'][:], inactiveOGlocations[t]['y_coord'][:], c='orange')  # InactiveLocation of clients
            plt.scatter(centroids['x_coord'][:], centroids['y_coord'][:], c='gray')  # Location of artificial
            
            # Plot Arcs
            coloursV0 = ['red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange',
                         'red', 'orange', 'darkred' , 'gold', 'tomato', 'hotpink', 'maroon' , 'firebrick' , 
                         'brown', 'sienna', 'salmon', 'coral' , 'darkorange']
            coloursV1 = ['cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                         'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                         'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                         'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                         'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                         'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                         'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                         'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                         'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                         'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',
                         'cyan', 'blue', 'darkblue' , 'blueviolet' , 'aqua' , 'darkturquoise' , 'cadetblue' , 'steelblue' ,
                         'mediumblue', 'lightsteelblue', 'cornflowerblue' , 'royalblue','indigo', 'navy',]
            coloursV2 = ['forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',
                         'forestgreen', 'limegreen', 'darkgreen' , 'green', 'lime', 'seagreen',]
            
            for idx, row in df_per_year[t].iterrows():
                #    label = f"{idx};"
                    label = f"{row['nodenumber']};"  
                    plt.annotate(label,  # this is the text
                                 (row['x_coord'], row['y_coord']),  # these are the coordinates to position the label
                                 textcoords="offset points",  # how to position the text
                                 xytext=(-10, 8),  # distance from text to points (x,y)
                                 ha='center')
            for v in VR2:
                    current_colour = coloursV2[v]
                    for i in DU[t]:
                        for j in J:
                                if x[i, j, v, t].X == 1.0:
                                    plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
             
            for v in VR1:
                    current_colour = coloursV1[v]
                    for i in DU[t]:
                        for j in J:
                                if x[i, j, v, t].X == 1.0:
                                    plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
             
                                
            for v in VR0:
                    current_colour = coloursV0[v]
                    for i in DT:
                        for j in C:
                                if x[i, j, v, t].X == 1.0:
                                    plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
             
                                
            for v in VR0:
                    current_colour = coloursV0[v]
                    for i in C:
                        for j in J:
                                if x[i, j, v, t].X == 1.0:
                                    plt.plot([df_per_year[t]['x_coord'][i], df_per_year[t]['x_coord'][j]], [df_per_year[t]['y_coord'][i], df_per_year[t]['y_coord'][j]], c=current_colour)
             
               
            plt.show() 
    
    print('')   
    print('START DRAWING IMG')
    # DRAW ON IMAGE
    img_path = {}
    for t in T:
        img_path[t] = './Thesis_North_Sea.png'
        img = cv2.imread(img_path[t])
        while True:
        
               for i, harbor in harborst.iterrows():
                   lat = harbor['y_coord']
                   lon = harbor['x_coord']
                   point = {
                       'lat': lat,
                       'lon': lon,
                   }
                   draw_point(img, point, color=(0, 0, 255))
                  
               for i, harbor in harborsu[t].iterrows():
                   lat = harbor['y_coord']
                   lon = harbor['x_coord']
                   point = {
                       'lat': lat,
                       'lon': lon,
                   }
                   draw_point(img, point, color=(0, 128, 0))
               
               for i, harbor in activelocations[t].iterrows():
                   lat = harbor['y_coord']
                   lon = harbor['x_coord']
                   point = {
                       'lat': lat,
                       'lon': lon,
                   }
                   draw_point(img, point, color=(255, 0, 0))
                         
               for i, harbor in centroids.iterrows():
                   lat = harbor['y_coord']
                   lon = harbor['x_coord']
                   point = {
                       'lat': lat,
                       'lon': lon,
                   }
                   draw_point(img, point, color=(0, 0, 0))
                
               
                # Plot Arcs
               for v in VR2:
                   current_colour = coloursV2[v]
                   for i in DU[t]:
                       for j in J:
                           # for t in T:
                               if x[i, j, v, t].X == 1.0:
                                   p1 = {
                                       'lat': df_per_year[t]['y_coord'][i],
                                       'lon': df_per_year[t]['x_coord'][i],
                                                              }
                                   p2 = {
                                       'lat': df_per_year[t]['y_coord'][j],
                                       'lon': df_per_year[t]['x_coord'][j],
                                   }
                                   draw_line(img, p1, p2, color=(0, 128, 0))
                
               for v in VR1:
                   current_colour = coloursV1[v]
                   for i in DU[t]:
                       for j in J:
                           # for t in T:
                               if x[i, j, v, t].X == 1.0:
                                   p1 = {
                                       'lat': df_per_year[t]['y_coord'][i],
                                       'lon': df_per_year[t]['x_coord'][i],
                                                              }
                                   p2 = {
                                       'lat': df_per_year[t]['y_coord'][j],
                                       'lon': df_per_year[t]['x_coord'][j],
                                   }
                                   draw_line(img, p1, p2, color=(0, 0, 128))
                   
               for v in V:
                    current_colour = coloursV0[v]
                    for i in DT:
                        for j in C:
                            # for t in T:
                                if x[i, j, v, t].X == 1.0:
                                    p1 = {
                                        'lat': df_per_year[t]['y_coord'][i],
                                        'lon': df_per_year[t]['x_coord'][i],
                                                               }
                                    p2 = {
                                        'lat': df_per_year[t]['y_coord'][j],
                                        'lon': df_per_year[t]['x_coord'][j],
                                    }
                                    draw_line(img, p1, p2, color=(0, 0, 255)) 
                                   
               for v in V:
                   current_colour = coloursV0[v]
                   for i in C:
                       for j in J:
                           # for t in T:
                               if x[i, j, v, t].X == 1.0:
                                   p1 = {
                                       'lat': df_per_year[t]['y_coord'][i],
                                       'lon': df_per_year[t]['x_coord'][i],
                                                              }
                                   p2 = {
                                       'lat': df_per_year[t]['y_coord'][j],
                                       'lon': df_per_year[t]['x_coord'][j],
                                   }
                                   draw_line(img, p1, p2, color=(0, 0, 255))                 
        
               # for t in T:
               cv2.imshow('0123'[t], img)
               res = cv2.waitKey()
               # CLOSE BY PRESSING 'q'
               if res == ord('q'):
                   cv2.destroyAllWindows()
                   break    
    
             
        else:
            print('')
            print('Finished')
            print('') 
            print("No Feasible Solution found with parameters:")
    

#-------------------------------------------------------- Get Results ------------------------------------------
# ## Output Results as dataframe

#All location results
# Define all location results dictionary
results = {}
    # Loop over locations and depots
if m.status == gb.GRB.Status.OPTIMAL:
    found_solution = True
    for i in J:
            visited_vehicle_list = []
            visited_from_list = []
            distance_to_hub_list =[]
            distance_to_centroid_list =[]
            servedUSV = []
            servedT = []
            distance_harbor_to_centroid_list = []
            for j in H:
                for v in V:
                    for t in T:
                        if x[j, i, v, t].X == 1:
                            visited_from_list.append(j)
                            visited_vehicle_list.append(v)
                            distance_to_hub_list = distance_list[j][i]
                            servedUSV.append(1)
                            
                            # Add results to dictionary
                            results[i] = {}
                            results[i]['customer number'] = i
                            results[i]['demand'] = df_per_year[t]['demand'][i]
                            # results[i]['operation time'] = df['time'][i]
                            results[i]['visited_from_z'] = visited_from_list
                            results[i]['visited_vehicle_v'] = visited_vehicle_list
                            results[i]['served by USV'] = servedUSV
                            results[i]['distance to node'] = distance_to_hub_list  
            for j in C:
                  for v in V:
                      for t in T:
                          if x[j, i, v, t].X == 1:
                              visited_from_list.append(j)
                              visited_vehicle_list.append(v)
                              distance_to_centroid_list = distance_list[j][i]
                              servedT.append(1)
                    
                              # Add results to dictionary
                              results[i] = {}
                              results[i]['customer number'] = i
                              results[i]['demand'] = df_per_year[t]['demand'][i]
                              # results[i]['operation time'] = df['time'][i]
                              results[i]['visited_from_z'] = visited_from_list
                              results[i]['visited_vehicle_v'] = visited_vehicle_list
                              results[i]['served by Traditional Vessel'] = servedT
                              results[i]['distance node to centroid'] = distance_to_centroid_list
                       
        # Define dataframe with results
    df_results = pd.DataFrame.from_dict(results, orient='index')
    
        # Served location results
        # DOET NU NOG NIKS
    served_results = {}
    df_served_assets_per_year = {}
    # Loop over locations and depots    
    for v in V:
            for i in J:
                for t in T:
                    visited_vehicle_list = []
                    visited_from_list = []
                    distance_to_hub_list =[]
                    profit_at_node_list = []
                    visited_in_year = []        
                    for j in H:
                        if x[j, i, v, t].X == 1:
                            visited_from_list.append(j)
                            visited_vehicle_list.append(v)
                            # visited_in_year.append(t)
                            distance_to_hub_list = distance_list[j][i]
                            profit_at_node_list = df_per_year[t]['demand'][i] - distance_list[i][j]                    
                            # Add results to dictionary
                            served_results[i] = {}
                            served_results[i]['customer number'] = i
                            served_results[i]['demand'] = df_per_year[t]['demand'][i]
                            # served_results[i]['operation time'] = df['time'][i]
                            served_results[i]['visited_from'] = visited_from_list
                            served_results[i]['visited_vehicle'] = visited_vehicle_list
                            served_results[i]['visited_in_year'] = visited_in_year
                            served_results[i]['distance'] = distance_to_hub_list
                            served_results[i]['profit at node'] = profit_at_node_list
    
    # df_served_assets_per_year[t] = pd.DataFrame.from_dict(served_results, orient='index')
    
    # Define dataframe with results
    df_served_results_final_year = pd.DataFrame.from_dict(served_results, orient='index')