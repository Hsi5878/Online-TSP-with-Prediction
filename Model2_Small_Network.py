 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 23:23:47 2022

@author:
"""

import random
import pandas as pd
import itertools
import time
import numpy as np
from operator import itemgetter, attrgetter
from sympy import *
import matplotlib.pyplot as plt
import statistics

#Help to find the distance that server has moved from the previous reequest when t_back occur  
def Move(o,a,b,remain):
    if (a == [0,0,0]):
        x = symbols('x')
        x_vec = x * b[0]/((b[0] ** 2 + b[1]** 2) ** (1/2))
        y_vec = x * b[1]/((b[0] ** 2 + b[1] ** 2) ** (1/2))
        return remain/2
    else:
        x = symbols('x')
        x_vec = a[0] +  x * (b[0]-a[0])/(((b[0]-a[0]) ** 2 + (b[1]-a[1]) ** 2) ** (1/2))
        y_vec = a[1] +  x * (b[1]-a[1])/(((b[0]-a[0]) ** 2 + (b[1]-a[1]) ** 2) ** (1/2))
        move = solve(((x_vec - a[0]) ** 2 + (y_vec - a[1]) ** 2) ** (1/2) + (x_vec ** 2 + y_vec **2) ** (1/2) - remain  , x)
        if (len(move)== 0):
            return -1
        else:
            return max(move[i] for i in range(len(move)))

#The time that server finish the last request of the route    
def last_serve(table,route,start_time):
    distance = 0.0
    for i in range(len(route)):
        if (i == 0):
            distance += (table[route[0]][0] ** 2 + table[route[0]][1] ** 2) ** (1/2)
        else :
            distance += ((table[route[i]][0] - table[route[i-1]][0]) ** (2) + (table[route[i]][1] - table[route[i-1]][1]) ** (2)) ** (1/2)
    return start_time + distance
    

# Brute Force Solver
def brutal_force_opt(table,index_to_serve,start_time):
    perm = list(itertools.permutations(index_to_serve))#list all permutations
    optimal_order = []
    optimal_length = 10000000
    for route_num in range(len(perm)):
        t = 0
        if((table[perm[route_num][0]][0] ** 2 + table[perm[route_num][0]][1] ** 2) ** (1/2) > (table[perm[route_num][0]][2]-start_time)):
            t = (table[perm[route_num][0]][0] ** 2 + table[perm[route_num][0]][1] ** 2) ** (1/2)
        else:
            t = table[perm[route_num][0]][2] - start_time
            
        for i in range(1, len(index_to_serve)):
            temp = ((table[perm[route_num][i]][0] - table[perm[route_num][i-1]][0]) ** 2 + 
                    (table[perm[route_num][i]][1] - table[perm[route_num][i-1]][1]) ** 2) ** (1/2)
            if(t+temp > table[perm[route_num][i]][2]-start_time):
                t += temp
            else:
                t = table[perm[route_num][i]][2]-start_time               
        t += (table[perm[route_num][-1]][0] ** 2 + table[perm[route_num][-1]][1] ** 2) ** (1/2)
        if t < optimal_length:
            optimal_order = perm[route_num]
            optimal_length =t
        
    return optimal_order, optimal_length

#Learning-Augmented Routing With Last Arrival Time (LAR-LAST)
def Last_opt(table_sort,n,t_hat):
    table = table_sort
    serve = []
    unserve = []
    arrival_time = []
    for i in range(n):
        unserve.append(i)
    t = 0.0
    i = 0 # count the number of requests which have been served 
    first = 0 #whether the first request has been served or not
    while (i < n):
        route = []
        route_requests = []
        if (i == 0 and first == 0):
            t = table[0][2]
            first = 1
        for j in range(len(unserve)):
            if (table[unserve[j]][2] <= t):
                route.append(table[unserve[j]])
                route_requests.append(unserve[j]) 
        if (len(route_requests) == 0):
            t = table[unserve[0]][2]
            for j in range(len(unserve)):
                if (table[unserve[j]][2] <= t):
                    route.append(table[unserve[j]])
                    route_requests.append(unserve[j])    
        route_order, route_length = brutal_force_opt(table, route_requests,t)# compute the optimal offline route that the requests are presented brfore t      
        temp_array = unserve[:]
        for j in range(len(route_order)):
            temp_array.remove(route_order[j])           
        temp = t + route_length 
        # find the next presented time of new request t_next
        if (len(temp_array) != 0):
            t_next = min(table[temp_array[k]][2] for k in range(len(temp_array)))
        else:
            t_next = 1000000
        #find the time that server finish the last request of the route 
        lastserve = last_serve(table, route_order, t) 
        
        # Start computing
        if ( t<t_hat and temp > t_hat):# Our Model 2's gadget 
            stop_back = t_hat - t #the distance that the server should get baxk to origin
            dis = 0.0
            temp_arrival_time = arrival_time[:]
            temp_serve = serve[:]
            temp_unserve = unserve[:]
            temp_count_i = 0 #count the requests that can be served in this route 
            for k in range(len(route_order)):
                if(k == 0):
                    dis += (table[route_order[0]][0] ** 2 +  table[route_order[0]][1] ** 2 ) ** (1/2)
                else :
                    dis += ((table[route_order[k]][0] - table[route_order[k-1]][0]) ** 2 +  (table[route_order[k]][1] - table[route_order[k-1]][1]) ** 2 ) ** (1/2)
                dis += (table[route_order[k]][0] ** 2 +  table[route_order[k]][1] ** 2 ) ** (1/2)
                if (dis <= stop_back):
                    dis -= (table[route_order[k]][0] ** 2 +  table[route_order[k]][1] ** 2 ) ** (1/2)
                    temp_count_i += 1
                    temp_arrival_time.append(t + dis)
                    temp_serve.append(route_order[k])
                    temp_unserve.remove(route_order[k])
                else: 
                    # this block is to find the time t_back
                    dis -= (table[route_order[k]][0] ** 2 +  table[route_order[k]][1] ** 2 ) ** (1/2)
                    if(k == 0):
                        remain = stop_back
                        move = Move([0,0,0],[0,0,0],table[k],remain)
                        t_back = t + move
                    else: 
                        remain = stop_back - dis + ((table[route_order[k]][0] - table[route_order[k-1]][0]) ** 2 +  (table[route_order[k]][1] - table[route_order[k-1]][1]) ** 2 ) ** (1/2)
                        move = Move([0,0,0],table[k-1],table[k],remain)
                        if (move == -1):
                            if(k-2 <0):
                                remain += ((table[route_order[k-1]][0]) ** 2 +  (table[route_order[k-1]][1] ** 2 )) ** (1/2)
                                Move([0,0,0],[0,0,0],table[k-1],remain)
                            else :
                                remain += ((table[route_order[k-1]][0] - table[route_order[k-2]][0]) ** 2 +  (table[route_order[k-1]][1] - table[route_order[k-2]][1]) ** 2 ) ** (1/2)
                                move = Move([0,0,0],table[k-2],table[k-1],remain)
                        t_back = temp_arrival_time[-1] + move
                    break
            if (t_next < t_back): #follow ALG Redesign
                stop = t_next - t
                distance = 0.0
                for k in range(len(route_order)):
                    if(k == 0):
                        distance += (table[route_order[0]][0] ** 2 +  table[route_order[0]][1] ** 2 ) ** (1/2)
                    else :
                        distance += ((table[route_order[k]][0] - table[route_order[k-1]][0]) ** 2 +  (table[route_order[k]][1] - table[route_order[k-1]][1]) ** 2 ) ** (1/2)
                    if (distance <= stop):
                        i += 1
                        arrival_time.append(t + distance)
                        #update the served and the unserved array
                        serve.append(route_order[k])
                        unserve.remove(route_order[k])
                    else:
                        #find the position of the server whem t_next occur
                        if (k == 0):
                            x = 0 + stop * (table[route_order[0]][0])/(((table[route_order[0]][0]) ** 2 + table[route_order[0]][1] ** 2) ** (1/2))
                            y = 0 + stop * (table[route_order[0]][1])/(((table[route_order[0]][0]) ** 2 + table[route_order[0]][1] ** 2) ** (1/2)) 
                        else :
                            stop += ((table[route_order[k]][0] - table[route_order[k-1]][0]) ** 2 +  (table[route_order[k]][1] - table[route_order[k-1]][1]) ** 2 ) ** (1/2)
                            x = table[route_order[k-1]][0] + stop * ((table[route_order[k]][0]-table[route_order[k-1]][0])/
                                                    ((table[route_order[k]][0]-table[route_order[k-1]][0]) ** 2 +
                                                    (table[route_order[k]][1]-table[route_order[k-1]][1]) ** 2) ** (1/2))
                            y = table[route_order[k-1]][1] + stop * ((table[route_order[k]][1]-table[route_order[k-1]][1])/
                                                    ((table[route_order[k]][0]-table[route_order[k-1]][0]) ** 2 +
                                                    (table[route_order[k]][1]-table[route_order[k-1]][1]) ** 2) ** (1/2))
                        break
                far = (x**2 + y ** 2 ) **(1/2)
                t = t_next + far
            else : # follow our Model2's gadget
                i += temp_count_i
                arrival_time = temp_arrival_time[:]
                #update the served and the unserved array
                serve = temp_serve[:]
                unserve = temp_unserve[:]
                t = t_hat 
        elif (lastserve > t_next):
            stop = t_next - t #the distance that the server can move in this route
            distance = 0.0
            for k in range(len(route_order)):
                if(k == 0):
                     distance += (table[route_order[0]][0] ** 2 +  table[route_order[0]][1] ** 2 ) ** (1/2)
                else :
                    distance += ((table[route_order[k]][0] - table[route_order[k-1]][0]) ** 2 +  (table[route_order[k]][1] - table[route_order[k-1]][1]) ** 2 ) ** (1/2)
                if (distance <= stop):
                    i += 1
                    arrival_time.append(t + distance)
                    #update the served and the unserved array
                    serve.append(route_order[k])
                    unserve.remove(route_order[k])
                else:
                    #find the position of the server whem t_next occur
                    if (k == 0):
                        x = 0 + stop * (table[route_order[0]][0])/(((table[route_order[0]][0]) ** 2 + table[route_order[0]][1] ** 2) ** (1/2))
                        y = 0 + stop * (table[route_order[0]][1])/(((table[route_order[0]][0]) ** 2 + table[route_order[0]][1] ** 2) ** (1/2)) 
                    else :
                        stop += ((table[route_order[k]][0] - table[route_order[k-1]][0]) ** 2 +  (table[route_order[k]][1] - table[route_order[k-1]][1]) ** 2 ) ** (1/2)
                        x = table[route_order[k-1]][0] + stop * ((table[route_order[k]][0]-table[route_order[k-1]][0])/
                                                    ((table[route_order[k]][0]-table[route_order[k-1]][0]) ** 2 +
                                                    (table[route_order[k]][1]-table[route_order[k-1]][1]) ** 2) ** (1/2))
                        y = table[route_order[k-1]][1] + stop * ((table[route_order[k]][1]-table[route_order[k-1]][1])/
                                                    ((table[route_order[k]][0]-table[route_order[k-1]][0]) ** 2 +
                                                    (table[route_order[k]][1]-table[route_order[k-1]][1]) ** 2) ** (1/2))
                    break
            far = (x**2 + y ** 2 ) **(1/2)
            t = t_next + far 
        else :# can complete all this route without any disturbance
            i += len(route)
            distance = 0.0
            for k in range(len(route_order)):
                if(k == 0):
                    distance += (table[route_order[0]][0] ** 2 +  table[route_order[0]][1] ** 2 ) ** (1/2)
                else :
                    distance += ((table[route_order[k]][0] - table[route_order[k-1]][0]) ** 2 +  (table[route_order[k]][1] - table[route_order[k-1]][1]) ** 2 ) ** (1/2)
                arrival_time.append(t + distance)
                #update the served and the unserved array
                serve.append(route_order[k])
                unserve.remove(route_order[k])
            t = temp        
    return t  

#Algorithm Smartstart
def Smartstart(table_sort,n,theta):
    table = table_sort
    serve = []
    unserve = []
    for i in range(n):
        unserve.append(i)
    t = 0.0 
    count = 0 # count the number of requests which have been served 
    first = 0 #whether the first request has been served or not
    while(count < n):
        route = []
        route_requests = []
        if (count == 0 and first == 0):
            t = table[0][2]
            first = 1
        for j in range(len(unserve)):
            if (table[unserve[j]][2] <= t):
                route.append(table[unserve[j]])
                route_requests.append(unserve[j])
        if (len(route_requests) == 0):
            t = table[unserve[0]][2]
            for j in range(len(unserve)):
                if (table[unserve[j]][2] <= t):
                    route.append(table[unserve[j]])
                    route_requests.append(unserve[j])
        route_order, route_length = brutal_force_opt(table, route_requests,t)  # compute the optimal offline route that the requests are presented brfore t        
        temp_array = unserve[:]
        for i in range(len(route_order)):
            temp_array.remove(route_order[i])
        while(t < route_length/(theta-1)):# The Smartstart's stratage of waiting
            if (len(temp_array) == 0):
                t = route_length/(theta-1)
            else:    
                t = table[temp_array[0]][2]
                add = []
                for j in range(len(temp_array)):
                    if (table[temp_array[j]][2] <= t):
                        route.append(table[temp_array[j]])
                        route_requests.append(temp_array[j])
                        add.append(temp_array[j])
                for k in range(len(add)):
                    temp_array.remove(add[k])
                route_order, route_length = brutal_force_opt(table, route_requests,t)#update the schedule
        t += route_length  
        for k in range(len(route_order)):
            #update the served and the unserved array
            serve.append(route_order[k])
            unserve.remove(route_order[k])
        count += len(route)     
    return t         
     
# Input data 
random.seed(66)
n = int(input("input the quantity : "))
df_1 = pd.read_excel('gr202.xlsx')
m = len(df_1)
for i in range(m):
    rand = random.randint(1, 500)
    df_1['t'].iloc[i]=rand
 
df_2 = pd.read_excel('gr229.xlsx')
m = len(df_2)
for i in range(m):
    rand = random.randint(1, 500)
    df_2['t'].iloc[i]=rand
    
df_3 = pd.read_excel('a280.xlsx')
m = len(df_3)
for i in range(m):
    rand = random.randint(1, 500)
    df_3['t'].iloc[i]=rand
    
y_model2 = []
y_smart = []
time_error = [0,0.2,0.4,0.6,0.8,1.0,1.2]
start = time.time()
for k in range(len(time_error)):
    std_scalar = time_error[k]
    avg_model2_ratio = 0.0 
    avg_smart_ratio = 0.0
    count = 50
    for num in range(count):
        print("\nIteration " + str(num+1))
        if(count < 20):
            df_all = df_1
        elif (count >=20 and count < 40):
            df_all = df_2
        else :
            df_all = df_3
        requests = []
        for i in range(n):
            requests.append(i)   
        select = []
        for i in range(len(df_all)):
            select.append(i)     
        pick = random.sample(select,n)
        print("pick : ",pick)
        table = []       
        for i in range(n):
            lst = []
            for j in range(3):
                lst.append(df_all.iloc[pick[i],j])
            table.append(lst)
             
        print('The actual value :')
        print(table)
        table_sort = sorted(table, key = itemgetter(2))
            
        # Add the Gaussian noise to actual requests as prediction
        predict = []
        std_array = []
        mean_array = []
        for i  in range(3):
            val_std = []
            val_mean = []
            for j in range(n):
                val_std.append(table[j][i])
                val_mean.append(table[j][i])
            std = np.std(val_std)
            mean = statistics.mean(val_mean)
            std_array.append(std)
            mean_array.append(mean)
        for i in range(n):
            pred2 = []
            for j in range(3):
                mu = 0.0
                std = std_scalar * std_array[j]
                np.random.seed(0)
                noise = np.random.normal(mu, std, 1)
                pred = table[i][j] + noise
                pred2.append(pred[0])
            predict.append((pred2))   
        print('The predict last arrival time : ')
        print(max(predict[i][2] for i in range(n)))       
        table_off = []
        for i in range(n):
            lst = []
            lst.append(table[i][0])
            lst.append(table[i][1])
            table_off.append(lst)    
        # Output
        opt_order, opt_length  = brutal_force_opt(table,requests,0)
        last_opt = Last_opt(table_sort,n,max(predict[i][2] for i in range(n)))
        smart_opt = Smartstart(table_sort,n,2) # 2 is best thata value of Smartstart under the optimal offline route 
        print("The length of the optimal route is " + str(opt_length))
        print("By Model 2 with opt : " + str(last_opt))
        print("the ratio is : " + str(last_opt/opt_length))
        print("By Smartstart : " + str(smart_opt))
        print("the ratio is : " + str(smart_opt/opt_length))
        avg_model2_ratio += last_opt/opt_length
        avg_smart_ratio += smart_opt/opt_length

    print("\nthe Model 2  average ratio : "+ str(avg_model2_ratio/count))
    print("the Smartstart average ratio : "+ str(avg_smart_ratio/count))
    
    y_model2.append(avg_model2_ratio/count)
    y_smart.append(avg_smart_ratio/count)
    
print(y_model2)
print(y_smart)

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

x = time_error
plt.rcParams["figure.figsize"] = (10,6)  
plt.xlabel('$\sigma_{err}$ (normalized as $\sigma_{err}/\mu_t$) ', fontsize = 28)
plt.ylabel('Competituve ratio', fontsize = 28)
plt.plot(x, y_smart, 'o-', label='SMARTSTART')
plt.plot(x, y_model2, 'o-', label='LAR-LAST',color='maroon')
plt.ylim([1.0,2.5])     
plt.legend(loc=2, prop={'size': 20})
plt.title("Model 2 $\epsilon_{last}$(small network)", fontsize = 32)
plt.savefig("Model 2 Last Arrival Time Error",dpi=300, bbox_inches = "tight")  
#plt.show()  
plt.close()

print("The time used to execute this is given below")
end = time.time()
print(end - start)    
