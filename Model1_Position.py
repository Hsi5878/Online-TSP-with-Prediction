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
import matplotlib.pyplot as plt
import statistics

# Brute Force Solver
def brutal_force_opt(table,index_to_serve,start_time):
    perm = list(itertools.permutations(index_to_serve))#list all permutations
    optimal_order = []
    optimal_length = 1000000
    for route_num in range(len(perm)):
        t = 0
        if((table[perm[route_num][0]][0] ** 2 + table[perm[route_num][0]][1] ** 2) ** (1/2) > (table[perm[route_num][0]][2]-start_time) ):
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

# Algorithm Trust
def Trust(table, predict, n):
    requests = [] 
    for i in range(n):
        requests.append(i)
    T_order, T_length = brutal_force_opt(predict, requests,0)#compute the optimal order of prdicted route
    t = 0.0
    #Update the predicted route by adding the request x(i) after the predicted request x_hat(i)
    for i in range(n):
        if (i == 0): 
            distance = (predict[T_order[0]][0] ** 2 + predict[T_order[0]][1] ** 2) ** (1/2)
            t = max(t + distance, predict[T_order[0]][2])
        else :
            distance = ((predict[T_order[i]][0] - table[T_order[i-1]][0]) ** 2 + 
                        (predict[T_order[i]][1]- table[T_order[i-1]][1]) ** 2 ) ** (1/2)
            t = max(t + distance, predict[T_order[0]][2])      
        if (table[T_order[i]][2] <= t):
            distance = ((table[T_order[i]][0]- predict[T_order[i]][0] ) ** 2 + 
                       (table[T_order[i]][1]- predict[T_order[i]][1] ) ** 2 ) ** (1/2)
            t += distance
        else :
            t = table[T_order[i]][2]
            distance = ((table[T_order[i]][0] - predict[T_order[i]][0]) ** 2 + 
                       (table[T_order[i]][1]- predict[T_order[i]][1] ) ** 2 ) ** (1/2)
            t += distance                      
    t += (table[T_order[-1]][0] ** 2 + table[T_order[-1]][1] ** 2) ** (1/2)
        
    return T_order, t  

# Learning-Augmented Routing With Identity (LAR-ID)
def ID(table, predict, n):
    route_1_order, route_1_length = Trust(table, predict, n) # The route desighed by ALG Trust
    unserve = []
    serve = []
    arrival_time = []        
    t_n = max(table[i][2] for i in range(n))  #t_n is the presented time of the last request 
    complete = 0
    t = 0.0
    for i in range(n):#Update the route
        # Server starts fromm the position of  the actual request to the position of the next predicted request
        if (i == 0): 
            distance = (predict[route_1_order[0]][0] ** 2 + predict[route_1_order[0]][1] ** 2) ** (1/2)
            t = max(t + distance, predict[route_1_order[0]][2])
        else :
            distance = ((predict[route_1_order[i]][0] - table[route_1_order[i-1]][0]) ** 2 + 
                        (predict[route_1_order[i]][1]- table[route_1_order[i-1]][1]) ** 2 ) ** (1/2)
            t = max(t + distance, predict[route_1_order[0]][2])      
        if (t <= t_n):
            arrival_time.append(t)
        else : # The currnrt time is greater than t_n
            complete = 1
            break
        # Server starts fromm the position of the predicted request to the position of the corrresponding predicted request
        if (table[route_1_order[i]][2] <= t):
            distance = ((table[route_1_order[i]][0]- predict[route_1_order[i]][0] ) ** 2 + 
                       (table[route_1_order[i]][1]- predict[route_1_order[i]][1] ) ** 2 ) ** (1/2)
            t += distance
        else :
            t = table[route_1_order[i]][2]
            distance = ((table[route_1_order[i]][0] - predict[route_1_order[i]][0]) ** 2 + 
                       (table[route_1_order[i]][1]- predict[route_1_order[i]][1] ) ** 2 ) ** (1/2)
            t += distance           
        if (t <= t_n):
            arrival_time.append(t)
            serve.append(route_1_order[i])
        else : # The currnrt time is greater than t_n
            complete = 0
            break 
    # find the position (x,y) when t_n is come
    if (i == 0):
        if(complete == 1):
            stop = t_n
            x = 0 + stop * (predict[route_1_order[0]][0])/(((predict[route_1_order[0]][0]) ** 2 + predict[route_1_order[0]][1] ** 2) ** (1/2))
            y = 0 + stop * (predict[route_1_order[0]][1])/(((predict[route_1_order[0]][0]) ** 2 + predict[route_1_order[0]][1] ** 2) ** (1/2))
        else :
            stop = t_n - max(arrival_time[0], table[route_1_order[0]][2])
            x = predict[route_1_order[0]][0] + stop * ((table[route_1_order[0]][0]-predict[route_1_order[0]][0])/
                                                    ((table[route_1_order[0]][0]-predict[route_1_order[0]][0]) ** 2 +
                                                    (table[route_1_order[0]][1]-predict[route_1_order[0]][1]) ** 2) ** (1/2))
            y = predict[route_1_order[0]][1] + stop * ((table[route_1_order[0]][1]-predict[route_1_order[0]][1])/
                                                    ((table[route_1_order[0]][0]-predict[route_1_order[0]][0]) ** 2 +
                                                    (table[route_1_order[0]][1]-predict[route_1_order[0]][1]) ** 2) ** (1/2))
    else:
        if(complete == 1):
            stop = t_n - arrival_time[i*2-1]
            x = table[route_1_order[i-1]][0] + stop * ((predict[route_1_order[i]][0]-table[route_1_order[i-1]][0])/
                                                    ((predict[route_1_order[i]][0]-table[route_1_order[i-1]][0]) ** 2 +
                                                    (predict[route_1_order[i]][1]-table[route_1_order[i-1]][1]) ** 2) ** (1/2))
            y = table[route_1_order[i-1]][1] + stop * ((predict[route_1_order[i]][1]-table[route_1_order[i-1]][1])/
                                                    ((predict[route_1_order[i]][0]-table[route_1_order[i-1]][0]) ** 2 +
                                                    (predict[route_1_order[i]][1]-table[route_1_order[i-1]][1]) ** 2) ** (1/2))
        else : 
            stop = t_n - max(arrival_time[i*2], table[route_1_order[i]][2])
            x = predict[route_1_order[i]][0] + stop * ((table[route_1_order[i]][0]-predict[route_1_order[i]][0])/
                                                    ((table[route_1_order[i]][0]-predict[route_1_order[i]][0]) ** 2 +
                                                    (table[route_1_order[i]][1]-predict[route_1_order[i]][1]) ** 2) ** (1/2))
            y = predict[route_1_order[i]][1] + stop * ((table[route_1_order[i]][1]-predict[route_1_order[i]][1])/
                                                    ((table[route_1_order[i]][0]-predict[route_1_order[i]][0]) ** 2 +
                                                    (table[route_1_order[i]][1]-predict[route_1_order[i]][1]) ** 2) ** (1/2))
        
    far = (x ** 2 + y ** 2) **(1/2)
    home_time = t_n + far
    for i in range(n):
        if(i not in serve):
            unserve.append(i)
    final_order , final_length = brutal_force_opt(table, unserve, home_time)  #Desighn the last schedule of the adjusted route 
    route_2_length = home_time + final_length
    
    return min(route_1_length, route_2_length) #choose the smaller route between Trust and adjusted route

# Algorithm Smartstart
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
        route_order, route_length = brutal_force_opt(table, route_requests,t) # compute the optimal offline route that the requests are presented brfore t     
        temp_array = unserve[:]
        for i in range(len(route_order)):
            temp_array.remove(route_order[i]) # The Smartstart's stratage of waiting
        while(t < route_length/(theta-1)):
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
                route_order, route_length = brutal_force_opt(table, route_requests,t) #update the schedule
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
    
pos_error = [0,0.2,0.4,0.6,0.8,1.0,1.2]
time_error = [0.2] # fixed

start = time.time()
# Start computing
y_trust = []
y_model1 = []
y_smart = []
for k in range(len(pos_error)):
    std_scalar_pos = pos_error[k]
    std_scalar_time = time_error[0]
    avg_ratio = 0.0 
    avg_smart_ratio = 0.0
    avg_trust_ratio = 0.0
    trust_ratio = 0.0
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
        table = [] 
        for i in range(n):
            lst = []
            for j in range(3):
                lst.append(df_all.iloc[pick[i],j])
            table.append(lst)       
        print('The actual value :')
        print(table)
        table_sort = sorted(table, key = itemgetter(2))       
        # add the Gaussian noise to the actual requests as prediction
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
        std_scalar = [std_scalar_pos, std_scalar_pos , std_scalar_time]
        for i in range(n):
            pred2 = []
            for j in range(3):
                mu = 0.0
                std = std_scalar[j] * mean_array[j]
                np.random.seed(0)
                noise = np.random.normal(mu, std, 1)
                pred = table[i][j] + noise
                pred2.append(pred[0])
            predict.append((pred2))   
        print('The predict value : ')
        print(predict)   
        # Output
        opt_order, opt_length  = brutal_force_opt(table,requests,0)
        trust_order, trust_length = Trust(table, predict, n)
        smart_opt = Smartstart(table_sort, n, 2)# 2 is best thata value of Smartstart under the optimal offline route 
        print("The length of the optimal route is " + str(opt_length))    
        print("By ALG Trust : " + str(trust_length))       
        model_1 = ID(table,predict,n)
        print("By Model 1 : " + str(model_1))
        print("By Smartstart : " + str(smart_opt))
        print ("The ratio of ALG Trust is : "+ str(trust_length/opt_length))
        print ("The ratio of Model1 is : "+ str(model_1/opt_length))
        print ("The ratio of Smartstart is : "+ str(smart_opt/opt_length))              

        avg_ratio += model_1/opt_length
        avg_smart_ratio += smart_opt/opt_length
        avg_trust_ratio += trust_length/opt_length
          
    print("\nthe ALG Trust average ratio : "+ str(avg_trust_ratio/count))
    print("the Model 1 average ratio : "+ str(avg_ratio/count))
    print("the Smartstart average ratio : "+ str(avg_smart_ratio/count))
    print("the trust ratio : " + str(trust_ratio/count))    
    y_trust.append(avg_trust_ratio/count)
    y_model1.append(avg_ratio/count)
    y_smart.append(avg_smart_ratio/count)

    
print(y_trust)
print(y_model1)
print(y_smart)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
x = pos_error
plt.rcParams["figure.figsize"] = (10,6)  
plt.xlabel('$\sigma_{err}$ (normalized as $\sigma_{err}/\mu_p)$ ', fontsize = 28)
plt.ylabel('Competituve ratio', fontsize = 28)
plt.plot(x, y_smart, 'o-', label='SMARTSTART')
plt.plot(x, y_model1, 'o-', label='LAR-SEQUENCE')
plt.plot(x, y_trust, 'o-', label='LAR-TRUST')
plt.ylim([1.0,2.5])     
plt.legend(loc=2, prop={'size': 20})
plt.title("Model 1 $\epsilon_{pos}$", fontsize = 32)
plt.savefig("Model 1 Position Error",dpi=300, bbox_inches = "tight")  
#plt.show()  
plt.close()
 
print("The time used to execute this is given below")
end = time.time()
print(end - start) 
