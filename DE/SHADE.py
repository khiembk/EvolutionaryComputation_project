import random
import numpy as np
from Test_func import First_Dejong_func,Second_Dejong_func,Griewangk_func,Third_DeJong_func,fourth_DeJong_func
from JADE import choose_random_exclude_n0, getBestsample,generateAvector,genneratePopulation,CheckInRange
from CEC2013 import BenchmarkFunctions

def Weighted_mean(data, Delta):
    w_sum = 0
    weight = sum(Delta)
    if (len(data) != len(Delta)):
        return None
    for index in range(len(data)):
        w_sum += Delta[index]*data[index]

    return w_sum/weight

def WeightedL_mean(data, Delta):
    w_sum = 0
    weight = 0

    if (len(data) != len(Delta)):
        return None
    for index in range(len(data)):
        weight += Delta[index]*data[index]
    for index in range(len(data)):
        w_sum += Delta[index]*data[index]*data[index]

    return w_sum/weight

def genSHADE(population, archive_set, current_index,S_CR,S_F,Delta,mean_CR=0.5, mean_F=0.5, func =First_Dejong_func,p=0.2):
    CR =  random.normalvariate(mean_CR, 0.1)
    F =  np.random.standard_cauchy() * 0.1 + mean_F
    if CR > 1 :
        CR = 1
    if CR < 0 :
        CR = 0
    
    while F <=0 :
        F =  np.random.standard_cauchy() * 0.1 + mean_F  
    if F > 1 :
        F = 1    

    size_po = len(population)
    if (current_index > size_po):
        return None  
    x_i = population[current_index]
    rand_1, rand_2 = choose_random_exclude_n0(size_po,current_index,2)
    x_r1 = population[rand_1]
    x_r2 = population[rand_2]
    
    if (len(archive_set)>0):
        archive_p = random.random()
        if archive_p <= (len(archive_set)/(len(archive_set) + len(population))):
           r_2 = random.randint(0, len(archive_set)-1)
           x_r2 = archive_set[r_2]
    x_best = getBestsample(population= population, func= func,p = p)
    data_size = len(x_i)
    v_i = [None]*data_size
    for index in range(data_size):
       v_i[index] = x_i[index] + F*(x_r1[index] - x_r2[index]) + F*(x_best[index]- x_i[index])
    
    u_i = [None]*data_size
    for index in range(data_size):
        ran_num = random.random()
        if (ran_num <= CR):
           u_i[index] = v_i[index]
        else:
           u_i[index] = x_i[index]    
    
    if (func(u_i) < func(x_i)):
        S_CR.append(CR)
        S_F.append(F)
        Delta.append(abs(func(x_i) - func(u_i)))
    return u_i, S_CR,S_F, Delta

def SHADE(population_size = 10, epochs = 10, data_size = 3,  genFunc = genSHADE, c = 0.1, func = First_Dejong_func, min_num = -5.12, max_num = 5.12, History_size = 5):
    population = genneratePopulation(population_size=population_size, data_size= data_size, max_num= max_num, min_num= min_num)
    archive_set = []
    H_CR  = [0.5]*History_size
    H_F = [0.5]*History_size
    k=0
    for epoch in range(epochs):
        
        S_CR = []
        S_F = []
        Delta = []
        
        for index in range(len(population)):
            p = random.uniform(2/population_size, 0.2)
            rand_index = random.randint(0,History_size-1)
            mean_CR = H_CR[rand_index]
            mean_F = H_F[rand_index]
            new_vec,S_CR,S_F, Delta = genFunc(population= population,archive_set = archive_set, current_index= index, S_CR = S_CR, S_F = S_F, Delta = Delta ,mean_F = mean_F, mean_CR = mean_CR, func = func, p=p)
            if new_vec is not None :
                x_i = population[index]
                if func(new_vec) <= func(x_i):
                    population[index] = new_vec

                if func(new_vec) < func(x_i):    
                    if (len(archive_set) < len(population)):
                        archive_set.append(x_i)
                    else : 
                        random_element = random.choice(archive_set)
                        archive_set.remove(random_element)
                        archive_set.append(x_i)
    
        print("S_CR: ", S_CR)
        print("S_F :",S_F)
        if (k > History_size -1): 
            k = 0
        if (len(S_CR) > 0) :
           mean_CR = Weighted_mean(S_CR,Delta)
           mean_CR = CheckInRange(mean_CR)
           H_CR[k] = mean_CR
        if (len(S_F) >0):    
           mean_F = WeightedL_mean(S_F,Delta)
           mean_F = CheckInRange(mean_F)
           H_F[k] = mean_F
        if (len(S_CR) >0 or len(S_F) > 0):
            k = k + 1   
        print("At epoch: ", epoch)
        print("current F History: ", H_F)
        print("curent CR History: ",H_CR)
        for index in range(population_size):
            print("population: ",population[index]," func_val:", func(population[index]))
            

    x_best = getBestsample(population, func= func, p=0)
    print("Solution: ",x_best,"func val: ", func(x_best))
    return  x_best        

if __name__ == "__main__":
    ben = BenchmarkFunctions(dimension= 10)
    func1 = ben.get_function(func_num = 1)
    SHADE(epochs= 100, func= func1, data_size= 10, History_size= 50)
    