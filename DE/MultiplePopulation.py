import random
import numpy as np
from SHADE import  Weighted_mean
from JADE import choose_random_exclude_n0, getBestsample,generateAvector,genneratePopulation,CheckInRange
from DE_simp import genSchemeDe1, genSchemeDe2
from Test_func import First_Dejong_func,Second_Dejong_func
from SHADE import WeightedL_mean
import math
from CEC2013 import BenchmarkFunctions
def getNBestsample(population,p = 0.2, func = First_Dejong_func, num=1):
    if(len(population) == 0):
        return None
    population_size = len(population)
    sorted_population = sorted(population, key= func)
    n_best = int(population_size*p)
    ListBest = []
    for index in range(num):
       n_ran = random.randint(0, n_best)
       ListBest.append(sorted_population[n_ran])

    return ListBest


def currentTopbest1(x_i, x_best, x_r1, x_r2, F): 
    data_size = len(x_i)
    v_i = [None]*data_size
    for index in range(data_size):
       v_i[index] = x_i[index] + F*(x_r1[index] - x_r2[index]) + F*(x_best[index]- x_i[index])
    
    return v_i

def currentTopbest2(x_i, x_best1, x_best2, x_r1, x_r2, F): 
    data_size = len(x_i)
    v_i = [None]*data_size
    for index in range(data_size):
       v_i[index] = x_i[index] + F*(x_r1[index] - x_r2[index]) + F*(x_best1[index]- x_i[index]) + F*(x_best2[index]- x_i[index]) 
    
    return v_i

def pbest1(x_i, x_best, x_r1, x_r2, F):
    data_size = len(x_i)
    v_i = [None]*data_size
    for index in range(data_size):
       v_i[index] =  F*(x_r1[index] - x_r2[index]) + x_best[index]
    
    return v_i

def sample_decision(score1 = 0, score2 = 0, score3 = 0):
    p1 = math.exp(score1) /(math.exp(score1) + math.exp(score2) + math.exp(score3))
    p2 = math.exp(score2) /(math.exp(score1) + math.exp(score2) + math.exp(score3))
    p3 = math.exp(score3) /(math.exp(score1) + math.exp(score2) + math.exp(score3))
    ran_num = random.uniform(0, 1)
    if ran_num <= p1:
        return 1
    if ran_num <=p1+p2 and ran_num > p1:
        return 2
    return 3        
    
def genSHADE(population, archive_set, current_index,S_CR,S_F,Delta,mean_CR=0.5, mean_F=0.5, func =First_Dejong_func,p=0.2, score1 = 0, score2 = 0, score_3 = 0):
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
    best_list = getNBestsample(population=population, func= func, p=p, num=2)
    best1 = best_list[0]
    best2 = best_list[1]
    decision = sample_decision(score1= score1, score2= score2, score3= score_3)

    data_size = len(x_i)
    v_i = [None]*data_size
    
    if (decision == 1):
        v_i = currentTopbest1(x_i= x_i, x_best= x_best, x_r1= x_r1, x_r2= x_r2, F= F)
    if (decision == 2):
        v_i = pbest1(x_i=x_i, x_best= x_best, x_r1= x_r1, x_r2= x_r2, F= F)
    else:
        v_i =  currentTopbest2(x_i=x_i,x_best1= best1, x_best2= best2, x_r1= x_r1, x_r2 = x_r2, F= F)

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
        if decision == 1: 
            score1 = score1 + min(1/4,abs(func(x_i) - func(u_i))/(abs(func(x_i)) + 1e-8))
        if decision == 2:
            score2 = score2 + min(1/4,abs(func(x_i) - func(u_i))/(abs(func(x_i)) + 1e-8))
        if decision == 3:     
            score2 = score2 + min(1/4,abs(func(x_i) - func(u_i))/(abs(func(x_i)) + 1e-8))
    return u_i, S_CR,S_F, Delta, score1, score2, score_3


def MixSHADE(population_size = 10, epochs = 10, data_size = 3,  c = 0.1, func = First_Dejong_func, min_num = -5.12, max_num = 5.12, History_size = 5):
    population = genneratePopulation(population_size=population_size, data_size= data_size, max_num= max_num, min_num= min_num)
    archive_set = []
    H_CR  = [0.5]*History_size
    H_F = [0.5]*History_size
    k=0
    score_1 = 0
    score_2 = 0
    score_3 = 0
    for epoch in range(epochs):
        
        S_CR = []
        S_F = []
        Delta = []
        
        for index in range(len(population)):
            p = random.uniform(2/population_size, 0.2)
            rand_index = random.randint(0,History_size-1)
            mean_CR = H_CR[rand_index]
            mean_F = H_F[rand_index]
            new_vec,S_CR,S_F, Delta,score_1, score_2, score_3 = genSHADE(population= population,archive_set = archive_set, current_index= index, S_CR = S_CR, S_F = S_F, Delta = Delta ,mean_F = mean_F, mean_CR = mean_CR, func = func, p=p, score1=score_1, score2=score_2, score_3=score_3)
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
        print(f"score_1: {score_1}, score2: {score_2}, score3: {score_3}")
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
    func1 = ben.get_function(func_num = 2)
    MixSHADE(epochs= 100, func= func1, History_size= 50, data_size= 10, population_size= 50)
    