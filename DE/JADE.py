import random
import numpy as np
from Test_func import First_Dejong_func,Second_Dejong_func


def choose_random_exclude_n0(n, n0, num=3):
    # Create a list of integers from 0 to n-1, excluding n0
    choices = [i for i in range(n) if i != n0]
    
    # Randomly select 3 integers from the list
    return random.sample(choices, num)

def getBestsample(population,p = 0.2, func = First_Dejong_func):
    if(len(population) == 0):
        return None
    population_size = len(population)
    sorted_population = sorted(population, key= func)
    n_best = int(population_size*p)
    n_ran = random.randint(0, n_best)

    return sorted_population[n_ran]
    


def genSchemeDe(population, archive_set, current_index,S_CR,S_F,mean_CR=0.5, mean_F=0.5, func =First_Dejong_func,p=0.2):
    CR =  random.normalvariate(mean_CR, 0.1)
    F =  np.random.standard_cauchy() * 0.1 + mean_F
    size_po = len(population)
    if (current_index > size_po):
        return None  
    x_i = population[current_index]
    rand_1, rand_2, rand_3 = choose_random_exclude_n0(size_po,current_index,3)
    x_r1 = population[rand_1]
    x_r2 = population[rand_2]
    x_r3 = population[rand_3]
    if (len(archive_set)>0):
        r_2 = random.randint(0, len(archive_set)-1)
        x_r2 = archive_set[r_2]
    x_best = getBestsample(population= population, func= func,p = p)
    data_size = len(x_i)
    v_i = [None]*data_size
    for index in range(data_size):
       v_i[index] = x_r1[index] + F*(x_r2[index] - x_r3[index]) + F*(x_best[index]- x_r1[index])
    
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
    return u_i, S_CR,S_F

    
def generateAvector(data_size, min_num = -5.12, max_num = 5.12):
    new_vec = [None]*data_size

    for index in range(data_size):
        ran_num = random.random()
        new_vec[index] = min_num + ran_num*(max_num - min_num)

    return new_vec

def genneratePopulation(population_size = 10, data_size = 3, max_num = 5.12, min_num= -5.12):
    population = []
    for index in range (population_size):
        new_vec = generateAvector(data_size=data_size, min_num= min_num, max_num= max_num)
        population.append(new_vec)
    
    return population

def meanLehmer(num_list):
    sum_list = sum(num_list)
    lehmer = 0
    for num in num_list :
        lehmer += num*num

    return lehmer/sum_list

def main(population_size = 10, epochs = 10, data_size = 3, mean_F = 0.5, mean_CR = 0.5, genFunc = genSchemeDe, c = 0.1, func = First_Dejong_func, min_num = -5.12, max_num = 5.12):
    population = genneratePopulation(population_size=population_size, data_size= data_size, max_num= max_num, min_num= min_num)
    archive_set = []
    for epoch in range(epochs):
        S_CR = []
        S_F = []

        for index in range(len(population)):
            new_vec,S_CR,S_F = genFunc(population= population,archive_set = archive_set, current_index= index, S_CR = S_CR, S_F = S_F, mean_F = mean_F, mean_CR = mean_CR, func = func)
            if new_vec is not None :
                x_i = population[index]
                if func(new_vec) < func(population[index]):
                    population[index] = new_vec
                    if (len(archive_set) < len(population)):
                        archive_set.append(x_i)
                    else : 
                        random_element = random.choice(archive_set)
                        archive_set.remove(random_element)
                        archive_set.append(x_i)
    
        print("S_CR: ", S_CR)
        print("S_F :",S_F)
        if (len(S_CR) > 0) :
           mean_CR = (1-c)*mean_CR + c*sum(S_CR)/len(S_CR)
        if (len(S_F) >0):    
           mean_F = (1-c)*mean_F + c*meanLehmer(S_F)
        print("At epoch: ", epoch)
        print("current mean F: ", mean_F, "cureent main CR: ", mean_CR)
        for index in range(population_size):
            print("population: ",population[index]," func_val:", func(population[index]))

if __name__ == "__main__":
    main(epochs= 20, genFunc= genSchemeDe, func= Second_Dejong_func, min_num=-2.48, max_num=2.48)
    