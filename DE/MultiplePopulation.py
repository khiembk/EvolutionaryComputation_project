import random
import numpy as np
from SHADE import genSHADE, Weighted_mean
from JADE import choose_random_exclude_n0, getBestsample,generateAvector,genneratePopulation,CheckInRange
from DE_simp import genSchemeDe1, genSchemeDe2
from Test_func import First_Dejong_func,Second_Dejong_func
def main(population_size = 10, epochs = 10, data_size = 3,  genFunc = genSHADE, c = 0.1, func = First_Dejong_func, min_num = -5.12, max_num = 5.12, History_size = 5):
    population = genneratePopulation(population_size=population_size, data_size= data_size, max_num= max_num, min_num= min_num)
    archive_set = []
    H_CR  = [0.5]*History_size
    H_F = [0.5]*History_size
    
    for epoch in range(epochs):
        rand_index = random.randint(0,History_size-1)
        mean_CR = H_CR[rand_index]
        mean_F = H_F[rand_index]
        S_CR = []
        S_F = []
        Delta = []
        p = random.uniform(2/population_size, 0.2)
        for index in range(len(population)):
            new_vec,S_CR,S_F, Delta = genFunc(population= population,archive_set = archive_set, current_index= index, S_CR = S_CR, S_F = S_F, Delta = Delta ,mean_F = mean_F, mean_CR = mean_CR, func = func, p=p)
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
           mean_CR = Weighted_mean(S_CR,Delta)
           mean_CR = CheckInRange(mean_CR)
           H_CR[rand_index] = mean_CR
        if (len(S_F) >0):    
           mean_F = Weighted_mean(S_F,Delta)
           mean_F = CheckInRange(mean_F)
           H_F[rand_index] = mean_F
        print("At epoch: ", epoch)
        print("current F History: ", H_F)
        print("curent CR History: ",H_CR)
        for index in range(population_size):
            print("population: ",population[index]," func_val:", func(population[index]))
            

    x_best = getBestsample(population, func= func, p=0)
    print("Solution: ",x_best,"func val: ", func(x_best))        

if __name__ == "__main__":
    main(epochs= 20, genFunc= genSHADE, func= First_Dejong_func, min_num=-5.12, max_num=5.12)
    