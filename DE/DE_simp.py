import random
def First_Dejong_func(data):
    data_size = len(data)
    sum = 0
    for item in data:
        sum += item*item

    return sum 

def choose_random_exclude_n0(n, n0, num=3):
    # Create a list of integers from 0 to n-1, excluding n0
    choices = [i for i in range(n) if i != n0]
    
    # Randomly select 3 integers from the list
    return random.sample(choices, num)

def getBestsample(population, func = First_Dejong_func):
    if(len(population) == 0):
        return None
    population_size = len(population)
    current_best = population[0]
    current_val = func(current_best)

    for index in range(1,population_size):
        x_i = population[index]
        val_i = func(x_i)
        if val_i < current_val :
            current_best = x_i
            current_val = val_i

    return current_best         


def genSchemeDe2(population, current_index,CR=0.5, F=0.5, func =First_Dejong_func, lamda = 0.5):
    size_po = len(population)
    if (current_index > size_po):
        return None  
    x_i = population[current_index]
    rand_1, rand_2, rand_3 = choose_random_exclude_n0(size_po,current_index,3)
    x_r1 = population[rand_1]
    x_r2 = population[rand_2]
    x_r3 = population[rand_3]
    x_best = getBestsample(population= population, func= func)
    data_size = len(x_i)
    v_i = [None]*data_size
    for index in range(data_size):
       v_i[index] = x_r1[index] + F*(x_r2[index] - x_r3[index]) + lamda*(x_best[index]- x_r1[index])
    
    u_i = [None]*data_size
    for index in range(data_size):
        ran_num = random.random()
        if (ran_num <= CR):
           u_i[index] = v_i[index]
        else:
           u_i[index] = x_i[index]    

    if (func(u_i)< func(x_i)):
        return u_i
    else: 
        return x_i
    
def genSchemeDe1(population,current_index,CR=0.5,F=0.5, func = First_Dejong_func):  
    size_po = len(population)
    if (current_index > size_po):
        return None  
    x_i = population[current_index]
    rand_1, rand_2, rand_3 = choose_random_exclude_n0(size_po,current_index,3)
    x_r1 = population[rand_1]
    x_r2 = population[rand_2]
    x_r3 = population[rand_3]
    data_size = len(x_i)
    v_i = [None]*data_size
    for index in range(data_size):
       v_i[index] = x_r1[index] + F*(x_r2[index] - x_r3[index])
    
    u_i = [None]*data_size
    for index in range(data_size):
        ran_num = random.random()
        if (ran_num <= CR):
           u_i[index] = v_i[index]
        else:
           u_i[index] = x_i[index]    

    if (func(u_i)< func(x_i)):
        return u_i
    else: 
        return x_i
    
def generateAvector(data_size, min_num = -5.12, max_num = 5.12):
    new_vec = [None]*data_size

    for index in range(data_size):
        ran_num = random.random()
        new_vec[index] = min_num + ran_num*(max_num - min_num)

    return new_vec

def genneratePopulation(population_size = 10, data_size = 3):
    population = []
    for index in range (population_size):
        new_vec = generateAvector(data_size=data_size)
        population.append(new_vec)
    
    return population

def main(population_size = 10, epochs = 10, data_size = 3, genFunc = genSchemeDe1):
    population = genneratePopulation(population_size=population_size, data_size= data_size)
    
    for epoch in range(epochs):

        for index in range(population_size):
            new_vec = genFunc(population= population, current_index= index)
            if new_vec is not None :
                population[index] = new_vec

        print("At epoch: ", epoch)
        for index in range(population_size):
            print("population: ",population[index]," func_val:", First_Dejong_func(population[index]))

if __name__ == "__main__":
    main(epochs= 20, genFunc= genSchemeDe2)