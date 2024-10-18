import math 
def First_Dejong_func(data):
    data_size = len(data)
    sum = 0
    for item in data:
        sum += item*item

    return sum 

def Second_Dejong_func(data):
    data_size = len(data)
    sum = 0 
    if (data_size == 0):
        return sum
    if (data_size == 1):
        return data[0]*data[0]
    
    if (data_size == 2):
        sum += 100*(data[0]**2 - data[1])**2 + (1 - data[1])**2
        return sum 
    
    if (data_size > 2):
        sum += 100*(data[0]**2 - data[1])**2 + (1 - data[1])**2
        for i in range(2, data_size):
            sum += data[i]**2

    return sum     

def Third_DeJong_func(data):
    sum = 30
    for index in range(len(data)):
        sum+= math.floor(data[index])
    return sum    

def fourth_DeJong_func(data, eta = 0.1):
    sum = 0
    for index in range(len(data)):
        sum += eta + (index+1)*(data[index])**4
    return sum 

def Griewangk_func(data):
    sum = 0
    mul = 1
    for index in range(len(data)):
        sum += 1 + ((data[index])**2)/4000 
        mul = mul*math.cos(data[index]/math.sqrt(index+1))
    return sum + mul