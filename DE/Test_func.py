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