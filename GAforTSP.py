import random
def loadData(file_path):
    data =[]
    with open(file_path, 'r') as file:
      for line in file:
        row = [int(x) for x in line.split()]
        data.append(row)
    return data

def fitnessScore(data,path):
   cost=0
   size_data = len(data)
   if (len(path) == size_data):
      for i in range(1,size_data):
          cost+= data[path[i-1]][path[i]]
      cost+= data[path[size_data-1]][0]    
      return -cost
   return None

def fitnessEvaluationProcedure(data,population):
   score=[]
   for item in population:
      score.append(fitnessScore(data,item))
   sorted_population = [x for x, _ in sorted(zip(population, score), key=lambda x: x[1], reverse=True)]   
   for item in sorted_population:
       print(f"Sorted population: {item}, score: {fitnessScore(data,item)}")

   return sorted_population
 
def CheckAppdendix(path, new_num):
   for item in path:
      if (item == new_num):
         return False
   return True

def initOneSolution(size_data):
   path=[]
   path.append(0)
   while(len(path)< size_data):
      new_num= random.randint(1, size_data-1)
      if (CheckAppdendix(path,new_num)):
         path.append(new_num)
   return path
def checkValidSolution(size_data,item):
   if (len(item)!=size_data):
      return False
   if (item[0]!=0):
      return False
   check=[]
   for i in range(size_data):
      check.append(0)
   for ele in item:
      if (ele >=0 and ele < size_data):
         check[ele] = check[ele]+1
      else:
          return False   
   for i in range(size_data):
       if (check[i]>1):
          return False  
   return True
def Mutation(child, p= 0.01):
    size_data = len(child)
    ran_num = random.sample(range(0,size_data))
    if (ran_num <= size_data*p):
      print("mutation")

def ordered_Crossover(p1, p2):
    if len(p1) != len(p2):
        return None
    
    size_data = len(p1)
    child1 = [None] * size_data
    child2 = [None] * size_data
    
    # Ensure the entire range is considered for crossover
    start, end = sorted(random.sample(range(1,size_data), 2))
    
    # Copy subsections from parents
    child1[start:end+1] = p2[start:end+1]
    child2[start:end+1] = p1[start:end+1]

    # Create mapping of parent subsections
    mapping2to1 = {p2[i]: p1[i] for i in range(start, end+1)}
    mapping1to2 = {p1[i]: p2[i] for i in range(start, end+1)}
    
    # Fill missing positions in child1
    for i in range(size_data):
        if child1[i] is None:
            current_gene = p1[i]
            # Resolve potential conflicts with the mapping
            while current_gene in mapping2to1:
                current_gene = mapping2to1[current_gene]
            child1[i] = current_gene
        
        # Fill missing positions in child2
        if child2[i] is None:
            current_gene = p2[i]
            # Resolve potential conflicts with the mapping
            while current_gene in mapping1to2:
                current_gene = mapping1to2[current_gene]
            child2[i] = current_gene
    
    return child1, child2

def InitPopulation(data, numOfPopulation=5):
   size_data = len(data)
   population= []
   for i in range(numOfPopulation):
      population.append(initOneSolution(size_data))

   return population

def BestFitnessSelection(data,population,num=5):
    sorted_population = fitnessEvaluationProcedure(data,population)
    if (num < len(sorted_population)):
         new_population = sorted_population[:num]
         return new_population
    else :
         return sorted_population

def random_parent_crossover(population, new_num= 5):
   size_data = len(population)
   for i in range(new_num):
      rand1, rand2 = sorted(random.sample(range(1,size_data), 2))
      p1 = population[rand1]
      p2 = population[rand2]
      c1,c2 = ordered_Crossover(p1,p2)
      population.append(c1)
      population.append(c2)

   return population

def StopCondition(solution, epoch_threshold=10):
   size_data = len(solution)
   if (size_data < epoch_threshold):
      return False
   best = solution[size_data-1]
   count = 0
   for item in solution:
      if (item == best):
         count = count + 1
      if (count > epoch_threshold):
         return True
   
   return False

def main(data_file='TSP_testcase/case_1.txt',epoch=50,num_init_population=5, epoch_threshold= 10, population_size=5):
   data = loadData(data_file)
   population= InitPopulation(data, numOfPopulation= num_init_population)     
   solution = [] 
   for i in range(epoch):
      print("epoch: ",i)
      population = random_parent_crossover(population,new_num=population_size)
      population = BestFitnessSelection(data,population, num= population_size) 
      solution.append(fitnessScore(data,population[0]))
      if StopCondition(solution,epoch_threshold=epoch_threshold):
         print("Catch Stop conditions")
         break

   print("solution: ",population[0], ", score: ",fitnessScore(data,population[0]))      

   
if __name__ == "__main__":
    main()   