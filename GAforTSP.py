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

def Crossover(p1,p2):
   child1 =[]
   child2 =[]
   child1.append(0)
   child2.append(0)

def InitPopulation(data, numOfPopulation=5):
   size_data = len(data)
   population= []
   for i in range(numOfPopulation):
      population.append(initOneSolution(size_data))

   return population

def BestFitnessSelection(data,population,num=5):
    sorted_population = fitnessEvaluationProcedure(data,population)
    new_population = sorted_population[:num]
    return new_population

def main():
   data = loadData('TSP_testcase/case_1.txt')
   population= InitPopulation(data)   
   fitnessEvaluationProcedure(data,population)   

if __name__ == "__main__":
    main()   