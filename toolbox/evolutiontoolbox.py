import random
import numpy as np
import matplotlib.pyplot as plt
import time
from toolbox.generaltoolbox import hammersley
        
#This file contains two classes: the population class and the evoalg class.
#The population class contains all the information and methods for one iteration of the Evolutionary Algorithm.
class population(object):
    def __init__(self,size,size_of_gene,number_of_offspring,fitnessfunction,minInputs,maxInputs,integer_values_bool=False,pseudo_random_bool=True):
        #This initialization method allows the user to determine the following:
        #<size> determines how large the population is.
        #<size_of_gene> determines how long the genes of the population are.
        #<number_of_offspring> determines how many recombinations of the original population get passed to the next iteration.
        #<fitness_function> determines the function that decides how fit a gene is. 
        #<minInputs> determines the minimum value for each element of the gene of the population.
        #<maxInputs> determines the maximum value for each element of the gene of the population.
        #<integer_values_bool> determines whether the genes are integer-valued or not.
        #<pseudo_random_bool> determines whether the initial genes are pseudo-randomized (instead of purely randomized).
        self.size=size
        self.size_of_gene=size_of_gene
        self.maxInputs=maxInputs.copy()
        self.minInputs=minInputs.copy()
        self.intValBool=integer_values_bool
        if self.intValBool:
            if pseudo_random_bool:
                ham_points=np.array(hammersley(self.size,self.size_of_gene,points=True))
                self.genes=[np.round(self.minInputs+ham_points[i]*(self.maxInputs-self.minInputs)).astype(int) for i in range(self.size)]
            else:
                self.genes=[np.array([random.randint(int(round(minInputs[j])),int(round(maxInputs[j]))) for j in range(self.size_of_gene)]) for i in range(self.size)]
        else:
            if pseudo_random_bool:
                ham_points=np.array(hammersley(self.size,self.size_of_gene,points=True))
                self.genes=[self.minInputs+ham_points[i]*(self.maxInputs-self.minInputs) for i in range(self.size)]
            else:
                self.genes=[self.minInputs+np.random.random_sample((self.size_of_gene,))*(self.maxInputs-self.minInputs) for i in range(self.size)]
        self.fitnessfunction=fitnessfunction
        if number_of_offspring<0:
            self.number_of_offspring=0
        elif number_of_offspring>self.size:
            self.number_of_offspring=self.size
        else:
            self.number_of_offspring=number_of_offspring
        self.current_generation=0
        self.computefitness()
        
    def computefitness(self):
        #This method computes the fitness of each gene in the population, then sorts the genes in ascending order based on fitness.
        self.fitness=np.array([self.fitnessfunction(self.genes[i]) for i in range(self.size)])
        pairs=sorted(zip(self.fitness,self.genes),key=lambda pair:pair[0],reverse=True)
        self.genes=[gene for _,gene in pairs]
        self.fitness=[fitness for fitness,_ in pairs]
        #self.genes=[gene for _,gene in sorted(zip(self.fitness,self.genes),key=lambda pair:pair[0],reverse=True)]
    
    def tournament_selection(self):
        #This method selects a gene from the population by selected the most fit gene from a randomly selected subset of the population. 
        k=min(int(self.size/2),1)
        random_index_Array=[random.randint(0,self.size-1) for i in range(k)]
        gindex=random_index_Array[0]
        for i in range(1,k):
            if self.fitness[random_index_Array[i]]>self.fitness[gindex]:
                gindex=random_index_Array[i]
            #if self.fitnessfunction(self.genes[indexArray[i]])>self.fitnessfunction(self.genes[gindex]):
            #    gindex=indexArray[i]
        return gindex
    
    def crossover(self):
        #This method generates a new generation of the population.
        #First, the method randomly recombines genes of the population a set number of times.
        #Then, the new set of recombined genes replaces the worst performing genes of the original population.
        if self.number_of_offspring>0:
            new_genes=[]
            while len(new_genes)<self.number_of_offspring:
                g1index=self.tournament_selection()
                if len(new_genes)==self.number_of_offspring-1 or self.size_of_gene==1:
                    new_genes.append(self.genes[g1index].copy())
                else:
                    g2index=self.tournament_selection()    
                    p=random.randint(1,self.size_of_gene-1)
                    g1=np.hstack((self.genes[g1index][:p],self.genes[g2index][p:]))#.copy()
                    g2=np.hstack((self.genes[g2index][:p],self.genes[g1index][p:]))#.copy()
                    new_genes.append(g1.copy())
                    new_genes.append(g2.copy())
                                 
            if self.number_of_offspring<self.size:
                self.genes=new_genes+self.genes[:self.size-self.number_of_offspring]
            else:
                self.genes=new_genes
    
    def mutate(self,mutation_chance=0.1):
        #This method adds very rarely random mutations to the population's genes in order to search for more optimal solutions.
        for i in range(self.size):
            if random.random()<mutation_chance:
                p=random.randint(0,self.size_of_gene-1)
                if self.intValBool:
                    self.genes[i][p]=random.randint(round(self.minInputs[p]),round(self.maxInputs[p]))
                else:
                    self.genes[i][p]=self.minInputs[p]+random.random()*(self.maxInputs[p]-self.minInputs[p])
            
#The evoalg class contains the information and methods for executing consecutive iterations of the Evolutionary Algorithm.
class evoAlg:
    def __init__(self,function,populationSize,geneLength,offspringNumber,minInputs,maxInputs,integer_values_bool=False,pseudo_random_bool=True):
        #This initialization method initializes the population and allows the user to determine the following:
        #<function> determines the function that decides how fit a gene is.
        #<populationSize> determines how large the population is.
        #<geneLength> determines how long the genes of the population are.
        #<offspringNumber> determines how many recombinations of the original population get passed to the next iteration. 
        #<minInputs> determines the minimum value for each element of the gene of the population.
        #<maxInputs> determines the maximum value for each element of the gene of the population.
        #<integer_values_bool> determines whether the genes are integer-valued or not.
        #<pseudo_random_bool> determines whether the initial genes are pseudo-randomized (instead of purely randomized).
        self.f=function
        self.popSize=populationSize
        self.geneLength=geneLength
        self.offspring=offspringNumber
        self.gens=0
        if "ndarray" not in type(maxInputs).__name__:
            self.maxInputs=np.array(maxInputs)
        else:
            self.maxInputs=maxInputs
        if "ndarray" not in type(minInputs).__name__:
            self.minInputs=np.array(minInputs)
        else:
            self.minInputs=minInputs
        if len(minInputs)==len(maxInputs):
            for i in range(len(minInputs)):
                if maxInputs[i]<minInputs[i]:
                    temp=maxInputs[i]
                    maxInputs[i]=minInputs[i]
                    minInputs[i]=temp
        else:
            raise Exception("minInputs (length of %i) and maxInputs (length of %i) are of different size"%(len(minInputs),len(maxInputs)))
        self.intValBool=integer_values_bool
        self.pseudoRandomBool=pseudo_random_bool
        self.population=population(self.popSize,self.geneLength,self.offspring,self.f,self.minInputs,self.maxInputs,integer_values_bool=self.intValBool,pseudo_random_bool=self.pseudoRandomBool)
        
    def evolve(self,generations,plot=True,inform=True,adapt=True,return_result=True):
        #This method manages the multiple iterations of the evolutionary algorithm.
        #The number of generations the algorithm is allowed to evolve the population is user-designated.
        #If <inform> is true, the algorithm interacts with the user to inform them of progress and results.
        self.gens=generations
        self.population.current_generation=0
        alpha=0.1
        if adapt:
            mutation_chance_A=alpha/(1-np.exp(-1))
            mutation_chance_B=-mutation_chance_A*np.exp(-1)
        if inform:
            percent_10=int(round(self.gens/10))
            self.fitness_per_generation=np.zeros(self.gens)
        stopbool=False
        while self.population.current_generation<self.gens and not stopbool:
            if inform and (self.population.current_generation+1)%percent_10==0:
                tic=time.time()
            self.population.crossover()
            #If <adapt> is true, the mutation chance drops exponentially per generation, starting at <alpha=0.1> and ending at 0.
            if adapt:
                self.population.mutate(mutation_chance=mutation_chance_A*np.exp(-self.population.current_generation/self.gens)+mutation_chance_B)
            else:
                self.population.mutate(mutation_chance=alpha)
            self.population.computefitness()
            if inform and (self.population.current_generation+1)%percent_10==0 and (self.population.current_generation!=generations):
                res_time=time.time()-tic
                print("Algorithm %0.1f percent complete. Remaining time is approximately %0.2f minutes"%(100*(self.population.current_generation+1)/generations, res_time*(generations-self.population.current_generation-1)/60))
                if np.std(self.population.fitness)<0.1:
                    stopbool=True
            if plot:
                self.fitness_per_generation[self.population.current_generation]=np.mean(self.population.fitness)
            self.population.current_generation+=1
        self.population.computefitness()
        print()    
        print("Best individual after %i generations with a fitness of %0.3f"%(self.population.current_generation,self.population.fitness[0]))
        print(self.population.genes[0])
        print("Average individual after %i generations with an average fitness of %0.3f"%(self.population.current_generation,np.mean(self.population.fitness)))
        print(np.average(self.population.genes,axis=0))
        print("Worst individual after %i generations with a fitness of %0.3f"%(self.population.current_generation,self.population.fitness[-1]))
        print(self.population.genes[-1])
        if plot:
            graphsize=7
            font = {"family": "serif",
            "color": "black",
            "weight": "bold",
            "size": "14"}
            plt.figure(1,figsize=(graphsize,graphsize))
            plt.plot([i for i in range(self.population.current_generation)],self.fitness_per_generation[:self.population.current_generation])
            plt.title("Average Fitness Function per Generation",fontdict=font)
            plt.xlabel("Generations",fontdict=font)
            plt.ylabel("Fitness Function",fontdict=font)
            plt.show()
        #If <return_result>, the algorithm returns the most fit gene.
        if return_result:
            return self.population.genes[0]
