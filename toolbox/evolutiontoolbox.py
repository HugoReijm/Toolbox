import random
import numpy as np
import matplotlib.pyplot as plt
import time
import toolbox.generaltoolbox as gtb

class population:
    def __init__(self,size,size_of_gene,number_of_offspring,fitnessfunction,minInputs,maxInputs,integer_values_bool=False,pseudo_random_bool=False):
        self.size=size
        self.size_of_gene=size_of_gene
        self.maxInputs=maxInputs.copy()
        self.minInputs=minInputs.copy()
        self.intValBool=integer_values_bool
        if self.intValBool:
            self.genes=[np.array([random.randint(int(round(minInputs[j])),int(round(maxInputs[j]))) for j in range(self.size_of_gene)]) for i in range(self.size)]
        else:
            if pseudo_random_bool:
                hammersley=np.array(gtb.hammersley(self.size,self.size_of_gene,points=True))
                self.genes=[self.minInputs+hammersley[i]*(self.maxInputs-self.minInputs) for i in range(self.size)]
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
        self.fitness=np.array([self.fitnessfunction(self.genes[i]) for i in range(self.size)])
        self.genes=[gene for _,gene in sorted(zip(self.fitness,self.genes),key=lambda pair:pair[0],reverse=True)]
        total=sum(self.fitness)
        if total==0:
            self.fitnessratio=np.array([i/self.size for i in range(self.size+1)])
        else:
            self.fitnessratio=np.array([sum(self.fitness[:i])/total for i in range(self.size+1)])
    
    def roulette_selection(self):
        p=random.random()
        gindex=0
        for i in range(self.size):
            if self.fitnessratio[i]<=p and p<=self.fitnessratio[i+1]:
                gindex=i
                break
        return gindex
    
    def tournament_selection(self):
        k=min(int(self.size/2),1)
        indexArray=[random.randint(0,self.size-1) for i in range(k)]
        gindex=indexArray[0]
        for i in range(1,k):
            if self.fitnessfunction(self.genes[indexArray[i]])>self.fitnessfunction(self.genes[gindex]):
                gindex=indexArray[i]
        return gindex
    
    def crossover(self):
        if self.number_of_offspring>0:
            new_genes=[]
            while len(new_genes)<self.number_of_offspring:
                g1index=self.tournament_selection()
                if len(new_genes)==self.number_of_offspring-1 or self.size_of_gene==1:
                    new_genes.append(self.genes[g1index].copy())
                else:
                    g2index=self.tournament_selection()    
                    p=random.randint(1,self.size_of_gene-1)
                    g1=np.hstack((self.genes[g1index][:p],self.genes[g2index][p:])).copy()
                    g2=np.hstack((self.genes[g2index][:p],self.genes[g1index][p:])).copy()
                    new_genes.append(g1.copy())
                    new_genes.append(g2.copy())
                                 
            if self.number_of_offspring<self.size:
                self.genes=new_genes+self.genes[:self.size-self.number_of_offspring]
            else:
                self.genes=new_genes
    
    def mutate(self,mutation_chance=0.1):
        for i in range(self.size):
            if random.random()<mutation_chance:
                p=random.randint(0,self.size_of_gene-1)
                if self.intValBool:
                    self.genes[i][p]=random.randint(round(self.minInputs[p]),round(self.maxInputs[p]))
                else:
                    self.genes[i][p]=self.minInputs[p]+random.random()*(self.maxInputs[p]-self.minInputs[p])
            
class evoAlg:
    def __init__(self,function,populationSize,geneLength,offspringNumber,minInputs,maxInputs,integer_values_bool=False,pseudo_random_bool=False):
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
        self.gens=generations
        self.population.current_generation=0
        self.fitness_per_generation=np.zeros(self.gens)
        stop_bool=False
        tracker_bool=False
        tracker_gen=0
        alpha=0.1
        if adapt:
            mutation_chance_A=alpha/(1-np.exp(-1))
            mutation_chance_B=-mutation_chance_A*np.exp(-1)
        while self.population.current_generation<self.gens and not stop_bool:
            if inform and (self.population.current_generation+1)%10==0:
                tic=time.time()
            self.population.crossover()
            if adapt:
                self.population.mutate(mutation_chance=mutation_chance_A*np.exp(-self.population.current_generation/self.gens)+mutation_chance_B)
            else:
                self.population.mutate(mutation_chance=alpha)
            self.population.computefitness()
            if inform and (self.population.current_generation+1)%10==0 and (self.population.current_generation!=generations):
                res_time=time.time()-tic
                print("Algorithm %0.1f percent complete. Remaining time is approximately %0.1f minutes"%(100*(self.population.current_generation+1)/generations, res_time*(generations-self.population.current_generation-1)/60))
            self.fitness_per_generation[self.population.current_generation]=np.mean(self.population.fitness)
            if self.population.current_generation>0 and not tracker_bool and self.fitness_per_generation[self.population.current_generation]==self.fitness_per_generation[self.population.current_generation-1]:
                tracker_bool=True
                tracker_gen=self.population.current_generation
            if tracker_bool and self.population.current_generation>=tracker_gen+10:
                if self.fitness_per_generation[self.population.current_generation]==self.fitness_per_generation[tracker_gen]:
                    stop_bool=True
                    print("Algorithm has reached convergence after %i generations"%(self.population.current_generation+1))
                else:
                    tracker_bool=False
            self.population.current_generation+=1
        print()    
        print("Best individual after %i generations"%self.population.current_generation)
        bstInd=0
        for i in range(len(self.population.genes)):
            if self.population.fitness[i]>self.population.fitness[bstInd]:
                bstInd=i
        print(self.population.genes[bstInd])
        if not self.intValBool:
            print("Average individual after %i generations"%self.population.current_generation)
            avInd=np.zeros((self.population.size_of_gene,))
            for gn in self.population.genes:
                avInd+=gn
            print(avInd/self.population.size)
        print("Worst individual after %i generations"%self.population.current_generation)
        wrstInd=0
        for i in range(len(self.population.genes)):
            if self.population.fitness[i]<self.population.fitness[wrstInd]:
                wrstInd=i
        print(self.population.genes[wrstInd])
        print()
        print("Average fitness after %i generations"%self.population.current_generation)
        print(sum(self.population.fitness)/self.population.size)
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
        if return_result:
            return self.population.genes[bstInd]
