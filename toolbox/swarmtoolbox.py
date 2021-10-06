import math
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import random
import time
from itertools import product

#This file contains the classes swarm and swarmalg.
#Swarm contains all the variables and methods needed for one iteration of the Particle Swarm Optimization algorithm.
class swarm:
    def __init__(self,gridsect,fitnessfunction,minInputs,maxInputs,localRange):
        #This initialization method lets the user designate the following about the swarm:
        #<gridsect> is a list of numbers where the length of the list equals the dimension of the search space,
        #and each element in the list represents how many subdivisions the corresponding dimension should be divided into.
        #This then rosters the search space, where each subsection will be the initialization point for a particle. 
        #<fitnessfunction> determines the fitness of each particle's location.
        #<minInputs> determines the minimum value for each particle of the swarm.
        #<maxInputs> determines the maximum value for each particle of the swarm.
        #<localRange> determines the range two particles have to be in for data exchange to occur.
        #Default is set to -1 to indicate a global relaying of information.
        self.dim=len(gridsect)
        self.fitnessfunction=fitnessfunction
        self.maxInputs=maxInputs.copy()
        self.minInputs=minInputs.copy()
        
        self.size=np.asscalar(np.prod(gridsect+1))
        grids=[[minInputs[i]+j*(maxInputs[i]-minInputs[i])/gridsect[i] for j in range(round(np.asscalar(gridsect[i]+1)))] for i in range(self.dim)]
        self.locations=[np.asarray(gridpoint) for gridpoint in product(*grids)]   
        
        self.velocities=[np.zeros((self.dim,)) for i in range(self.size)]
        self.personalbests=[self.locations[i].copy() for i in range(self.size)]
        self.localRange=localRange
    
    def move(self,inertiaConst,cognConst,socialConst):
        #This method moves each particle by taking the weighted sum of three vectors.
        #The first vector is pointed in the current direction of the particle's movement. It is multiplied by the inertial constant inertiaConst.
        #The second vector is pointed toward's the particle's best solution it has found so far. It is multiplied by the cognative constant cognativeConst.
        #The third vector is pointed toward's the (local) swarm's best solution it has found so far. It is multiplied by the social constant socialConst.
        #Personal and global bests are updated afterwards if needed.
        if self.localRange<0:
            tempbest=self.personalbests[0].copy()
            for i in range(1,self.size):
                if self.fitnessfunction(self.personalbests[i])>self.fitnessfunction(tempbest):
                    tempbest=self.personalbests[i].copy()
            self.bests=[tempbest for i in range(self.size)]
        else:
            self.bests=[]
            for i in range(self.size):
                self.bests.append(self.personalbests[i].copy())
                for j in range(self.size):
                    if i!=j:
                        if np.sum((self.locations[i]-self.locations[j])**2)<=self.localRange**2:
                            if self.fitnessfunction(self.personalbests[j])>self.fitnessfunction(self.bests[i]):
                                self.bests[i]=self.personalbests[j].copy()

        for i in range(self.size):
            self.velocities[i]=(inertiaConst*self.velocities[i]
                                +cognConst*random.random()*(self.personalbests[i]-self.locations[i])  
                                +socialConst*random.random()*(self.bests[i]-self.locations[i]))
            self.locations[i]+=self.velocities[i]
            if self.fitnessfunction(self.locations[i])>self.fitnessfunction(self.personalbests[i]):
                self.personalbests[i]=self.locations[i].copy()

#The swarmAlg class contains all the variables and methods needed to run the multiple iterations of the Particle Swarm Optimization Algorithm.
class swarmAlg:
    def __init__(self,gridSections,fitnessFunction,minInputs,maxInputs,inertiaConst=0.9,cognativeConst=2.0,socialConst=2.0,localRange=-1.0):
        #This initialization method acts as the user interface object and lets the user designate the following about the swarm:
        #<gridsect> is a list of numbers where the length of the list equals the dimension of the search space,
        #and each element in the list represents how many subdivisions the corresponding dimension should be divided into.
        #This then rosters the search space, where each subsection will be the initialization point for a particle. 
        #<fitnessfunction> determines the fitness of each particle's location.
        #<minInputs> determines the minimum value for each particle of the swarm.
        #<maxInputs> determines the maximum value for each particle of the swarm.
        #<inertialConst>, <cognativeConst>, and <socialConst> determine the inertial constant, the cognative constant, and the social constant, respectively
        #<localRange> determines the range two particles have to be in for data exchange to occur.
        #Default is set to -1 to indicate a global relaying of information.
        if "ndarray" not in type(gridSections).__name__:
            self.gridSect=np.array(gridSections)
        else:
            self.gridSect=gridSections
        self.dim=len(gridSections)
        self.f=fitnessFunction
        if "ndarray" not in type(maxInputs).__name__:
            self.maxInputs=np.array(maxInputs)
        else:
            self.maxInputs=maxInputs
        if "ndarray" not in type(minInputs).__name__:
            self.minInputs=np.array(minInputs)
        else:
            self.minInputs=minInputs
        self.inertiaConst=inertiaConst
        self.cognConst=cognativeConst
        self.socialConst=socialConst
        self.localRange=localRange
        self.swarm=swarm(self.gridSect,self.f,self.minInputs,self.maxInputs,localRange=self.localRange)
        
    def run(self,steps,inform=True,adapt=True,return_result=True):
        #This method manages the multiple iterations of the particle swarm optimization algorithm.
        #The number of steps the algorithm is allowed to move the swarm is user-designated.
        #If <inform> is true, the algorithm interacts with the user to inform them of progress and results.
        best_fitness_per_step=[[self.f(self.swarm.locations[i])] for i in range(self.swarm.size)]
        locations_per_step=[[[self.swarm.locations[j][i]] for j in range(self.swarm.size)] for i in range(self.dim)]
        velocities_per_step=[[[self.swarm.velocities[j][i]] for j in range(self.swarm.size)] for i in range(self.dim)]
        count=0
        if adapt:
            inertiaConst_A=-self.inertiaConst/steps
            inertiaConst_B=self.inertiaConst
        while count<steps:
            if inform and (count+1)%10==0:
                tic=time.time()
            #If <adapt> is true, the inertial constant drops linearly per step, starting at <self.inertiaConst> and ending at 0.
            if adapt:
                self.swarm.move(inertiaConst_A*count+inertiaConst_B,self.cognConst,self.socialConst)
            else:
                self.swarm.move(self.inertiaConst,self.cognConst,self.socialConst)
            for i in range(self.swarm.size):
                best_fitness_per_step[i].append(self.f(self.swarm.personalbests[i]))
            for i in range(self.dim):
                for j in range(self.swarm.size):
                    locations_per_step[i][j].append(self.swarm.locations[j][i])
                    velocities_per_step[i][j].append(self.swarm.velocities[j][i])
            count+=1
            if inform and count%10==0 and count!=steps:
                res_time=time.time()-tic
                print("Algorithm %0.1f percent complete. Remaining time is approximately %0.1f minutes"%(100*count/steps, res_time*(steps-count)/60))
            
        globalbest=self.f(self.swarm.bests[0])
        isGlobalBest=[True]+[False for i in range(1,self.swarm.size)]
        for i in range(1,self.swarm.size):
            temp=self.f(self.swarm.bests[i])
            if temp>globalbest:
                globalbest=temp
                for j in range(i):
                    isGlobalBest[j]=False
                isGlobalBest[i]=True
            elif temp==globalbest:
                isGlobalBest[i]=True
        
        boolarray=[(self.localRange>=0) for i in range(self.swarm.size)]
        for i in range(self.swarm.size):
            for j in range(i+1,self.swarm.size):
                if self.localRange>=0:
                    if npla.norm(self.swarm.bests[i]-self.swarm.bests[j])<self.localRange:
                        boolarray[j]=False
                else:
                    if self.f(self.swarm.bests[i])>self.f(self.swarm.bests[j]):
                        boolarray[i]=True
                        boolarray[j]=False
                    else:
                        boolarray[i]=False
                        boolarray[j]=True
                            
        if inform:
            print("Best Solutions:")
            for i in range(self.swarm.size):
                if boolarray[i]:
                    strng="["
                    for j in range(self.dim-1):
                        strng+="%0.9f, "%self.swarm.bests[i][j]
                    strng+="%0.9f]"%self.swarm.bests[i][self.dim-1]
                    strng+=" with a fitness value of %0.9f"%self.f(self.swarm.bests[i])
                    if isGlobalBest[i]:
                        strng+=" (Global Best)"
                    print(strng)
            
            graphsize=7
            font = {"family": "serif",
            "color": "black",
            "weight": "bold",
            "size": "14"}
            plt.figure(1,figsize=(graphsize,graphsize))
            for i in range(self.swarm.size):
                plt.plot([j for j in range(steps+1)],best_fitness_per_step[i])
            plt.title("Best Fitness of Swarm Members",fontdict=font)
            plt.xlabel("Time",fontdict=font)
            plt.ylabel("Fitness",fontdict=font)
            
            for i in range(self.dim):
                plt.figure(2+i,figsize=(graphsize,graphsize))
                for j in range(self.swarm.size):
                    plt.plot([k for k in range(steps+1)],locations_per_step[i][j])
                plt.title("Dimension (Coordinate %s) of Swarm Members"%(1+i),fontdict=font)
                plt.xlabel("Time",fontdict=font)
                plt.ylabel("Locations %s"%(1+i),fontdict=font)
            
            plt.show()
        
        #If <return_result>, the algorithm results the most fit solutions the particle found.
        if return_result:
            return [self.swarm.bests[i] for i in range(len(self.swarm.bests)) if boolarray[i]]
