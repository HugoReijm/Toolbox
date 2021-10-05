import math
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import random
from itertools import product

class swarm:
    def __init__(self,gridsect,dim,fitnessfunction,maxInputs,minInputs,inertiaConst=1.0,cognativeConst=1.0,socialConst=1.0,localRange=-1.0):
        self.dim=max(dim,0)
        self.fitnessfunction=fitnessfunction
        self.maxInputs=maxInputs.copy()
        self.minInputs=minInputs.copy()
        
        self.size=np.asscalar(np.prod(gridsect+1))
        grids=[[minInputs[i]+j*(maxInputs[i]-minInputs[i])/gridsect[i] for j in range(round(np.asscalar(gridsect[i]+1)))] for i in range(self.dim)]
        self.locations=[np.asarray(gridpoint) for gridpoint in product(*grids)]   
        
        self.velocities=[np.zeros((self.dim,)) for i in range(self.size)]
        self.personalbests=[self.locations[i].copy() for i in range(self.size)]
        self.inertiaConst=inertiaConst
        self.cognConst=cognativeConst
        self.socialConst=socialConst
        self.localRange=localRange
    
    def move(self,inertiaConst,cognConst,socialConst):
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

class swarmAlg:
    def __init__(self,gridSections,dim,fitnessFunction,maxInputs,minInputs,inertiaConst=1.0,cognativeConst=1.0,socialConst=1.0,localRange=-1.0):
        if "ndarray" not in type(gridSections).__name__:
            self.gridSect=np.array(gridSections)
        else:
            self.gridSect=gridSections
        self.dim=dim
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
        self.swarm=swarm(self.gridSect,self.dim,self.f,self.maxInputs,self.minInputs,inertiaConst=self.inertiaConst,cognativeConst=self.cognConst,socialConst=self.socialConst,localRange=self.localRange)
        
    def run(self,steps,plot=False):
        best_fitness_per_step=[[self.f(self.swarm.locations[i])] for i in range(self.swarm.size)]
        locations_per_step=[[[self.swarm.locations[j][i]] for j in range(self.swarm.size)] for i in range(self.dim)]
        velocities_per_step=[[[self.swarm.velocities[j][i]] for j in range(self.swarm.size)] for i in range(self.dim)]
        count=0
        while count<steps:
            self.swarm.move(self.inertiaConst,self.cognConst,self.socialConst)
            for i in range(self.swarm.size):
                best_fitness_per_step[i].append(self.f(self.swarm.personalbests[i]))
            for i in range(self.dim):
                for j in range(self.swarm.size):
                    locations_per_step[i][j].append(self.swarm.locations[j][i])
                    velocities_per_step[i][j].append(self.swarm.velocities[j][i])
            count+=1
            
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
            
        if plot:
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
                plt.title("Locations (Coordinate %s) of Swarm Members"%(1+i),fontdict=font)
                plt.xlabel("Time",fontdict=font)
                plt.ylabel("Locations %s"%(1+i),fontdict=font)
            
            for i in range(self.dim):
                plt.figure(2+self.dim+i,figsize=(graphsize,graphsize))
                for j in range(self.swarm.size):
                    plt.plot([k for k in range(steps+1)],velocities_per_step[i][j])
                plt.title("Velocities (Coordinate %s) of Swarm Members"%(1+i),fontdict=font)
                plt.xlabel("Time",fontdict=font)
                plt.ylabel("Velocities %s"%(1+i),fontdict=font)
            
            plt.show()
