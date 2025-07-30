import numpy as np
import random
from collections import defaultdict

def kdTree(data):
    dim = len(data[0])
    
    #Node class for a K-D Tree
    class Node(object):
        def __init__(self,data):
            self.data = np.asarray(data)
            self.left = None
            self.right = None
            self.level = 0
        
        #Builds the tree from the root
        def insert(self,data):
            if data[self.level%dim] <= self.data[self.level%dim]:
                if self.left is None:
                    self.left = Node(data)
                    self.left.level = self.level + 1
                else:
                    self.left.insert(data)
            else:
                if self.right is None:
                    self.right = Node(data)
                    self.right.level = self.level + 1
                else:
                    self.right.insert(data)
        
        #Finds the nearest point for the inputted data
        def nearestNeighbor(self,data):
            if data[self.level%dim] <= self.data[self.level%dim]:
                nextNode = self.left
                otherNode = self.right
            else:
                nextNode = self.right
                otherNode = self.left
    
            if nextNode is not None:
                best = nextNode.nearestNeighbor(data)
                bestDist = np.sum((best.data-data)**2)
                optionalDist = np.sum((self.data-data)**2)
                if optionalDist < bestDist:
                    best = self
                    bestDist = optionalDist
            else:
                best = self
                bestDist = np.sum((self.data-data)**2)
    
            if otherNode is not None and bestDist >= (data[self.level%dim]-self.data[self.level%dim])**2:
                optional = otherNode.nearestNeighbor(data)
                optionalDist = np.sum((optional.data-data)**2)
                if optionalDist < bestDist:
                    best = optional
            
            return best
    
    random.shuffle(data)
    root = Node(data[0])
    for i in range(1,len(root)):
        root.insert(data[i])
    return root

def rTree(data,minPoints=5):
    dim = len(data[0])
    
    #Box class for a R-Tree
    class Rbox(object):
        def __init__(self,data,level = 0):
            self.lowerBound = np.nanmin(data,axis = 0)
            self.upperBound = np.nanmax(data,axis = 0)
            self.trunk = None
            self.twigs = []
            self.data = data
            self.divided = False
            self.level = level
            self.split()
        
        #Splits the box if needed into smaller boxes
        def split(self):
            if (len(self.data) > minPoints):
                self.data.sort(key = lambda value:value[self.level])
                deltaIndex = minPoints**(dim-self.level)
                for index in range(0,len(self.data),deltaIndex):
                    twig = Rbox(self.data[index:min(index+deltaIndex,len(data))],level=self.level+1)
                    twig.trunk = self
                    self.twigs.append(twig)
                self.data = []
                self.divided = True
        
        #Finds the points that fall into query box defined by
        #upper- and lower-bound arrays
        def query(self,lowerBound,upperBound):
            lowerBound = np.asarray(lowerBound)
            upperBound = np.asarray(upperBound)
            if not self.divided:
                return [value for value in self.data if all(lowerBound <= value) and all(value <= upperBound)]
            else:
                if any(self.upperBound < lowerBound) or any(self.lowerBound > upperBound):
                    return []
                else:
                    resultList = []
                    for twig in self.twigs:
                        resultList += twig.query(lowerBound,upperBound)
                    return resultList
        
        #Graphing capabilities added
        def graph(self,plotaxis):
            plotaxis.plot([self.lowerBound[0],self.upperBound[0],
                           self.upperBound[0],self.lowerBound[0],
                           self.lowerBound[0]],
                          [self.lowerBound[1],self.lowerBound[1],
                           self.upperBound[1],self.upperBound[1],
                           self.lowerBound[1]],color="black")
            for twig in self.twigs:
                twig.graph(plotaxis)
    
    root = Rbox(data)
    return root

def lsh(data,start,stop,num_tables=5,hash_size=10):
    try:
        dim = len(data[0])
    except TypeError:
        dim = 1
    class LSH:
        def __init__(self):
            self.hash_tables = [defaultdict(list) for _ in range(num_tables)]
            self.hyperplanes = [np.random.randn(hash_size,dim) for _ in range(num_tables)]
            self.biases = [np.random.uniform(low=start,high=stop,size=(hash_size,dim)) for _ in range(num_tables)]
            
        def _hash_function(self,point,hyperplanes,biases):
            #return tuple((np.dot(hyperplanes,point)>0).astype(int))
            return tuple((np.dot(hyperplanes[i],point-biases[i])>0).astype(int) for i in range(len(hyperplanes)))
        
        def insert(self,pointIndex,point):
            for i in range(num_tables):
                hash_key = self._hash_function(point,self.hyperplanes[i],self.biases[i])
                if dim < 2:
                    self.hash_tables[i][hash_key].append((pointIndex,(point[()],)))
                else:
                    self.hash_tables[i][hash_key].append((pointIndex,tuple(point)))

        def query(self,point,k=10):
            candidates = set()
            for i in range(num_tables):
                hash_key = self._hash_function(point,self.hyperplanes[i],self.biases[i])
                if hash_key in self.hash_tables[i]:
                    candidates.update(self.hash_tables[i][hash_key])
            candidates = list(candidates)
            if dim < 2:
                candidates.sort(key = lambda x:sum([(x[1]-point)**2 for i in range(dim)]))
            else:
                candidates.sort(key = lambda x:sum([(x[1][i]-point[i])**2 for i in range(dim)]))
            return [candidate[0] for candidate in candidates[:k]]
    
    lsh = LSH()
    for i in range(len(data)):
        lsh.insert(i,data[i])
    return lsh

def hnsw(data,connectivity=10):
    class HNSW(object):
        def __init__(self):
            self.connectivity = connectivity
            self.layers = []
            self.points = []
        
        def distance(self,a,b):
            return sum([(a[i]-b[i])**2 for i in range(len(a))])
        
        def insert(self,point):
            index = len(self.points)
            self.points.append(point)
                
            max_layer = int(np.log2(len(self.points)+1))
            level = int(-np.log(random.random())*max_layer/2)
            if level >= len(self.layers):
                level = len(self.layers)
                self.layers.append({0:set()})
            
            current_index = random.choice(list(self.layers[-1].keys()))
            current_dist = self.distance(self.points[current_index],point)
            for l in range(len(self.layers)-1,level,-1):
                searching_bool = True
                while searching_bool:
                    searching_bool = False
                    for neighbor in self.layers[l].get(current_index,[]):
                        neighbor_dist = self.distance(self.points[neighbor],point)
                        if neighbor_dist < current_dist:
                            current_index = neighbor
                            current_dist = neighbor_dist
                            searching_bool = True
            
            links = [(current_index,current_dist)]
            for l in range(level,-1,-1):
                self.layers[l][index] = set()
                candidates = minHeap(links)
                links = []
                while len(candidates.queue) > 0 and len(links) < self.connectivity:
                    current_index,current_dist = candidates.extractMin()
                    self.layers[l][index].add(current_index)
                    self.layers[l][current_index].add(index)
                    links.append((current_index,current_dist))
                    for neighbor in self.layers[l].get(current_index,[]):
                        neighbor_dist = self.distance(self.points[neighbor],point)
                        if (neighbor,neighbor_dist) not in links and neighbor not in candidates.nodeLookUp:
                            candidates.insert(neighbor,neighbor_dist)
    
        def query(self,point,k=1):
            if len(self.points) == 0:
                return []
            
            current_index = random.choice(list(self.layers[-1].keys()))
            current_dist = self.distance(self.points[current_index],point)
            for layer in range(len(self.layers)-1,-1,-1):
                searching_bool = True
                while searching_bool:
                    searching_bool = False
                    for neighbor in self.layers[layer].get(current_index,[]):
                        neighbor_dist = self.distance(self.points[neighbor],point)
                        if neighbor_dist < current_dist:
                            current_index = neighbor
                            current_dist = neighbor_dist
                            searching_bool = True
            
            candidates = minHeap([current_index,current_dist])
            links = []
            while len(candidates.queue) > 0 and len(links) < k:
                current_index,current_dist = candidates.extractMin()
                links.append((current_index,current_dist))
                for neighbor in self.layers[0].get(current_index,[]):
                    neighbor_dist = self.distance(self.points[neighbor],point)
                    if (neighbor,neighbor_dist) not in links and neighbor not in candidates.nodeLookUp:
                        candidates.insert(neighbor,neighbor_dist)
            links.sort(key = lambda x:x[1])
            return [self.points[link[0]] for link in links[:k]]
        
    graph = HNSW()
    for value in data:
        graph.insert(value)
    return graph

def minHeap(data):
    class minQueue(object):
        def __init__(self):
            self.queue = []
            self.nodeLookUp = {}
            
        def swap(self,i,j):
            self.queue[i],self.queue[j] = self.queue[j],self.queue[i]
            self.nodeLookUp[self.queue[i][0]] = i
            self.nodeLookUp[self.queue[j][0]] = j
    
        def insert(self,value,priority,extra=None):
            self.queue.append([value,priority,extra])
            i = len(self.queue) - 1
            self.nodeLookUp[self.queue[i][0]] = i
            while i > 0 and self.queue[(i-1)//2][1] > self.queue[i][1]:
                self.swap(i,(i-1)//2)
                i = (i-1)//2
                    
        def extractMin(self):
            self.swap(0,-1)
            res = self.queue.pop(-1)
            i = 0
            while True:
                i1 = 2*i+1
                i2 = 2*i+2
                if i2 < len(self.queue):
                    if self.queue[i][1] > self.queue[i1][1]:
                        if self.queue[i][1] > self.queue[i2][1]:
                            if self.queue[i1][1] < self.queue[i2][1]:
                                self.swap(i,i1)
                                i = i1
                            else:
                                self.swap(i,i2)
                                i = i2
                        else:
                            self.swap(i,i1)
                            i = i1
                    elif self.queue[i][1] > self.queue[i2][1]:
                        self.swap(i,i2)
                        i = i2
                    else:
                        break
                elif i1 < len(self.queue):
                    if self.queue[i][1] > self.queue[i1][1]:
                        self.swap(i,i1)
                        i = i1
                    else:
                        break
                else:
                    break
            if res[2] is None:
                return res[0],res[1]
            else:
                return res[0],res[1],res[2]
        
        def decreaseKey(self,value,newPriority,newExtra=None):
            try:
                index = self.nodeLookUp[value]
                if newPriority < self.queue[index][1]:
                    self.queue[index][1] = newPriority
                    self.queue[index][2] = newExtra
                    while index > 0 and self.queue[(index-1)//2][1] > self.queue[index][1]:
                        self.swap(index,(index-1)//2)
                        index = (index-1)//2
            except Exception:
                pass
        
        def __repr__(self):
            return str(self.queue)
    
    priorityQueue = minQueue()
    for value in data:
        if isinstance(value[1],int) or isinstance(value[1],float):
            if isinstance(value[0],int) or isinstance(value[0],float):
                priorityQueue.insert(value[0],value[1],extra=value[2])
            else:
                priorityQueue.insert(tuple(value[0]),value[1],extra=value[2])
    return priorityQueue