import numpy as np
import matplotlib.pyplot as plt

class point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

class quadbox(object):
    def __init__(self,trunk,lower_bound,upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.center_x = (self.lower_bound[0]+self.upper_bound[0])/2.0
        self.center_y = (self.lower_bound[1]+self.upper_bound[1])/2.0
        self.trunk = trunk
        self.twigs = []
        self.data = []
        self.divided = False
        self.capacity = 1

    def __repr__(self):
        res = "Quadtree:\n"
        res += str(self.lower_bound[0])+" <= x <= "+str(self.upper_bound[0])+"\n"
        res += str(self.lower_bound[1])+" <= y <= "+str(self.upper_bound[1])+"\n"
        if self.divided:
            res += "Divided"
        else:
            res += "Number of points: "+str(len(self.data))
        return res

    def split(self):
        self.twigs = [quadbox(self,[self.lower_bound[0],self.lower_bound[1]],[self.center_x,self.center_y]),
                     quadbox(self,[self.lower_bound[0],self.center_y],[self.center_x,self.upper_bound[1]]),
                     quadbox(self,[self.center_x,self.lower_bound[1]],[self.upper_bound[0],self.center_y]),
                     quadbox(self,[self.center_x,self.center_y],[self.upper_bound[0],self.upper_bound[1]])]
        for twig in self.twigs:
            twig.capacity = self.capacity
        for p in self.data:
            if self.lower_bound[0] <= p.x < self.center_x:
                if self.lower_bound[1] <= p.y < self.center_y:
                    self.twigs[0].data.append(p)
                else:
                    self.twigs[1].data.append(p)
            else:
                if self.lower_bound[1] <= p.y < self.center_y:
                    self.twigs[2].data.append(p)
                else:
                    self.twigs[3].data.append(p)
        self.divided = True
        self.data = []

    def insert_point(self,point):
        if len(self.data) == self.capacity:
            self.split()
        if self.divided:
            if self.lower_bound[0] <= point.x < self.center_x:
                if self.lower_bound[1] <= point.y < self.center_y:
                    self.twigs[0].insert_point(point)
                else:
                    self.twigs[1].insert_point(point)
            else:
                if self.lower_bound[1] <= point.y < self.center_y:
                    self.twigs[2].insert_point(point)
                else:
                    self.twigs[3].insert_point(point)
        else:
            self.data.append(point)

    def clear(self):
        if self.divided:
            for twig in self.twigs:
                twig.clear()
            self.twigs = []
            self.divided = False
        else:
            self.data = []

    def query(self,lower_bound,upper_bound,query_result):
        if not (self.upper_bound[0] < lower_bound[0] or
            self.lower_bound[0] > upper_bound[0] or
            self.upper_bound[1] < lower_bound[1] or
            self.lower_bound[1] > upper_bound[1]):
            if self.divided:
                for twig in self.twigs:
                    twig.query(lower_bound,upper_bound,query_result)
            else:
                for p in self.data:
                    if (lower_bound[0] <= p.x <= upper_bound[0] and
                        lower_bound[1] <= p.y <= upper_bound[1]):
                        query_result.append(p)

    def plot(self,plotaxis):
        if self.divided:
            for twig in self.twigs:
                twig.plot()
        else:
            plotaxis.plot([self.lower_bound[0],self.upper_bound[0],
                       self.upper_bound[0],self.lower_bound[0],
                       self.lower_bound[0]],
                      [self.lower_bound[1],self.lower_bound[1],
                       self.upper_bound[1],self.upper_bound[1],
                       self.lower_bound[1]])

class quadtree(object):
    def __init__(self,lower_bound,upper_bound,capacity = 1):
        self.boxes = quadbox(None,lower_bound,upper_bound)
        try:
            self.boxes.capacity = max(int(round(capacity)),1)
        except Exception:
            self.boxes.capacity = 1
    
    def load_data(self,x,y):
        try:
            x = np.asarray(x)
            y = np.asarray(y)
            for i in range(min(len(x),len(y))):
                if x[i] < self.boxes.lower_bound[0]:
                    self.boxes.lower_bound[0] = x[i]
                elif x[i] > self.boxes.upper_bound[0]:
                    self.boxes.upper_bound[0] = x[i]
                if y[i] < self.boxes.lower_bound[1]:
                    self.boxes.lower_bound[1] = y[i]
                elif y[i] > self.boxes.upper_bound[1]:
                    self.boxes.upper_bound[1] = y[i]
                self.boxes.insert_point(point(x[i],y[i]))
        except Exception:
            print("Unable to load data into the quadtree")

    def delete_load(self):
        self.boxes.clear()
    
    def query(self,lower_bound,upper_bound):
        query_result = []
        self.boxes.query(lower_bound,upper_bound,query_result)
        return query_result

    def plot(self):
        plotaxis = plt.figure(figsize = (9,9)).add_subplot(111)
        self.boxes.plot(plotaxis)
