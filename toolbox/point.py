import numpy as np

#This file contains the point class and its various methods.
class point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        #This list keeps track of which edges the point object is a constituent of.
        self.edges=[]
        #This list keeps track of which triangles the point object is a constituent of.
        self.triangles=[]
        self.index=-1
        self.value=None
    
    def is_point(self,p,errtol=1e-12):
        #This method compares whether two points are equal or not.
        return isinstance(p,point) and abs(self.x-p.x)<abs(errtol) and abs(self.y-p.y)<abs(errtol)
    
    def __add__(self,p):
        return point(self.x+p.x,self.y+p.y)
    
    def __sub__(self,p):
        return point(self.x-p.x,self.y-p.y)
    
    def __mul__(self,p):
        return point(self.x*p.x,self.y*p.y)
    
    def __truediv__(self,p):
        return point(self.x/p.x,self.y/p.y)
    
    def distance(self,p):
        #This method computes the distance between the point object and the other inputted point.
        return np.sqrt((self.x-p.x)**2+(self.y-p.y)**2)
    
    def __repr__(self):
        #This method returns a string representation of the point object.
        return "Point "+str(self.index)+": ("+str(round(self.x,6))+", "+str(round(self.y,6))+")"
              
    def copy(self):
        #This method returns an exact copy of the point object.
        #The objects edge and triangle lists are not copied.
        return point(self.x,self.y)
