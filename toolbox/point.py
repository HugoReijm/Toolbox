import math
import matplotlib.colors as pltcolors

#This file contains the point class and its various methods.
class point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.edges=[]
        self.triangles=[]
        self.index=-1
    
    def is_point(self,p,errtol=1e-12):
        #This method compares whether two points are equal or not.
        return isinstance(p,point) and abs(self.x-p.x)<abs(errtol) and abs(self.y-p.y)<abs(errtol)
    
    def distance(self,p):
        #This method computes the distance between the point object and the other inputted point.
        return math.sqrt((self.x-p.x)**2+(self.y-p.y)**2)
    
    def __repr__(self):
        #This method returns a string representation of the point object.
        return "Point ("+str(round(self.x,6))+", "+str(round(self.y,6))+")"
              
    def copy(self):
        #This method returns an exact copy of the point object.
        #The objects edge and triangle lists are not included.
        return point(self.x,self.y)
    
    def kill(self,P=None):
        #This method removes the point object from any of its edge's point lists.
        #This method also removes the point object from any of its triangle's point lists.
        #Finally this method removes the point object from any inputted list, then deletes itself.
        for e in self.edges:
            try:
                e.points.remove(self)
            except ValueError:
                pass
        for t in self.triangles:
            try:
                t.points.remove(self)
            except ValueError:
                pass
        if isinstance(P,list):
            try:
                P.remove(self)
            except ValueError:
                pass
        del self
