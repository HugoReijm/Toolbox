import numpy as np
import numpy.linalg as npla

#This file contains the point class and its various methods.
class point(object):
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
        #This list keeps track of which points are connects to this one via an edge.
        self.neighbors=[]
        #This list keeps track of which edges the point object is a constituent of.
        self.edges=[]
        #This list keeps track of which triangles the point object is a constituent of.
        self.triangles=[]
        self.tetrahedra=[]
        self.index=-1
        self.value=None
    
    def is_point(self,p,errtol=1e-12):
        #This method compares whether two points are equal or not.
        #Absolute values are apparently slow in Python
        errtol=abs(errtol)
        return (-errtol<=self.x-p.x<=errtol and -errtol<=self.y-p.y<=errtol and -errtol<=self.z-p.z<=errtol)
    
    def __add__(self,p):
        return point(self.x+p.x,self.y+p.y,self.z+p.z)
    
    def scalar_add(self,s):
        return point(self.x+s,self.y+s,self.z+s)
    
    def __sub__(self,p):
        return point(self.x-p.x,self.y-p.y,self.z-p.z)
    
    def scalar_sub(self,s):
        return point(self.x-s,self.y-s,self.z-s)
    
    def __mul__(self,p):
        return point(self.x*p.x,self.y*p.y,self.z*p.z)
    
    def scalar_mul(self,s):
        return point(self.x*s,self.y*s,self.z*s)
    
    def __truediv__(self,p):
        return point(self.x/p.x,self.y/p.y,self.z/p.z)
    
    def scalar_div(self,s):
        return point(self.x/s,self.y/s,self.z/s)
    
    def distance(self,p):
        #This method computes the distance between the point object and the other inputted point.
        return npla.norm(np.array([self.x-p.x,self.y-p.y,self.z-p.z]))
    
    def __repr__(self,round_off_digit=6):
        #This method returns a string representation of the point object.
        return "Point "+str(self.index)+": ("+str(round(self.x,round_off_digit))+", "+str(round(self.y,round_off_digit))+", "+str(round(self.z,round_off_digit))+")"

    def copy(self):
        #This method returns an exact copy of the point object.
        #The objects edge and triangle lists are not copied.
        return point(self.x,self.y,self.z)
    
    def draw(self,plotaxis,flat=True,color="black",alpha=1):
        #This method plots the point object into an inputted figure axis object. 
        if flat:
            plotaxis.scatter(self.x,self.y,color=color,alpha=alpha)
        else:
            plotaxis.scatter3D(self.x,self.y,self.z,color=color,alpha=alpha)
