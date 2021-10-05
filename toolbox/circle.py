import math
import numpy as np
import numpy.linalg as npla
from point import point
import matplotlib.colors as pltcolors

#This file contains the circle class and its various methods.
class circle(object):
    def __init__(self,center,radius):
        self.center=center
        self.radius=radius
        self.area=math.pi*self.radius**2
        self.circumference=2*math.pi*self.radius
        
    def is_circle(self,circVar,errtol=1e-12):
        #This method compares whether two circles are equal or not.
        return self.center.is_point(circVar.center,errtol=errtol) and abs(self.radius-circVar.radius)<=abs(errtol)
    
    def inCircle(self,pointVar,includeboundary=True,errtol=1e-12):
        #This method checks if an inputted point is in the circle or not.
        p1=point(self.radius+self.center.x,self.center.y)
        p2=point(self.radius-self.center.x,self.center.y)
        p3=point(self.center.x,self.radius+self.center.y)
        if (p2.x-p1.x)*(p3.y-p2.y)-(p2.y-p1.y)*(p3.x-p2.x)>0:
            res=npla.det([[p1.x-pointVar.x,p2.x-pointVar.x,p3.x-pointVar.x],
                          [p1.y-pointVar.y,p2.y-pointVar.y,p3.y-pointVar.y],
                          [(p1.x-pointVar.x)**2+(p1.y-pointVar.y)**2,
                           (p2.x-pointVar.x)**2+(p2.y-pointVar.y)**2,
                           (p3.x-pointVar.x)**2+(p3.y-pointVar.y)**2]])
        else:
            res=npla.det([[p1.x-pointVar.x,p3.x-pointVar.x,p2.x-pointVar.x],
                          [p1.y-pointVar.y,p3.y-pointVar.y,p2.y-pointVar.y],
                          [(p1.x-pointVar.x)**2+(p1.y-pointVar.y)**2,
                           (p3.x-pointVar.x)**2+(p3.y-pointVar.y)**2,
                           (p2.x-pointVar.x)**2+(p2.y-pointVar.y)**2]])
        if includeboundary:
            return res>=-abs(errtol)
        else:
            return res>abs(errtol)
        
    def __repr__(self):
        #This method prints the circle object.
        return "Circle <"+self.center.__repr__()+", %0.3f>"%self.radius
        
    def copy(self):
        #This method returns an exact copy of the circle object.
        return circle(self.center.copy(),self.radius)
    
    def draw(self,plotaxis,center=True,color="black",alpha=1):
        #This method plots the circle object into an inputted figure axis object. 
        theta=np.linspace(0,2*math.pi,100)
        if sum(pltcolors.to_rgb(color))<=1.0:
            plotaxis.fill(self.radius*np.cos(theta)+self.center.x,self.radius*np.sin(theta)+self.center.y,
                          facecolor=color,edgecolor="white",alpha=alpha)
            if center:
                plotaxis.scatter([self.center.x],[self.center.y],facecolor=color,edgecolor="white",alpha=alpha,zorder=1)
        else:
            plotaxis.fill(self.radius*np.cos(theta)+self.center.x,self.radius*np.sin(theta)+self.center.y,
                          facecolor=color,edgecolor="black",alpha=alpha)
            if center:
                plotaxis.scatter([self.center.x],[self.center.y],facecolor=color,edgecolor="black",alpha=alpha,zorder=1)
            
    def kill(self,C=None):
        #This method removes the circle object from any inputted list, then deletes itself.
        if isinstance(C,list):
            try:
                C.remove(self)
            except ValueError:
                pass
        del self
