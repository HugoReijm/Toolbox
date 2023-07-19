import numpy as np
import matplotlib.colors as pltcolors

#This file contains the circle class and its various methods.
class circle(object):
    def __init__(self,center,radius):
        self.center=center
        self.radius=radius
        self.area=np.pi*self.radius**2
        self.circumference=2*np.pi*self.radius
        self.index=-1
        
    def is_circle(self,c,errtol=1e-12):
        #This method compares whether two circles are equal or not.
        return self.center.is_point(c.center,errtol=errtol) and abs(self.radius-c.radius)<=abs(errtol)
    
    def inCircle(self,p,includeboundary=True,errtol=1e-12):
        #This method checks if an inputted point is in the circle or not.
        if includeboundary:
            return p.distance(self.center)<=self.radius+abs(errtol)
        elif not includeboundary:
            return p.distance(self.center)<self.radius-abs(errtol)
        return False

    def __repr__(self):
        #This method returns a string representation of the circle object.
        return "Circle <"+self.center.__repr__()+", %0.6f>"%self.radius
        
    def copy(self):
        #This method returns an exact copy of the circle object.
        return circle(self.center.copy(),self.radius)
    
    def draw(self,plotaxis,color="black",alpha=1):
        #This method plots the circle object into an inputted figure axis object. 
        theta=np.linspace(0,2*np.pi,100)
        if sum(pltcolors.to_rgb(color))<=1.0:
            plotaxis.fill(self.radius*np.cos(theta)+self.center.x,self.radius*np.sin(theta)+self.center.y,
                          facecolor=color,edgecolor="white",alpha=alpha)
        else:
            plotaxis.fill(self.radius*np.cos(theta)+self.center.x,self.radius*np.sin(theta)+self.center.y,
                          facecolor=color,edgecolor="black",alpha=alpha)
