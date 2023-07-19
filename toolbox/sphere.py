import numpy as np
import matplotlib.colors as pltcolors

class sphere(object):
    def __init__(self,center,radius):
        self.center=center
        self.radius=radius
        self.volume=4*np.pi*self.radius**3/3
        self.surface_area=4*np.pi*self.radius**2
        self.index=-1
        
    def is_sphere(self,s,errtol=1e-12):
        #This method compares whether two spheres are equal or not.
        return self.center.is_point(s.center,errtol=errtol) and abs(self.radius-s.radius)<=abs(errtol)
    
    def inSphere(self,p,includeboundary=True,errtol=1e-12):
        #This method checks if an inputted point is in the circle or not.
        if includeboundary:
            return p.distance(self.center)<=self.radius+abs(errtol)
        else:
            return p.distance(self.center)<self.radius-abs(errtol)
        
    def __repr__(self):
        #This method returns a string representation of the circle object.
        return "Sphere <"+self.center.__repr__()+", %0.6f>"%self.radius
        
    def copy(self):
        #This method returns an exact copy of the circle object.
        return sphere(self.center.copy(),self.radius)
    
    def draw(self,plotaxis,elev=10,azim=-75,color="black",alpha=1):
        #This method plots the circle object into an inputted figure axis object. 
        u = np.linspace(0,2*np.pi,20)
        v = np.linspace(0,np.pi,10)
        x = np.outer(np.cos(u),np.sin(v))
        y = np.outer(np.sin(u),np.sin(v))
        z = np.outer(np.ones(np.size(u)),np.cos(v))
        if sum(pltcolors.to_rgb(color))<=1.0:
            plotaxis.plot_surface(self.radius*x+self.center.x,self.radius*y+self.center.y,self.radius*z+self.center.z,
                          color=color,edgecolor="white",alpha=alpha)
        else:
            plotaxis.plot_surface(self.radius*x+self.center.x,self.radius*y+self.center.y,self.radius*z+self.center.z,
                          color=color,edgecolor="black",alpha=alpha)
        plotaxis.view_init(elev=elev,azim=azim)
