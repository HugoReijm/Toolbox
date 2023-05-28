import numpy as np
from toolbox.point import point
from toolbox.sphere import sphere
import matplotlib.colors as pltcolors

#This file contains the edge class and its various methods.
class edge(object):
    def __init__(self,p1,p2):
        #This list keeps track of which points constitute the end-points of the edge object.
        self.points=[p1,p2]
        self.edgeAv=None
        #This list keeps track of which triangles the edge object is a constituent of.
        self.triangles=[]
        self.edgeLength=None
        self.enclosed=True
        self.constraint=False
        self.index=-1
        self.circumSphere=None
        
    def is_edge(self,e,errtol=1e-12):
        #This method compares whether two edges are equal or not.
        for i in range(2):
            if self.points[0].is_point(e.points[i],errtol=errtol):
                for j in range(2):
                    if j!=i and self.points[1].is_point(e.points[j],errtol=errtol):
                        return True
        return False
        
    def update(self):
        #For each end-point of the edge object, this method adds the edge to the point's list of edges it is a constituent of.
        for p in self.points:
            p.edges.append(self)
        
    def average(self):
        #This method computes the centroid of the triangle object and makes the centroid variable a singleton.
        if self.edgeAv is None:
            self.edgeAv=self.points[0]+self.points[1]
            self.edgeAv.x/=2.0
            self.edgeAv.y/=2.0
            self.edgeAv.z/=2.0
        return self.edgeAv
    
    def length(self):
        #This method computes the length of the edge object and makes the length variable a singleton.
        if self.edgeLength is None:
            self.edgeLength=self.points[0].distance(self.points[1])
        return self.edgeLength
    
    def is_longer_than(self,dist,or_equal_to=False):
        #This method computes whether the edge is longer (or equal to) a certain
        #distance without using a square root function (which isnotoriously slow)
        if or_equal_to:
            return (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2+(self.points[1].z-self.points[0].z)**2>=dist
        else:
            return (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2+(self.points[1].z-self.points[0].z)**2>dist
        
    def is_shorter_than(self,dist,or_equal_to=False):
        #This method computes whether the edge is shorter (or equal to) a certain
        #distance without using a square root function (which isnotoriously slow)
        if or_equal_to:
            return (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2+(self.points[1].z-self.points[0].z)**2<=dist
        else:
            return (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2+(self.points[1].z-self.points[0].z)**2<dist
    
    def dot(self,e):
        #This method returns the 2-dimensional dot product of the edge object and another inputted edge.
        if self.points[0].is_point(e.points[0]):
            return ((self.points[1].x-self.points[0].x)*(e.points[1].x-e.points[0].x)
                    +(self.points[1].y-self.points[0].y)*(e.points[1].y-e.points[0].y)
                    +(self.points[1].z-self.points[0].z)*(e.points[1].z-e.points[0].z))
        elif self.points[0].is_point(e.points[1]):
            return ((self.points[1].x-self.points[0].x)*(e.points[0].x-e.points[1].x)
                    +(self.points[1].y-self.points[0].y)*(e.points[0].y-e.points[1].y)
                    +(self.points[1].z-self.points[0].z)*(e.points[0].z-e.points[1].z))
        elif self.points[1].is_point(e.points[0]):
            return ((self.points[0].x-self.points[1].x)*(e.points[1].x-e.points[0].x)
                    +(self.points[0].y-self.points[1].y)*(e.points[1].y-e.points[0].y)
                    +(self.points[0].z-self.points[1].z)*(e.points[1].z-e.points[0].z))
        elif self.points[1].is_point(e.points[1]):
            return ((self.points[0].x-self.points[1].x)*(e.points[0].x-e.points[1].x)
                    +(self.points[0].y-self.points[1].y)*(e.points[0].y-e.points[1].y)
                    +(self.points[0].z-self.points[1].z)*(e.points[0].z-e.points[1].z))
        return None
    
    def cross(self,e):
        #This method returns the cross product of the edge object and another inputted edge.
        if self.points[0].is_point(e.points[0]):
            cross_x=(self.points[1].y-self.points[0].y)*(e.points[1].z-e.points[0].z)-(self.points[1].z-self.points[0].z)*(e.points[1].y-e.points[0].y)
            cross_y=(self.points[1].z-self.points[0].z)*(e.points[1].x-e.points[0].x)-(self.points[1].x-self.points[0].x)*(e.points[1].z-e.points[0].z)
            cross_z=(self.points[1].x-self.points[0].x)*(e.points[1].y-e.points[0].y)-(self.points[1].y-self.points[0].y)*(e.points[1].x-e.points[0].x)
            return edge(point(cross_x,cross_y,cross_z),point(0.0,0.0,0.0))
        elif self.points[0].is_point(e.points[1]):
            cross_x=(self.points[1].y-self.points[0].y)*(e.points[0].z-e.points[1].z)-(self.points[1].z-self.points[0].z)*(e.points[0].y-e.points[1].y)
            cross_y=(self.points[1].z-self.points[0].z)*(e.points[0].x-e.points[1].x)-(self.points[1].x-self.points[0].x)*(e.points[0].z-e.points[1].z)
            cross_z=(self.points[1].x-self.points[0].x)*(e.points[0].y-e.points[1].y)-(self.points[1].y-self.points[0].y)*(e.points[0].x-e.points[1].x)
            return edge(point(cross_x,cross_y,cross_z),point(0.0,0.0,0.0))
        elif self.points[1].is_point(e.points[0]):
            cross_x=(self.points[0].y-self.points[1].y)*(e.points[1].z-e.points[0].z)-(self.points[0].z-self.points[1].z)*(e.points[1].y-e.points[0].y)
            cross_y=(self.points[0].z-self.points[1].z)*(e.points[1].x-e.points[0].x)-(self.points[0].x-self.points[1].x)*(e.points[1].z-e.points[0].z)
            cross_z=(self.points[0].x-self.points[1].x)*(e.points[1].y-e.points[0].y)-(self.points[0].y-self.points[1].y)*(e.points[1].x-e.points[0].x)
            return edge(point(cross_x,cross_y,cross_z),point(0.0,0.0,0.0))
        elif self.points[1].is_point(e.points[1]):
            cross_x=(self.points[0].y-self.points[1].y)*(e.points[0].z-e.points[1].z)-(self.points[0].z-self.points[1].z)*(e.points[0].y-e.points[1].y)
            cross_y=(self.points[0].z-self.points[1].z)*(e.points[0].x-e.points[1].x)-(self.points[0].x-self.points[1].x)*(e.points[0].z-e.points[1].z)
            cross_z=(self.points[0].x-self.points[1].x)*(e.points[0].y-e.points[1].y)-(self.points[0].y-self.points[1].y)*(e.points[0].x-e.points[1].x)
            return edge(point(cross_x,cross_y,cross_z),point(0.0,0.0,0.0))
        return None
        #return (self.points[1].x-self.points[0].x)*(e.points[1].y-e.points[0].y)-(self.points[1].y-self.points[0].y)*(e.points[1].x-e.points[0].x)
    
    def angle(self,e):
        #This method computes the angle in radians between the edge object and another inputted edge.
        return np.arccos(self.dot(e)/(self.length()*e.length()))

    def point_edge_intersect(self,p,includeboundary=True,errtol=1e-12):
        #This method returns whether an inputted point lies on the edge object or not.
        #Distinction is made whether to include the boundaries of the edge or not.
        errtol=abs(errtol)
        if any([p.is_point(elem,errtol=errtol) for elem in self.points]):
            if includeboundary:
                return True
            else:
                return False
        else:
            e1=edge(self.points[0],p)
            e2=edge(self.points[1],p)
            if (self.cross(e1).is_shorter_than(errtol)
                and e1.is_shorter_than(self.length()+errtol,or_equal_to=True)
                and e2.is_shorter_than(self.length()+errtol,or_equal_to=True)):
                if includeboundary:
                    return True
                elif (not p.is_point(self.points[0],errtol=errtol)
                      and not p.is_point(self.points[1],errtol=errtol)):
                    return True
                else:
                    return False
        return False

    def edge_edge_intersect(self,e,includeboundary=True,errtol=1e-6):
        #This method returns whether the edge object intersects the other inputted edge.
        #Distinction is made whether to include the boundaries of the edges or not.
        #If the edges do intersect. 
        if self.is_edge(e):
            return self
        
        errtol=abs(errtol)
        
        #An edge is a 1D object that can be expressed as X_0+tau*(X_1-X_0) with
        #0<=tau<=1. The method sets up the equations for both edges, sets them
        #equal to each other, and rearranges them into a 3x2 system of equations...
        A=np.array([[e.points[0].x-e.points[1].x,self.points[1].x-self.points[0].x],
                    [e.points[0].y-e.points[1].y,self.points[1].y-self.points[0].y],
                    [e.points[0].z-e.points[1].z,self.points[1].z-self.points[0].z]])
        b=np.array([e.points[0].x-self.points[0].x,
                    e.points[0].y-self.points[0].y,
                    e.points[0].z-self.points[0].z])
        
        #...which it then tries to solve. If this is solvable, the method returns
        #the intersection point. If this isn't solvable, this means that the
        #two edges are parrallel.
        for i in range(0,2):
            for j in range(i+1,3):
                try:
                    tau=np.linalg.solve(A[[i,j]],b[[i,j]])
                    if max(abs(np.dot(A,tau)-b))<errtol:
                        if ((includeboundary and max(tau)<=1+errtol and min(tau)>=-errtol)
                            or (not includeboundary and max(tau)<1-errtol and min(tau)>errtol)):
                            if abs(tau[1])<=errtol:
                                return self.points[0]
                            elif abs(tau[1]-1)<=errtol:
                                return self.points[1]
                            else:
                                return self.points[0]+point(tau[1],tau[1],tau[1])*(self.points[1]-self.points[0])
                        else:
                            return None
                    else:
                        break
                except Exception:
                    pass
        
        #The two edges must be parallel from here on out, but they might still
        #intersect. The method now sees this edge object as 3D straight line,
        #again parameterized by the variable tau. It then takes the endpoints of
        #the other edge, finds the closest points on this edge object to those
        #endpoints, and sees if they are close enough.
        tau_1=edge(e.points[0],self.points[0]).dot(self)/(self.dot(self))
        if edge(self.points[0]+point(tau_1,tau_1,tau_1)*(self.points[1]-self.points[0]),e.points[0]).is_shorter_than(errtol):
            #If the edges are close enough, then the method finds the
            #intersection between the edges.
            tau_2=edge(e.points[1],self.points[0]).dot(self)/(self.dot(self))
            tau_array=sorted([0,tau_1,tau_2,1])
            if abs(tau_array[1])<=errtol:
                if abs(tau_array[2]-1)<=errtol:
                    return self
                else:
                    return edge(self.points[0],
                             self.points[0]+point(tau_array[2],tau_array[2],tau_array[2])*(self.points[1]-self.points[0]))
            else:
                if abs(tau_array[2]-1)<=errtol:
                    return edge(self.points[0]+point(tau_array[1],tau_array[1],tau_array[1])*(self.points[1]-self.points[0]),
                             self.points[1])
                else:
                    return edge(self.points[0]+point(tau_array[1],tau_array[1],tau_array[1])*(self.points[1]-self.points[0]),
                             self.points[0]+point(tau_array[2],tau_array[2],tau_array[2])*(self.points[1]-self.points[0]))
        return None
    
    def circumsphere(self):
        #This method returns the smallest circumcircle of the edge object and makes the circumcircl object a singleton.
        if self.circumSphere is None:
            self.circumSphere=sphere(self.average(),self.points[0].distance(self.points[1])/2.0)
        return self.circumSphere
    
    def inCircumsphere(self,p,includeboundary=True,errtol=1e-12):
        #This method calculates whether an inputted point lies inside of the edge object's circumcircle.
        #Distinction is made whether to include the boundary of the circumcle or not.
        center=self.average()
        if includeboundary and (p.x-center.x)**2+(p.y-center.y)**2+(p.z-center.z)**2<=((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2+(self.points[1].z-self.points[0].z)**2)/4.0+abs(errtol):
            return True
        elif not includeboundary and (p.x-center.x)**2+(p.y-center.y)**2+(p.z-center.z)**2<((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2+(self.points[1].z-self.points[0].z)**2)/4.0-abs(errtol):
            return True
        return False
        """
        pointd=self.average()
        pointc=point(self.points[0].y-pointd.y+pointd.x,pointd.x-self.points[0].x+pointd.y)
        if (self.points[1].x-self.points[0].x)*(pointc.y-self.points[1].y)-(self.points[1].y-self.points[0].y)*(pointc.x-self.points[1].x)>0:
            res=det([[self.points[0].x-p.x,self.points[1].x-p.x,pointc.x-p.x],
                     [self.points[0].y-p.y,self.points[1].y-p.y,pointc.y-p.y],
                     [(self.points[0].x-p.x)**2+(self.points[0].y-p.y)**2,(self.points[1].x-p.x)**2+(self.points[1].y-p.y)**2,(pointc.x-p.x)**2+(pointc.y-p.y)**2]])
        else:
            res=det([[self.points[0].x-p.x,pointc.x-p.x,self.points[1].x-p.x],
                     [self.points[0].y-p.y,pointc.y-p.y,self.points[1].y-p.y],
                     [(self.points[0].x-p.x)**2+(self.points[0].y-p.y)**2,(pointc.x-p.x)**2+(pointc.y-p.y)**2,(self.points[1].x-p.x)**2+(self.points[1].y-p.y)**2]])
        if includeboundary:
            return res>=-abs(errtol)
        else:
            return res>abs(errtol)
        """
    
    def __repr__(self):
        #This method returns a string representation of the edge object.
        return "Edge "+str(self.index)+": <"+self.points[0].__repr__()+", "+self.points[1].__repr__()+">"
    
    def copy(self,blank=True):
        #This method returns a copy of the edge object.
        #Whether the copy contains all variables of the original used for constructing triangulations is user-designated.
        #The edge object's triangle list is not copied.
        e=edge(self.points[0].copy(),self.points[1].copy())
        if self.edgeAv is not None:
            e.edgeAv=self.edgeAv.copy()
        if self.edgeLength is not None:
            e.edgeLength=self.edgeLength
        if self.circumsphere is not None:
            e.circumsphere=self.circumsphere.copy()
        if not blank:
            e.enclosed=self.enclosed
            e.constraint=self.constraint
        return e

    def draw(self,plotaxis,points=True,flat=True,color="black",alpha=1):
        #This method plots the edge object into an inputted figure axis object. 
        if flat:
            if self.constraint:
                plotaxis.plot([self.points[0].x,self.points[1].x],
                              [self.points[0].y,self.points[1].y],lw=4,color=color,alpha=alpha)
            else:
                plotaxis.plot([self.points[0].x,self.points[1].x],
                              [self.points[0].y,self.points[1].y],color=color,alpha=alpha)
            if points:
                if sum(pltcolors.to_rgb(color))<=1.0:
                    plotaxis.scatter([self.points[0].x,self.points[1].x],
                                     [self.points[0].y,self.points[1].y],facecolor=color,edgecolor="white",alpha=alpha)
                else:
                    plotaxis.scatter([self.points[0].x,self.points[1].x],
                                     [self.points[0].y,self.points[1].y],facecolor=color,edgecolor="black",alpha=alpha)
        else:        
            if self.constraint:
                plotaxis.plot3D([self.points[0].x,self.points[1].x],
                                [self.points[0].y,self.points[1].y],
                                [self.points[0].z,self.points[1].z],lw=4,color=color,alpha=alpha)
            else:
                plotaxis.plot3D([self.points[0].x,self.points[1].x],
                                [self.points[0].y,self.points[1].y],
                                [self.points[0].z,self.points[1].z],color=color,alpha=alpha)
            if points:
                if sum(pltcolors.to_rgb(color))<=1.0:
                    plotaxis.scatter([self.points[0].x,self.points[1].x],
                                     [self.points[0].y,self.points[1].y],
                                     [self.points[0].z,self.points[1].z],facecolor=color,edgecolor="white",alpha=alpha)
                else:
                    plotaxis.scatter([self.points[0].x,self.points[1].x],
                                     [self.points[0].y,self.points[1].y],
                                     [self.points[0].z,self.points[1].z],facecolor=color,edgecolor="black",alpha=alpha)
