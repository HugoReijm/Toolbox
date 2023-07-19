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
        self.tetrahedra=[]
        self.neighbors=[]
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
        
    def update_network(self):
        #For each end-point of the edge object, this method adds the edge to the point's list of edges it is a constituent of.
        for p in self.points:
            for e in p.edges:
                if not e.is_edge(self):
                    self.neighbors.append(e)
                    e.neighbors.append(self)
            p.edges.append(self)
        self.points[0].neighbors.append(self.points[1])
        self.points[1].neighbors.append(self.points[0])
        
    def average(self):
        #This method computes the centroid of the triangle object and makes the centroid variable a singleton.
        if self.edgeAv is None:
            self.edgeAv=(self.points[0]+self.points[1]).scalar_div(2.0)
        return self.edgeAv
    
    def length(self):
        #This method computes the length of the edge object and makes the length variable a singleton.
        if self.edgeLength is None:
            self.edgeLength=self.points[0].distance(self.points[1])
        return self.edgeLength
    
    def length_squared(self):
        #This method computes the squared length of the edge object.
        return (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2+(self.points[1].z-self.points[0].z)**2
        
    def is_longer_than(self,dist,or_equal_to=False):
        #This method computes whether the edge is longer (or equal to) a certain
        #distance without using a square root function (which isnotoriously slow)
        if or_equal_to:
            return self.length_squared()>=dist**2
        else:
            return self.length_squared()>dist**2
        
    def is_shorter_than(self,dist,or_equal_to=False):
        #This method computes whether the edge is shorter (or equal to) a certain
        #distance without using a square root function (which isnotoriously slow)
        if or_equal_to:
            return self.length_squared()<=dist**2
        else:
            return self.length_squared()<dist**2
    
    def dot(self,e):
        #This method returns the dot product of the edge object and
        #another inputted edge.
        if self.points[0].is_point(e.points[0]):
            return np.array([self.points[1].x-self.points[0].x,
                             self.points[1].y-self.points[0].y,
                             self.points[1].z-self.points[0].z]).dot(
                   np.array([e.points[1].x-e.points[0].x,
                             e.points[1].y-e.points[0].y,
                             e.points[1].z-e.points[0].z]))
        elif self.points[0].is_point(e.points[1]):
            return np.array([self.points[1].x-self.points[0].x,
                             self.points[1].y-self.points[0].y,
                             self.points[1].z-self.points[0].z]).dot(
                   np.array([e.points[0].x-e.points[1].x,
                             e.points[0].y-e.points[1].y,
                             e.points[0].z-e.points[1].z]))
        elif self.points[1].is_point(e.points[0]):
            return np.array([self.points[0].x-self.points[1].x,
                             self.points[0].y-self.points[1].y,
                             self.points[0].z-self.points[1].z]).dot(
                   np.array([e.points[1].x-e.points[0].x,
                             e.points[1].y-e.points[0].y,
                             e.points[1].z-e.points[0].z]))
        elif self.points[1].is_point(e.points[1]):
            return np.array([self.points[0].x-self.points[1].x,
                             self.points[0].y-self.points[1].y,
                             self.points[0].z-self.points[1].z]).dot(
                   np.array([e.points[0].x-e.points[1].x,
                             e.points[0].y-e.points[1].y,
                             e.points[0].z-e.points[1].z]))
        return None
    
    def cross(self,e):
        #This method returns the cross product of the edge object and
        #another inputted edge. Debate was given to whether np.cross would be
        #better, but it turns out np.cross is very slow.
        if self.points[0].is_point(e.points[0]):
            ax,bx=self.points[1].x-self.points[0].x,e.points[1].x-e.points[0].x
            ay,by=self.points[1].y-self.points[0].y,e.points[1].y-e.points[0].y
            az,bz=self.points[1].z-self.points[0].z,e.points[1].z-e.points[0].z
            return edge(point(ay*bz-az*by+self.points[0].x,
                              az*bx-ax*bz+self.points[0].y,
                              ax*by-ay*bx+self.points[0].z),self.points[0])
        elif self.points[0].is_point(e.points[1]):
            ax,bx=self.points[1].x-self.points[0].x,e.points[0].x-e.points[1].x
            ay,by=self.points[1].y-self.points[0].y,e.points[0].y-e.points[1].y
            az,bz=self.points[1].z-self.points[0].z,e.points[0].z-e.points[1].z
            return edge(point(ay*bz-az*by+self.points[0].x,
                              az*bx-ax*bz+self.points[0].y,
                              ax*by-ay*bx+self.points[0].z),self.points[0])
        elif self.points[1].is_point(e.points[0]):
            ax,bx=self.points[0].x-self.points[1].x,e.points[1].x-e.points[0].x
            ay,by=self.points[0].y-self.points[1].y,e.points[1].y-e.points[0].y
            az,bz=self.points[0].z-self.points[1].z,e.points[1].z-e.points[0].z
            return edge(point(ay*bz-az*by+self.points[1].x,
                              az*bx-ax*bz+self.points[1].y,
                              ax*by-ay*bx+self.points[1].z),self.points[1])
        elif self.points[1].is_point(e.points[1]):
            ax,bx=self.points[0].x-self.points[1].x,e.points[0].x-e.points[1].x
            ay,by=self.points[0].y-self.points[1].y,e.points[0].y-e.points[1].y
            az,bz=self.points[0].z-self.points[1].z,e.points[0].z-e.points[1].z
            return edge(point(ay*bz-az*by+self.points[1].x,
                              az*bx-ax*bz+self.points[1].y,
                              ax*by-ay*bx+self.points[1].z),self.points[1])
        return None

    def angle(self,e):
        #This method computes the angle in radians between the edge object and another inputted edge.
        try:
            res=self.dot(e)/(self.length()*e.length())
            if res>=1:
                return 0
            elif res<=-1:
                return np.pi
            return np.arccos(self.dot(e)/(self.length()*e.length()))
        except Exception:
            return None

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
        parrallel_bool=False
        for i in range(0,2):
            for j in range(i+1,3):
                if abs(A[i,0]*A[j,1]-A[i,1]*A[j,0])>errtol:
                    tau=np.linalg.solve(A[[i,j]],b[[i,j]])
                    if max(abs(np.dot(A,tau)-b))<errtol:
                        #If the lines do intersect, then the method needs to check
                        #if the lines intersect in the appropriate sections of
                        #each line. That being said, the lines may intersect, but
                        #the edges (which are only subsections of the lines) may not.
                        if ((includeboundary and max(tau)<=1+errtol and min(tau)>=-errtol)
                            or (not includeboundary and max(tau)<1-errtol and min(tau)>errtol)):
                            return True
                        else:
                            return False
                    else:
                        parrallel_bool=True
                        break
            if parrallel_bool:
                break

        #The two edges must be parallel from here on out, but they might still
        #intersect. The method now sees this edge object as 3D straight line,
        #again parameterized by the variable tau. It then takes the endpoints of
        #the other edge, finds the closest points on this edge object to those
        #endpoints, and sees if they are close enough.
        tau_1=edge(e.points[0],self.points[0]).dot(self)/(self.length_squared())
        if edge(self.points[0]+(self.points[1]-self.points[0]).scalar_mul(tau_1),e.points[0]).is_shorter_than(errtol):
            #Now it checks if the other edge falls into this edge
            tau_2=edge(e.points[1],self.points[0]).dot(self)/(self.length_squared())
            if includeboundary and (-errtol<=tau_1<=1+errtol or -errtol<=tau_2<=1+errtol):
                return True
            if not includeboundary and (errtol<tau_1<1-errtol or errtol<tau_2<1-errtol):
                return True
        return False
    
    def circumsphere(self):
        #This method returns the smallest circumcircle of the edge object and makes the circumcircl object a singleton.
        if self.circumSphere is None:
            self.circumSphere=sphere(self.average(),self.points[0].distance(self.points[1])/2.0)
        return self.circumSphere
    
    def inCircumsphere(self,p,includeboundary=True,errtol=1e-12):
        #This method calculates whether an inputted point lies inside of the edge object's circumcircle.
        #Distinction is made whether to include the boundary of the circumcle or not.
        center=self.average()
        if includeboundary:
            return edge(p,center).length_squared()<=self.length_squared()/4.0+abs(errtol)
        else:
            return edge(p,center).length_squared()<self.length_squared()/4.0-abs(errtol)

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

    def draw(self,plotaxis,points=True,flat=True,elev=10,azim=-75,color="black",alpha=1):
        #This method plots the edge object into an inputted figure axis object. 
        if flat:
            if self.constraint:
                plotaxis.plot([p.x for p in self.points],
                              [p.y for p in self.points],
                              lw=4,color=color,alpha=alpha)
            else:
                plotaxis.plot([p.x for p in self.points],
                              [p.y for p in self.points],
                              color=color,alpha=alpha)
            if points:
                if sum(pltcolors.to_rgb(color))<=1.0:
                    plotaxis.scatter([p.x for p in self.points],
                                     [p.y for p in self.points],
                                     facecolor=color,edgecolor="white",alpha=alpha)
                else:
                    plotaxis.scatter([p.x for p in self.points],
                                     [p.y for p in self.points],
                                     facecolor=color,edgecolor="black",alpha=alpha)
        else:        
            if self.constraint:
                plotaxis.plot3D([p.x for p in self.points],
                                [p.y for p in self.points],
                                [p.z for p in self.points],
                                lw=4,color=color,alpha=alpha)
            else:
                plotaxis.plot3D([p.x for p in self.points],
                                [p.y for p in self.points],
                                [p.z for p in self.points],
                                color=color,alpha=alpha)
            if points:
                if sum(pltcolors.to_rgb(color))<=1.0:
                    plotaxis.scatter3D([p.x for p in self.points],
                                       [p.y for p in self.points],
                                       [p.z for p in self.points],
                                       facecolor=color,edgecolor="white",alpha=alpha)
                else:
                    plotaxis.scatter3D([p.x for p in self.points],
                                       [p.y for p in self.points],
                                       [p.z for p in self.points],
                                       facecolor=color,edgecolor="black",alpha=alpha)
            plotaxis.view_init(elev=elev,azim=azim)
