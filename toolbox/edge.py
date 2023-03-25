import numpy as np
from numpy.linalg import det
from toolbox.point import point
from toolbox.circle import circle
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
        self.circumCircle=None
        
    def is_edge(self,e,errtol=1e-12):
        #This method compares whether two edges are equal or not.
        return isinstance(e,edge) and ((self.points[0].is_point(e.points[0],errtol=errtol)
                                        and self.points[1].is_point(e.points[1],errtol=errtol))
                                        or (self.points[0].is_point(e.points[1],errtol=errtol)
                                        and self.points[1].is_point(e.points[0],errtol=errtol)))
        
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
        return self.edgeAv
    
    def length(self):
        #This method computes the length of the edge object and makes the length variable a singleton.
        if self.edgeLength is None:
            #self.edgeLength=np.sqrt((self.points[0].x-self.points[1].x)**2+(self.points[0].y-self.points[1].y)**2)
            self.edgeLength=np.linalg.norm(np.array([self.points[0].x,self.points[0].y])-np.array([self.points[1].x,self.points[1].y]))
        return self.edgeLength
    
    def dot(self,e):
        #This method returns the 2-dimensional dot product of the edge object and another inputted edge.
        if self.points[0].is_point(e.points[1]):
            e.swap()
        if e.points[0].is_point(self.points[1]):
            self.swap()
        return (self.points[1].x-self.points[0].x)*(e.points[1].x-e.points[0].x)+(self.points[1].y-self.points[0].y)*(e.points[1].y-e.points[0].y) 
        
    def cross(self,e):
        #This method returns the 2-dimensional "cross product" of the edge object and another inputted edge.
        return (self.points[1].x-self.points[0].x)*(e.points[1].y-e.points[0].y)-(self.points[1].y-self.points[0].y)*(e.points[1].x-e.points[0].x)
    
    def angle(self,e):
        #This method computes the angle in radians between the edge object and another inputted edge.
        return np.arccos(self.dot(e)/(self.length()*e.length()))

    def point_edge_intersect(self,p,includeboundary=True,errtol=1e-12):
        #This method returns whether an inputted point lies on the edge object or not.
        #Distinction is made whether to include the boundaries of the edge or not.
        #if abs((self.points[1].x-self.points[0].x)*(p.y-self.points[0].y)-(self.points[1].y-self.points[0].y)*(p.x-self.points[0].x))<abs(errtol):
        if any([elem.is_point(p,errtol=errtol) for elem in self.points]):
            if includeboundary:
                return True
            else:
                return False
        elif abs(self.cross(edge(self.points[0],p)))<abs(errtol):
            if ((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2-(p.x-self.points[0].x)**2-(p.y-self.points[0].y)**2>=-abs(errtol)
                and (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2-(p.x-self.points[1].x)**2-(p.y-self.points[1].y)**2>=-abs(errtol)):
                if includeboundary:
                    return True
                elif abs(p.x-self.points[0].x)>abs(errtol) or abs(p.y-self.points[0].y)>abs(errtol):
                    return True
                else:
                    return False
        return False

    def edge_edge_intersect(self,e,includeboundary=True,errtol=1e-12):
        #This method returns whether the edge object intersects the other inputted edge.
        #Distinction is made whether to include the boundaries of the edges or not.
        #If the edges do intersect, the method can either return true or false, or return the intersection. 
        if self.is_edge(e):
            return self
        
        cross_prod=-self.cross(e)
        if abs(cross_prod)>abs(errtol):
            s1=-edge(self.points[0],e.points[0]).cross(e)/cross_prod
            s2=-edge(self.points[0],e.points[0]).cross(self)/cross_prod
            #s1=((e.points[0].x-self.points[0].x)*(e.points[0].y-e.points[1].y)-(e.points[0].y-self.points[0].y)*(e.points[0].x-e.points[1].x))/cross_prod
            #s2=((e.points[0].x-self.points[0].x)*(self.points[0].y-self.points[1].y)+(e.points[0].y-self.points[0].y)*(self.points[1].x-self.points[0].x))/cross_prod
            if (includeboundary and -abs(errtol)<=s1<=1+abs(errtol) and -abs(errtol)<=s2<=1+abs(errtol)) or (not includeboundary and abs(errtol)<s1<1-abs(errtol) and abs(errtol)<s2<1-abs(errtol)):
                return point(self.points[0].x+s1*(self.points[1].x-self.points[0].x),self.points[0].y+s1*(self.points[1].y-self.points[0].y))
        elif abs(self.cross(edge(self.points[1],e.points[0])))<=abs(errtol):
            if abs(self.points[1].x-self.points[0].x)>=abs(self.points[1].y-self.points[0].y):
                if self.points[1].x>self.points[0].x:
                    min1,max1=0,1
                else:
                    min1,max1=1,0
                if e.points[1].x>e.points[0].x:
                    min2,max2=0,1
                else:
                    min2,max2=1,0
                if includeboundary:
                    if self.points[min1].x-abs(errtol)<=e.points[min2].x<=self.points[max1].x+abs(errtol) and self.points[max1].x<=e.points[max2].x+abs(errtol):
                        if abs(self.points[max1].x-e.points[min2].x)<=abs(errtol):
                            return self.points[max1].copy()
                        return edge(e.points[min2],self.points[max1])
                    elif e.points[min2].x<=self.points[min1].x+abs(errtol) and self.points[min1].x-abs(errtol)<=e.points[max2].x<=self.points[max1].x+abs(errtol):
                        if abs(self.points[min1].x-e.points[max2].x)<=abs(errtol):
                            return self.points[min1].copy()
                        return edge(self.points[min1],e.points[max2])
                    elif self.points[min1].x-abs(errtol)<=e.points[min2].x<=self.points[max1].x+abs(errtol) and self.points[min1].x-abs(errtol)<=e.points[max2].x<=self.points[max1].x+abs(errtol):
                        return edge(e.points[min2],e.points[max2])
                    elif e.points[min2].x-abs(errtol)<=self.points[min1].x<=e.points[max2].x+abs(errtol) and e.points[min2].x-abs(errtol)<=self.points[max1].x<=e.points[max2].x+abs(errtol):
                        return edge(self.points[min1],self.points[max1])
                else:
                    if self.points[min1].x-abs(errtol)<=e.points[min2].x<self.points[max1].x-abs(errtol) and self.points[max1].x<=e.points[max2].x+abs(errtol):
                        return edge(e.points[min2],self.points[max1])
                    elif e.points[min2].x<=self.points[min1].x+abs(errtol) and self.points[min1].x+abs(errtol)<e.points[max2].x<=self.points[max1].x+abs(errtol):
                        return edge(self.points[min1],e.points[max2])
                    elif self.points[min1].x-abs(errtol)<=e.points[min2].x<=self.points[max1].x+abs(errtol) and self.points[min1].x-abs(errtol)<=e.points[max2].x<=self.points[max1].x+abs(errtol):
                        return edge(e.points[min2],e.points[max2])
                    elif e.points[min2].x-abs(errtol)<=self.points[min1].x<=e.points[max2].x+abs(errtol) and e.points[min2].x-abs(errtol)<=self.points[max1].x<=e.points[max2].x+abs(errtol):
                        return edge(self.points[min1],self.points[max1])
            else:
                if self.points[1].y>self.points[0].y:
                    min1,max1=0,1
                else:
                    min1,max1=1,0
                if e.points[1].y>e.points[0].y:
                    min2,max2=0,1
                else:
                    min2,max2=1,0
                if includeboundary:
                    if self.points[min1].y-abs(errtol)<=e.points[min2].y<=self.points[max1].y+abs(errtol) and self.points[max1].y<=e.points[max2].y+abs(errtol):
                        if abs(self.points[max1].y-e.points[min2].y)<=abs(errtol):
                            return self.points[max1].copy()
                        return edge(e.points[min2],self.points[max1])
                    elif e.points[min2].y<=self.points[min1].y+abs(errtol) and self.points[min1].y-abs(errtol)<=e.points[max2].y<=self.points[max1].y+abs(errtol):
                        if abs(self.points[min1].y-e.points[max2].y)<=abs(errtol):
                            return self.points[min1].copy()
                        return edge(self.points[min1],e.points[max2])
                    elif self.points[min1].y-abs(errtol)<=e.points[min2].y<=self.points[max1].y+abs(errtol) and self.points[min1].y-abs(errtol)<=e.points[max2].y<=self.points[max1].y+abs(errtol):
                        return edge(e.points[min2],e.points[max2])
                    elif e.points[min2].y-abs(errtol)<=self.points[min1].y<=e.points[max2].y+abs(errtol) and e.points[min2].y-abs(errtol)<=self.points[max1].y<=e.points[max2].y+abs(errtol):
                        return edge(self.points[min1],self.points[max1])
                else:
                    if self.points[min1].y-abs(errtol)<=e.points[min2].y<self.points[max1].y-abs(errtol) and self.points[max1].y<=e.points[max2].y+abs(errtol):
                        return edge(e.points[min2],self.points[max1])
                    elif e.points[min2].y<=self.points[min1].y+abs(errtol) and self.points[min1].y+abs(errtol)<e.points[max2].y<=self.points[max1].y+abs(errtol):
                        return edge(self.points[min1],e.points[max2])
                    elif self.points[min1].y-abs(errtol)<=e.points[min2].y<=self.points[max1].y+abs(errtol) and self.points[min1].y-abs(errtol)<=e.points[max2].y<=self.points[max1].y+abs(errtol):
                        return edge(e.points[min2],e.points[max2])
                    elif e.points[min2].y-abs(errtol)<=self.points[min1].y<=e.points[max2].y+abs(errtol) and e.points[min2].y-abs(errtol)<=self.points[max1].y<=e.points[max2].y+abs(errtol):
                        return edge(self.points[min1],self.points[max1])
        return None
    
    def circumcircle(self):
        #This method returns the smallest circumcircle of the edge object and makes the circumcircl object a singleton.
        if self.circumCircle is None:
            self.circumCircle=circle(point((self.points[0].x+self.points[1].x)/2,(self.points[0].y+self.points[1].y)/2),
                                     np.sqrt((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2)/2)
        return self.circumCircle
    
    def inCircumcircle(self,p,includeboundary=True,errtol=1e-12):
        #This method calculates whether an inputted point lies inside of the edge object's circumcircle.
        #Distinction is made whether to include the boundary of the circumcle or not.
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
    
    def swap(self):
        #This method swaps the local indexes of the end-points of the edge object.
        self.points.reverse()
    
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
        if self.circumcircle is not None:
            e.circumcircle=self.circumcircle.copy()
        if not blank:
            e.enclosed=self.enclosed
            e.constraint=self.constraint
        return e

    def draw(self,plotaxis,points=True,color="black",alpha=1):
        #This method plots the edge object into an inputted figure axis object. 
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
