import numpy as np
import numpy.linalg as npla
from toolbox.point import point
from toolbox.edge import edge
from toolbox.circle import circle
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

#This file contains the triangle class and its various methods.
class triangle(object):
    def __init__(self,p1,p2,p3,constructedges=True):
        #This list keeps track of which points constitute the end-points of the triangle object.
        self.points=[p1,p2,p3]
        self.triagAv=None
        #This list keeps track of which edge constitute the boundaries of the triangle object.
        self.edges=[]
        self.neighbors=[]
        self.interiorAngles=None
        self.triagArea=None
        self.index=-1
        self.circumCircle=None
            
    def is_triangle(self,t,errtol=1e-12):
        #This method compares whether two triangles are equal or not.
        if (self.points[0].is_point(t.points[0],errtol=errtol)
            and self.points[1].is_point(t.points[1],errtol=errtol)
            and self.points[2].is_point(t.points[2],errtol=errtol)):
            return True
        elif (self.points[0].is_point(t.points[0],errtol=errtol)
              and self.points[1].is_point(t.points[2],errtol=errtol)
              and self.points[2].is_point(t.points[1],errtol=errtol)):
            return True
        elif (self.points[0].is_point(t.points[1],errtol=errtol)
              and self.points[1].is_point(t.points[0],errtol=errtol)
              and self.points[2].is_point(t.points[2],errtol=errtol)):
            return True
        elif (self.points[0].is_point(t.points[1],errtol=errtol)
              and self.points[1].is_point(t.points[2],errtol=errtol)
              and self.points[2].is_point(t.points[0],errtol=errtol)):
            return True
        elif (self.points[0].is_point(t.points[2],errtol=errtol)
              and self.points[1].is_point(t.points[0],errtol=errtol)
              and self.points[2].is_point(t.points[1],errtol=errtol)):
            return True
        elif (self.points[0].is_point(t.points[2],errtol=errtol)
              and self.points[1].is_point(t.points[1],errtol=errtol)
              and self.points[2].is_point(t.points[0],errtol=errtol)):
            return True
        return False
    
    def update(self):
        #For each end-point of the triangle object, this method adds the triangle to the point's list of triangles it is a constituent of.
        #For each boundary edge of the triangle object, this method adds the triangle to the edge's list of triangles it is a constituent of.
        for p in self.points:
            p.triangles.append(self)
        for e in self.edges:
            e.triangles.append(self)
            for triangle_var in e.triangles:
                if not triangle_var.is_triangle(self):
                    self.neighbors.append(triangle_var)
                    triangle_var.neighbors.append(self)
    
    def average(self):
        #This method computes the centroid of the triangle object and makes the centroid variable a singleton.
        if self.triagAv is None:
            self.triagAv=self.points[0]+self.points[1]+self.points[2]
            self.triagAv.x/=3.0
            self.triagAv.y/=3.0
        return self.triagAv
    
    def area(self):
        #This method computes the area of the triangle object and makes the area variable a singleton.
        if self.triagArea is None:
            self.triagArea=0.5*abs(self.points[0].x*(self.points[1].y-self.points[2].y)
                                  +self.points[1].x*(self.points[2].y-self.points[0].y)
                                  +self.points[2].x*(self.points[0].y-self.points[1].y))
        return self.triagArea
    
    def angles(self):
        #This method computes the interior angles of the triangle object and makes the interior angles object a singleton.
        if self.interiorAngles is None:
            if len(self.edges)==3:
                angle0=0
                for i in range(3):
                    if not self.edges[i].points[0].is_point(self.points[0]) and not self.edges[i].points[1].is_point(self.points[0]):
                        angle0=np.acos((sum([self.edges[j].length()**2 for j in range(3) if j!=i])-self.edges[i].length()**2)/(2*np.prod([self.edges[j].length() for j in range(3) if j!=i])))
                        break
                angle1=0
                for i in range(3):
                    if not self.edges[i].points[0].is_point(self.points[1]) and not self.edges[i].points[1].is_point(self.points[1]):
                        angle1=np.acos((sum([self.edges[j].length()**2 for j in range(3) if j!=i])-self.edges[i].length()**2)/(2*np.prod([self.edges[j].length() for j in range(3) if j!=i])))
                        break
                angle2=np.pi-angle0-angle1
            else:
                a=np.sqrt((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2)
                b=np.sqrt((self.points[2].x-self.points[0].x)**2+(self.points[2].y-self.points[0].y)**2)
                c=np.sqrt((self.points[2].x-self.points[1].x)**2+(self.points[2].y-self.points[1].y)**2)
                angle0=np.acos((a**2+b**2-c**2)/(2*a*b))
                angle1=np.acos((a**2+c**2-b**2)/(2*a*c))
                angle2=np.pi-angle0-angle1
            self.interiorAngles=[angle0,angle1,angle2]
        return self.interiorAngles
    
    def point_triangle_intersect(self,pointVar,includeboundary=True,errtol=1e-12):
        #This method returns whether a point is inside the triangle object or not.
        #Distinction is made whether to include the boundary edges or not.
        
        cross_prod=edge(self.points[0],self.points[1]).cross(edge(self.points[0],self.points[2]))
        w1=edge(self.points[0],pointVar).cross(edge(self.points[0],self.points[2]))/cross_prod
        w2=edge(self.points[0],pointVar).cross(edge(self.points[1],self.points[0]))/cross_prod
        #cross_prod=(self.points[1].x-self.points[0].x)*(self.points[2].y-self.points[0].y)-(self.points[1].y-self.points[0].y)*(self.points[2].x-self.points[0].x)
        #w1=((pointVar.x-self.points[0].x)*(self.points[2].y-self.points[0].y)-(pointVar.y-self.points[0].y)*(self.points[2].x-self.points[0].x))/cross_prod
        #w2=((pointVar.x-self.points[0].x)*(self.points[0].y-self.points[1].y)-(pointVar.y-self.points[0].y)*(self.points[0].x-self.points[1].x))/cross_prod
        
        if includeboundary:
            if w1>=-abs(errtol) and w2>=-abs(errtol) and w1+w2<=1+abs(errtol):
                return True
        else:
            if w1>abs(errtol) and w2>abs(errtol) and w1+w2<1-abs(errtol):
                return True
        return False
        
    def edge_triangle_intersect(self,edgeVar,includeboundary=True,errtol=1e-12):
        #This method returns whether an edge intersects the triangle object or not.
        #Distinction is made whether to include the boundaries of the triangle and edge objects or not.
        if len(self.edges)==0:
            self.edges=[edge(self.points[0],self.points[1]),edge(self.points[0],self.points[2]),edge(self.points[1],self.points[2])]
        if includeboundary:
            if any([self.point_triangle_intersect(p,errtol=errtol) for p in edgeVar.points]):
                return True
            else:
                if any([edgeVar.edge_edge_intersect(e,errtol=errtol) for e in self.edges]):
                    return True
            return False
        else:
            if any([self.point_triangle_intersect(p,includeboundary=False,errtol=errtol) for p in edgeVar.points]):
                return True
            else:        
                p1bool1=any([e.point_edge_intersect(edgeVar.points[0],errtol=errtol) for e in self.edges])
                p1bool2=any([p.is_point(edgeVar.points[0]) for p in self.points])
                if p1bool1 and not p1bool2:
                    state1=1
                elif p1bool2:
                    state1=2
                else:
                    state1=3
                
                p2bool1=any([e.point_edge_intersect(edgeVar.points[1],errtol=errtol) for e in self.edges])
                p2bool2=any([p.is_point(edgeVar.points[1]) for p in self.points])
                if p2bool1 and not p2bool2:
                    state2=1
                elif p2bool2:
                    state2=2
                else:
                    state2=3
                    
                state=state1*state2
                
                if state==1 or state==2:
                    if any([e.point_edge_intersect(edgeVar.points[0],errtol=errtol) and e.point_edge_intersect(edgeVar.points[1],errtol=errtol) for e in self.edges]):
                        return False
                    else:
                        return True
                elif state==3:
                    if any([self.edges[0].point_edge_intersect(p,errtol=errtol) for p in edgeVar.points]):
                        return self.edges[1].edge_edge_intersect(edgeVar,includeboundary=False,errtol=errtol) or self.edges[2].edge_edge_intersect(edgeVar,includeboundary=False,errtol=errtol)
                    elif any([self.edges[1].point_edge_intersect(p,errtol=errtol) for p in edgeVar.points]):
                        return self.edges[0].edge_edge_intersect(edgeVar,includeboundary=False,errtol=errtol) or self.edges[2].edge_edge_intersect(edgeVar,includeboundary=False,errtol=errtol)
                    else:
                        return self.edges[0].edge_edge_intersect(edgeVar,includeboundary=False,errtol=errtol) or self.edges[1].edge_edge_intersect(edgeVar,includeboundary=False,errtol=errtol)
                elif state==6:
                    return any([edgeVar.edge_edge_intersect(e,includeboundary=False,errtol=errtol) for e in self.edges])
                elif state==9:
                    if any([edgeVar.edge_edge_intersect(e,includeboundary=False,errtol=errtol) for e in self.edges]):
                        return True
            return False
                
    def triangle_triangle_intersect(self,triangle,includeboundary=True,errtol=1e-12):
        #This method returns whether an inputted triangle intersects the triangle object or not.
        #Distinction is made whether to include the boundaries of the triangle objects or not.
        if len(self.edges)==0:
            self.edges=[edge(self.points[0],self.points[1]),edge(self.points[0],self.points[2]),edge(self.points[1],self.points[2])]
        if len(triangle.edges)==0:
            triangle.edges=[edge(triangle.points[0],triangle.points[1]),edge(triangle.points[0],triangle.points[2]),edge(triangle.points[1],triangle.points[2])]
        return any([triangle.edge_triangle_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in self.edges]) or any([self.edge_triangle_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in triangle.edges])
    
    def circumcircle(self):
        #This method computes the circumcircle of the triangle object and makes the circumcircle variable a singleton.
        if self.circumCircle is None:
            Sx=npla.det([[self.points[0].x**2+self.points[0].y**2,self.points[0].y,1],
                         [self.points[1].x**2+self.points[1].y**2,self.points[1].y,1],
                         [self.points[2].x**2+self.points[2].y**2,self.points[2].y,1]])/2
            Sy=npla.det([[self.points[0].x,self.points[0].x**2+self.points[0].y**2,1],
                         [self.points[1].x,self.points[1].x**2+self.points[1].y**2,1],
                         [self.points[2].x,self.points[2].x**2+self.points[2].y**2,1]])/2
            a=npla.det([[self.points[0].x,self.points[0].y,1],
                        [self.points[1].x,self.points[1].y,1],
                        [self.points[2].x,self.points[2].y,1]])
            b=npla.det([[self.points[0].x,self.points[0].y,self.points[0].x**2+self.points[0].y**2],
                        [self.points[1].x,self.points[1].y,self.points[1].x**2+self.points[1].y**2],
                        [self.points[2].x,self.points[2].y,self.points[2].x**2+self.points[2].y**2]])
            self.circumCircle=circle(point(Sx/a,Sy/a),np.sqrt(b/a+(Sx**2+Sy**2)/a**2))
        return self.circumCircle
    
    def inCircumcircle(self,p,includeboundary=True,errtol=1e-12):
        #This method calculates whether an inputted point lies inside of the triangle object's circumcircle.
        #Distinction is made whether to include the boundary of the circumcircle or not.
        if edge(self.points[0],self.points[1]).cross(edge(self.points[1],self.points[2]))>0:
            res=npla.det([[self.points[0].x-p.x,self.points[1].x-p.x,self.points[2].x-p.x],
                          [self.points[0].y-p.y,self.points[1].y-p.y,self.points[2].y-p.y],
                          [(self.points[0].x-p.x)**2+(self.points[0].y-p.y)**2,(self.points[1].x-p.x)**2+(self.points[1].y-p.y)**2,(self.points[2].x-p.x)**2+(self.points[2].y-p.y)**2]])
        else:
            res=npla.det([[self.points[0].x-p.x,self.points[2].x-p.x,self.points[1].x-p.x],
                          [self.points[0].y-p.y,self.points[2].y-p.y,self.points[1].y-p.y],
                          [(self.points[0].x-p.x)**2+(self.points[0].y-p.y)**2,(self.points[2].x-p.x)**2+(self.points[2].y-p.y)**2,(self.points[1].x-p.x)**2+(self.points[1].y-p.y)**2]])
        
        if includeboundary:
            return res>=-abs(errtol)
        else:
            return res>abs(errtol)    
    
    def __repr__(self):
        #This method returns a string representation of the triangle object.
        return "Triangle "+str(self.index)+": {"+self.points[0].__repr__()+", "+self.points[1].__repr__()+", "+self.points[2].__repr__()+"}"
    
    def copy(self,blank=True):
        #This method returns a copy of the triangle object.
        #Whether the copy contains all variables of the original used for constructing triangulations is user-designated.
        #The triangle object's edge list is not copied.
        t=triangle(self.points[0].copy(),self.points[1].copy(),self.points[2].copy())
        if self.triagAv is not None:
            t.triagAv=self.triagAv.copy()
        if self.triAgarea is not None:
            t.triagArea=self.triagArea
        if self.interiorAngles is not None:
            t.interiorAngles=self.interiorAngles
        if self.circumcircle is not None:    
            t.circumcircle=self.circumcircle.copy() 
        return t
    
    def draw(self,plotaxis,points=True,edges=True,fill=True,color="black",alpha=1):    
        #This method plots the triangle object into an inputted figure axis object.
        if sum(pltcolors.to_rgb(color))<=1.0:
            color_alt="white"
        else:
            color_alt="black"
            
        if fill:
            face_color=color
            if edges:
                edge_color=color_alt
            else:
                edge_color=color
            plotaxis.add_patch(plt.Polygon([[p.x,p.y] for p in self.points],facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
        else:
            if edges:
                for e in self.edges:
                    plotaxis.plot([p.x for p in e.points],[p.y for p in e.points],color=color,alpha=alpha,zorder=0)

        if points:
            if fill:
                plotaxis.scatter([p.x for p in self.points],[p.y for p in self.points],facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
            else:
                plotaxis.scatter([p.x for p in self.points],[p.y for p in self.points],facecolor=color,edgecolor=color,alpha=alpha,zorder=1)
