import math
import numpy as np
import numpy.linalg as npla
from point import point
from edge import edge
from circle import circle
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

class triangle(object):
    def __init__(self,p1,p2,p3):
        self.points=[p1,p2,p3]
        self.edges=[]
        self.interiorAngles=None
        self.triagarea=None
        self.index=-1
        self.circumCircle=None
        
    def update(self):
        for p in self.points:
            p.triangles.append(self)
        for e in self.edges:
            e.triangles.append(self)
            
    def is_triangle(self,triangle,errtol=1e-12):
        if self.points[0].is_point(triangle.points[0],errtol=errtol) and self.points[1].is_point(triangle.points[1],errtol=errtol) and self.points[2].is_point(triangle.points[2],errtol=errtol):
            return True
        elif self.points[0].is_point(triangle.points[0],errtol=errtol) and self.points[1].is_point(triangle.points[2],errtol=errtol) and self.points[2].is_point(triangle.points[1],errtol=errtol):
            return True
        elif self.points[0].is_point(triangle.points[1],errtol=errtol) and self.points[1].is_point(triangle.points[0],errtol=errtol) and self.points[2].is_point(triangle.points[2],errtol=errtol):
            return True
        elif self.points[0].is_point(triangle.points[1],errtol=errtol) and self.points[1].is_point(triangle.points[2],errtol=errtol) and self.points[2].is_point(triangle.points[0],errtol=errtol):
            return True
        elif self.points[0].is_point(triangle.points[2],errtol=errtol) and self.points[1].is_point(triangle.points[0],errtol=errtol) and self.points[2].is_point(triangle.points[1],errtol=errtol):
            return True
        elif self.points[0].is_point(triangle.points[2],errtol=errtol) and self.points[1].is_point(triangle.points[1],errtol=errtol) and self.points[2].is_point(triangle.points[0],errtol=errtol):
            return True
        return False
            
    def point_triangle_intersect(self,pointVar,includeboundary=True,errtol=1e-12):
        det=(self.points[1].x-self.points[0].x)*(self.points[2].y-self.points[0].y)-(self.points[1].y-self.points[0].y)*(self.points[2].x-self.points[0].x)
        w1=((pointVar.x-self.points[0].x)*(self.points[2].y-self.points[0].y)+(pointVar.y-self.points[0].y)*(self.points[0].x-self.points[2].x))/det
        w2=((pointVar.x-self.points[0].x)*(self.points[0].y-self.points[1].y)+(pointVar.y-self.points[0].y)*(self.points[1].x-self.points[0].x))/det
        
        if includeboundary:
            if w1>=-abs(errtol) and w2>=-abs(errtol) and w1+w2<=1+abs(errtol):
                return True
        else:
            if w1>abs(errtol) and w2>abs(errtol) and w1+w2<1-abs(errtol):
                return True
        return False
        
    def edge_triangle_intersect(self,edgeVar,includeboundary=True,errtol=1e-12):
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
        if len(self.edges)==0:
            self.edges=[edge(self.points[0],self.points[1]),edge(self.points[0],self.points[2]),edge(self.points[1],self.points[2])]
        if len(triangle.edges)==0:
            triangle.edges=[edge(triangle.points[0],triangle.points[1]),edge(triangle.points[0],triangle.points[2]),edge(triangle.points[1],triangle.points[2])]
        return any([triangle.edge_triangle_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in self.edges]) or any([self.edge_triangle_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in triangle.edges])
        
    def area(self):
        if self.triagarea is None:
            self.triagarea=0.5*abs(self.points[0].x*(self.points[1].y-self.points[2].y)+self.points[1].x*(self.points[2].y-self.points[0].y)+self.points[2].x*(self.points[0].y-self.points[1].y))
        return self.triagarea
    
    def angles(self):
        if self.interiorAngles is None:
            if len(self.edges)==3:
                angle0=0
                for i in range(3):
                    if not self.edges[i].points[0].is_point(self.points[0]) and not self.edges[i].points[1].is_point(self.points[0]):
                        angle0=math.acos((sum([self.edges[j].length()**2 for j in range(3) if j!=i])-self.edges[i].length()**2)/(2*np.prod([self.edges[j].length() for j in range(3) if j!=i])))
                        break
                angle1=0
                for i in range(3):
                    if not self.edges[i].points[0].is_point(self.points[1]) and not self.edges[i].points[1].is_point(self.points[1]):
                        angle1=math.acos((sum([self.edges[j].length()**2 for j in range(3) if j!=i])-self.edges[i].length()**2)/(2*np.prod([self.edges[j].length() for j in range(3) if j!=i])))
                        break
                angle2=math.pi-angle0-angle1
            else:
                a=math.sqrt((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2)
                b=math.sqrt((self.points[2].x-self.points[0].x)**2+(self.points[2].y-self.points[0].y)**2)
                c=math.sqrt((self.points[2].x-self.points[1].x)**2+(self.points[2].y-self.points[1].y)**2)
                angle0=math.acos((a**2+b**2-c**2)/(2*a*b))
                angle1=math.acos((a**2+c**2-b**2)/(2*a*c))
                angle2=math.pi-angle0-angle1
            self.interiorAngles=[angle0,angle1,angle2]
        return self.interiorAngles
    
    def circumcircle(self):
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
            self.circumCircle=circle(point(Sx/a,Sy/a),math.sqrt(b/a+(Sx**2+Sy**2)/a**2))
        return self.circumCircle
    
    def inCircumcircle(self,pointVar,includeboundary=True,errtol=1e-12):
        if (self.points[1].x-self.points[0].x)*(self.points[2].y-self.points[1].y)-(self.points[1].y-self.points[0].y)*(self.points[2].x-self.points[1].x)>0:
            res=npla.det([[self.points[0].x-pointVar.x,self.points[1].x-pointVar.x,self.points[2].x-pointVar.x],
                          [self.points[0].y-pointVar.y,self.points[1].y-pointVar.y,self.points[2].y-pointVar.y],
                          [(self.points[0].x-pointVar.x)**2+(self.points[0].y-pointVar.y)**2,(self.points[1].x-pointVar.x)**2+(self.points[1].y-pointVar.y)**2,(self.points[2].x-pointVar.x)**2+(self.points[2].y-pointVar.y)**2]])
        else:
            res=npla.det([[self.points[0].x-pointVar.x,self.points[2].x-pointVar.x,self.points[1].x-pointVar.x],
                          [self.points[0].y-pointVar.y,self.points[2].y-pointVar.y,self.points[1].y-pointVar.y],
                          [(self.points[0].x-pointVar.x)**2+(self.points[0].y-pointVar.y)**2,(self.points[2].x-pointVar.x)**2+(self.points[2].y-pointVar.y)**2,(self.points[1].x-pointVar.x)**2+(self.points[1].y-pointVar.y)**2]])
        
        if includeboundary:
            return res>=-abs(errtol)
        else:
            return res>abs(errtol)
    
    def __repr__(self):
        return "Triangle <"+self.points[0].__repr__()+", "+self.points[1].__repr__()+", "+self.points[2].__repr__()+">"
    
    def copy(self,blank=True):
        t=triangle(self.points[0].copy(),self.points[1].copy(),self.points[2].copy())
        if self.triagarea is not None:
            t.triagarea=self.triagarea
        if self.interiorAngles is not None:
            t.interiorAngles=self.interiorAngles
        if self.circumcircle is not None:    
            t.circumcircle=self.circumcircle.copy() 
        return t
    
    def draw(self,plotaxis,points=True,edges=True,fill=True,color="black",alpha=1):    
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
                
    def kill(self,T=None):
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
        if self.circumcircle is not None:
            del self.circumcircle
        if isinstance(T,list):
            try:
                T.remove(self)
            except ValueError:
                pass
        del self
