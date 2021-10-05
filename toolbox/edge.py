import math
import numpy.linalg as npla
from point import point
from circle import circle
import matplotlib.colors as pltcolors

class edge(object):
    def __init__(self,p1,p2):
        self.points=[p1,p2]
        self.triangles=[]
        self.edgelength=None
        self.enclosed=False
        self.constraint=False
        self.index=-1
        self.circumCircle=None
        
    def is_edge(self,e,errtol=1e-12):
        return isinstance(e,edge) and ((self.points[0].is_point(e.points[0],errtol=errtol) and self.points[1].is_point(e.points[1],errtol=errtol)) or (self.points[0].is_point(e.points[1],errtol=errtol) and self.points[1].is_point(e.points[0],errtol=errtol)))
        
    def update(self):
        for p in self.points:
            p.edges.append(self)
        
    def length(self):
        if self.edgelength is None:
            self.edgelength=math.sqrt((self.points[0].x-self.points[1].x)**2+(self.points[0].y-self.points[1].y)**2)
        return self.edgelength
    
    def cross(self,edge):
        return (self.points[1].x-self.points[0].x)*(edge.points[1].y-edge.points[0].y)-(self.points[1].y-self.points[0].y)*(edge.points[1].x-edge.points[0].x)
    
    def point_edge_intersect(self,pointVar,includeboundary=True,errtol=1e-12):
        if includeboundary:
            if abs((self.points[1].x-self.points[0].x)*(pointVar.y-self.points[0].y)-(pointVar.x-self.points[0].x)*(self.points[1].y-self.points[0].y))<abs(errtol):
                if ((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2-(pointVar.x-self.points[0].x)**2-(pointVar.y-self.points[0].y)**2>=-abs(errtol)
                    and (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2-(pointVar.x-self.points[1].x)**2-(pointVar.y-self.points[1].y)**2>=-abs(errtol)):
                    return True
        elif abs(pointVar.x-self.points[0].x)>abs(errtol) or abs(pointVar.y-self.points[0].y)>abs(errtol):
            if abs((self.points[1].x-self.points[0].x)*(pointVar.y-self.points[0].y)-(pointVar.x-self.points[0].x)*(self.points[1].y-self.points[0].y))<abs(errtol):
                if ((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2-(pointVar.x-self.points[0].x)**2-(pointVar.y-self.points[0].y)**2>abs(errtol)
                    and (self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2-(pointVar.x-self.points[1].x)**2-(pointVar.y-self.points[1].y)**2>abs(errtol)):
                    return True
        return False

    def edge_edge_intersect(self,edgeVar,includeboundary=True,boolean=True,errtol=1e-12):
        if self.is_edge(edgeVar):
            if boolean:
                return True
            return self
        
        det=(self.points[1].x-self.points[0].x)*(edgeVar.points[0].y-edgeVar.points[1].y)-(self.points[1].y-self.points[0].y)*(edgeVar.points[0].x-edgeVar.points[1].x)
        if abs(det)>abs(errtol):
            s1=((edgeVar.points[0].y-edgeVar.points[1].y)*(edgeVar.points[0].x-self.points[0].x)+(edgeVar.points[1].x-edgeVar.points[0].x)*(edgeVar.points[0].y-self.points[0].y))/det
            s2=((self.points[0].y-self.points[1].y)*(edgeVar.points[0].x-self.points[0].x)+(self.points[1].x-self.points[0].x)*(edgeVar.points[0].y-self.points[0].y))/det
            if (includeboundary and -abs(errtol)<=s1<=1+abs(errtol) and -abs(errtol)<=s2<=1+abs(errtol)) or (not includeboundary and abs(errtol)<s1<1-abs(errtol) and abs(errtol)<s2<1-abs(errtol)):
                if boolean:
                    return True
                return point(self.points[0].x+s1*(self.points[1].x-self.points[0].x),self.points[0].y+s1*(self.points[1].y-self.points[0].y))
        elif abs(self.cross(edge(self.points[1],edgeVar.points[0])))<=abs(errtol):
            if abs(self.points[1].x-self.points[0].x)>=abs(self.points[1].y-self.points[0].y):
                if self.points[1].x>self.points[0].x:
                    min1,max1=0,1
                else:
                    min1,max1=1,0
                if edgeVar.points[1].x>edgeVar.points[0].x:
                    min2,max2=0,1
                else:
                    min2,max2=1,0
                if includeboundary:
                    if self.points[min1].x-abs(errtol)<=edgeVar.points[min2].x<=self.points[max1].x+abs(errtol) and self.points[max1].x<=edgeVar.points[max2].x+abs(errtol):
                        if boolean:
                            return True
                        if abs(self.points[max1].x-edgeVar.points[min2].x)<=abs(errtol):
                            return self.points[max1].copy()
                        return edge(edgeVar.points[min2],self.points[max1])
                    elif edgeVar.points[min2].x<=self.points[min1].x+abs(errtol) and self.points[min1].x-abs(errtol)<=edgeVar.points[max2].x<=self.points[max1].x+abs(errtol):
                        if boolean:
                            return True
                        if abs(self.points[min1].x-edgeVar.points[max2].x)<=abs(errtol):
                            return self.points[min1].copy()
                        return edge(self.points[min1],edgeVar.points[max2])
                    elif self.points[min1].x-abs(errtol)<=edgeVar.points[min2].x<=self.points[max1].x+abs(errtol) and self.points[min1].x-abs(errtol)<=edgeVar.points[max2].x<=self.points[max1].x+abs(errtol):
                        if boolean:
                            return True
                        return edge(edgeVar.points[min2],edgeVar.points[max2])
                    elif edgeVar.points[min2].x-abs(errtol)<=self.points[min1].x<=edgeVar.points[max2].x+abs(errtol) and edgeVar.points[min2].x-abs(errtol)<=self.points[max1].x<=edgeVar.points[max2].x+abs(errtol):
                        if boolean:
                            return True
                        return edge(self.points[min1],self.points[max1])
                else:
                    if self.points[min1].x-abs(errtol)<=edgeVar.points[min2].x<self.points[max1].x-abs(errtol) and self.points[max1].x<=edgeVar.points[max2].x+abs(errtol):
                        if boolean:
                            return True
                        return edge(edgeVar.points[min2],self.points[max1])
                    elif edgeVar.points[min2].x<=self.points[min1].x+abs(errtol) and self.points[min1].x+abs(errtol)<edgeVar.points[max2].x<=self.points[max1].x+abs(errtol):
                        if boolean:
                            return True
                        return edge(self.points[min1],edgeVar.points[max2])
                    elif self.points[min1].x-abs(errtol)<=edgeVar.points[min2].x<=self.points[max1].x+abs(errtol) and self.points[min1].x-abs(errtol)<=edgeVar.points[max2].x<=self.points[max1].x+abs(errtol):
                        if boolean:
                            return True
                        return edge(edgeVar.points[min2],edgeVar.points[max2])
                    elif edgeVar.points[min2].x-abs(errtol)<=self.points[min1].x<=edgeVar.points[max2].x+abs(errtol) and edgeVar.points[min2].x-abs(errtol)<=self.points[max1].x<=edgeVar.points[max2].x+abs(errtol):
                        if boolean:
                            return True
                        return edge(self.points[min1],self.points[max1])
            else:
                if self.points[1].y>self.points[0].y:
                    min1,max1=0,1
                else:
                    min1,max1=1,0
                if edgeVar.points[1].y>edgeVar.points[0].y:
                    min2,max2=0,1
                else:
                    min2,max2=1,0
                if includeboundary:
                    if self.points[min1].y-abs(errtol)<=edgeVar.points[min2].y<=self.points[max1].y+abs(errtol) and self.points[max1].y<=edgeVar.points[max2].y+abs(errtol):
                        if boolean:
                            return True
                        if abs(self.points[max1].y-edgeVar.points[min2].y)<=abs(errtol):
                            return self.points[max1].copy()
                        return edge(edgeVar.points[min2],self.points[max1])
                    elif edgeVar.points[min2].y<=self.points[min1].y+abs(errtol) and self.points[min1].y-abs(errtol)<=edgeVar.points[max2].y<=self.points[max1].y+abs(errtol):
                        if boolean:
                            return True
                        if abs(self.points[min1].y-edgeVar.points[max2].y)<=abs(errtol):
                            return self.points[min1].copy()
                        return edge(self.points[min1],edgeVar.points[max2])
                    elif self.points[min1].y-abs(errtol)<=edgeVar.points[min2].y<=self.points[max1].y+abs(errtol) and self.points[min1].y-abs(errtol)<=edgeVar.points[max2].y<=self.points[max1].y+abs(errtol):
                        if boolean:
                            return True
                        return edge(edgeVar.points[min2],edgeVar.points[max2])
                    elif edgeVar.points[min2].y-abs(errtol)<=self.points[min1].y<=edgeVar.points[max2].y+abs(errtol) and edgeVar.points[min2].y-abs(errtol)<=self.points[max1].y<=edgeVar.points[max2].y+abs(errtol):
                        if boolean:
                            return True
                        return edge(self.points[min1],self.points[max1])
                else:
                    if self.points[min1].y-abs(errtol)<=edgeVar.points[min2].y<self.points[max1].y-abs(errtol) and self.points[max1].y<=edgeVar.points[max2].y+abs(errtol):
                        if boolean:
                            return True
                        return edge(edgeVar.points[min2],self.points[max1])
                    elif edgeVar.points[min2].y<=self.points[min1].y+abs(errtol) and self.points[min1].y+abs(errtol)<edgeVar.points[max2].y<=self.points[max1].y+abs(errtol):
                        if boolean:
                            return True
                        return edge(self.points[min1],edgeVar.points[max2])
                    elif self.points[min1].y-abs(errtol)<=edgeVar.points[min2].y<=self.points[max1].y+abs(errtol) and self.points[min1].y-abs(errtol)<=edgeVar.points[max2].y<=self.points[max1].y+abs(errtol):
                        if boolean:
                            return True
                        return edge(edgeVar.points[min2],edgeVar.points[max2])
                    elif edgeVar.points[min2].y-abs(errtol)<=self.points[min1].y<=edgeVar.points[max2].y+abs(errtol) and edgeVar.points[min2].y-abs(errtol)<=self.points[max1].y<=edgeVar.points[max2].y+abs(errtol):
                        if boolean:
                            return True
                        return edge(self.points[min1],self.points[max1])
        if boolean:
            return False
        return None
    
    def circumcircle(self):
        if self.circumCircle is None:
            self.circumCircle=circle(point((self.points[0].x+self.points[1].x)/2,(self.points[0].y+self.points[1].y)/2),
                                     math.sqrt((self.points[1].x-self.points[0].x)**2+(self.points[1].y-self.points[0].y)**2)/2)
        return self.circumCircle
    
    def inCircumcircle(self,pointVar,includeboundary=True,errtol=1e-12):
        pointd=point((self.points[0].x+self.points[1].x)/2,(self.points[0].y+self.points[1].y)/2)
        pointc=point(self.points[0].y-pointd.y+pointd.x,pointd.x-self.points[0].x+pointd.y)
        if (self.points[1].x-self.points[0].x)*(pointc.y-self.points[1].y)-(self.points[1].y-self.points[0].y)*(pointc.x-self.points[1].x)>0:
            res=npla.det([[self.points[0].x-pointVar.x,self.points[1].x-pointVar.x,pointc.x-pointVar.x],
                          [self.points[0].y-pointVar.y,self.points[1].y-pointVar.y,pointc.y-pointVar.y],
                          [(self.points[0].x-pointVar.x)**2+(self.points[0].y-pointVar.y)**2,(self.points[1].x-pointVar.x)**2+(self.points[1].y-pointVar.y)**2,(pointc.x-pointVar.x)**2+(pointc.y-pointVar.y)**2]])
        else:
            res=npla.det([[self.points[0].x-pointVar.x,pointc.x-pointVar.x,self.points[1].x-pointVar.x],
                          [self.points[0].y-pointVar.y,pointc.y-pointVar.y,self.points[1].y-pointVar.y],
                          [(self.points[0].x-pointVar.x)**2+(self.points[0].y-pointVar.y)**2,(pointc.x-pointVar.x)**2+(pointc.y-pointVar.y)**2,(self.points[1].x-pointVar.x)**2+(self.points[1].y-pointVar.y)**2]])
        if includeboundary:
            return res>=-abs(errtol)
        else:
            return res>abs(errtol)
    
    def swap(self):
        tempP=self.points[0]
        self.points[0]=self.points[1]
        self.points[1]=tempP
    
    def __repr__(self):
        return "Edge <"+self.points[0].__repr__()+", "+self.points[1].__repr__()+">"
    
    def copy(self,blank=True):
        e=edge(self.points[0].copy(),self.points[1].copy())
        if self.edgelength is not None:
            e.edgelength=self.edgelength
        if self.circumcircle is not None:
            e.circumcircle=self.circumcircle.copy()
        if not blank:
            e.enclosed=self.enclosed
            e.constraint=self.constraint
        return e

    def draw(self,plotaxis,points=True,color="black",alpha=1):
        plotaxis.plot([self.points[0].x,self.points[1].x],
                      [self.points[0].y,self.points[1].y],color=color,alpha=alpha)
        if points:
            if sum(pltcolors.to_rgb(color))<=1.0:
                plotaxis.scatter([self.points[0].x,self.points[1].x],
                                 [self.points[0].y,self.points[1].y],facecolor=color,edgecolor="white",alpha=alpha)
            else:
                plotaxis.scatter([self.points[0].x,self.points[1].x],
                                 [self.points[0].y,self.points[1].y],facecolor=color,edgecolor="black",alpha=alpha)
        
    def kill(self,E=None):
        for p in self.points:
            try:
                p.edges.remove(self)
            except ValueError:
                pass
        for t in self.triangles:
            try:
                t.edges.remove(self)
            except ValueError:
                pass
        if self.circumcircle is not None:
            del self.circumcircle
        if isinstance(E,list):
            try:
                E.remove(self)
            except ValueError:
                pass
        del self
