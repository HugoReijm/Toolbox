import math
import matplotlib.colors as pltcolors

class point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.edges=[]
        self.triangles=[]
        self.index=-1
    
    def is_point(self,p,errtol=1e-12):
        return isinstance(p,point) and abs(self.x-p.x)<abs(errtol) and abs(self.y-p.y)<abs(errtol)
    
    def distance(self,p):
        return math.sqrt((self.x-p.x)**2+(self.y-p.y)**2)
    
    def __repr__(self):
        return "Point ("+str(round(self.x,6))+", "+str(round(self.y,6))+")"
              
    def copy(self):
        return point(self.x,self.y)
    
    def kill(self,P=None):
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
        if isinstance(P,list):
            try:
                P.remove(self)
            except ValueError:
                pass
        del self
