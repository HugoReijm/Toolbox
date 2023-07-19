import numpy as np
import numpy.linalg as npla
from toolbox.point import point
from toolbox.edge import edge
from toolbox.triangle import triangle
from toolbox.sphere import sphere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as pltcolors

class tetrahedron(object):
    def __init__(self,p1,p2,p3,p4,e1,e2,e3,e4,e5,e6,t1,t2,t3,t4):
        self.points=[p1,p2,p3,p4]
        self.tetraAv=None
        self.edges=[e1,e2,e3,e4,e5,e6]
        self.triangles=[t1,t2,t3,t4]
        self.order_triangles()
        self.neighbors=[None,None,None,None]
        self.tetraAv=None
        self.tetraVolume=None
        self.index=-1
        self.circumSphere=None
        
    def order_triangles(self):
        #This method sorts triangles to the tetrahedron object in such a way that
        #point 0 does not intersect with triangle 0, point 1 does not intersect
        #with triangle 1, point 2 does not intersect with triangle 2, and point 3
        #does not intersect with triangle 3
        temp=[]
        
        if not self.triangles[0].point_triangle_intersect(self.points[0]):
            temp.append(self.triangles[0])
        elif not self.triangles[1].point_triangle_intersect(self.points[0]):
            temp.append(self.triangles[1])
        elif not self.triangles[2].point_triangle_intersect(self.points[0]):
            temp.append(self.triangles[2])
        else:
            temp.append(self.triangles[3])

        if not self.triangles[0].point_triangle_intersect(self.points[1]):
            temp.append(self.triangles[0])
        elif not self.triangles[1].point_triangle_intersect(self.points[1]):
            temp.append(self.triangles[1])
        elif not self.triangles[2].point_triangle_intersect(self.points[1]):
            temp.append(self.triangles[2])
        else:
            temp.append(self.triangles[3])

        if not self.triangles[0].point_triangle_intersect(self.points[2]):
            temp.append(self.triangles[0])
        elif not self.triangles[1].point_triangle_intersect(self.points[2]):
            temp.append(self.triangles[1])
        elif not self.triangles[2].point_triangle_intersect(self.points[2]):
            temp.append(self.triangles[2])
        else:
            temp.append(self.triangles[3])
        
        if not self.triangles[0].point_triangle_intersect(self.points[3]):
            temp.append(self.triangles[0])
        elif not self.triangles[1].point_triangle_intersect(self.points[3]):
            temp.append(self.triangles[1])
        elif not self.triangles[2].point_triangle_intersect(self.points[3]):
            temp.append(self.triangles[2])
        else:
            temp.append(self.triangles[3])
        
        self.triangles=temp

    def is_tetrahedron(self,tet,errtol=1e-12):
        #This method compares whether two tetrahedra are equal or not.
        if isinstance(tet,tetrahedron):
            for i in range(4):
                if self.points[0].is_point(tet.points[i],errtol=errtol):
                    for j in range(4):
                        if j!=i and self.points[1].is_point(tet.points[j],errtol=errtol):
                            for k in range(4):
                                if k!=i and k!=j and self.points[2].is_point(tet.points[k],errtol=errtol):
                                    for l in range(4):
                                        if l!=i and l!=j and l!=k and self.points[3].is_point(tet.points[l],errtol=errtol):
                                            return True
        return False
    
    def update_network(self):
        #For each end-point of the tetrahedron object, this method adds
        #the tetrahedron to the point's list of tetrahedra it is a constituent of.
        #For each edge of the tetrahedron object, this method adds
        #the tetrahedron to the edge's list of tetrahedra it is a constituent of.
        #For each faces of the tetrahedron object, this method adds
        #the tetrahedron to the triangles's list of tetrahedra it is a constituent of.
        for p in self.points:
            p.tetrahedra.append(self)
        for e in self.edges:
            e.tetrahedra.append(self)
        for i,t_i in enumerate(self.triangles):
            t_i.tetrahedra.append(self)
            for s in t_i.tetrahedra:
                if not s.is_tetrahedron(self):
                    self.neighbors[i]=s
                    for j,t_j in enumerate(s.triangles):
                        if t_j.is_triangle(t_i):
                            s.neighbors[j]=self
                            break
                break

    def average(self):
        #This method computes the centroid of the tetrahedron object
        #and makes the centroid variable a singleton.
        if self.tetraAv is None:
            self.tetraAv=(self.points[0]+self.points[1]+self.points[2]+self.points[3]).scalar_div(4.0)
        return self.tetraAv
    
    def volume(self):
        #This method computes the volume of the tetrahedron object
        #and makes the volume variable a singleton.
        if self.tetraVolume is None:
            self.tetraVolume=npla.det(np.array([[self.points[1].x-self.points[0].x,self.points[2].x-self.points[0].x,self.points[3].x-self.points[0].x],
                                                [self.points[1].y-self.points[0].y,self.points[2].y-self.points[0].y,self.points[3].y-self.points[0].y],
                                                [self.points[1].z-self.points[0].z,self.points[2].z-self.points[0].z,self.points[3].z-self.points[0].z]]))
            self.tetraVolume=abs(self.tetraVolume)/6.0
        return self.tetraVolume

    def point_tetrahedron_intersect(self,p,includeboundary=True,errtol=1e-12):
        errtol=abs(errtol)
        
        A=np.array([[self.points[1].x-self.points[0].x,self.points[2].x-self.points[0].x,self.points[3].x-self.points[0].x],
                    [self.points[1].y-self.points[0].y,self.points[2].y-self.points[0].y,self.points[3].y-self.points[0].y],
                    [self.points[1].z-self.points[0].z,self.points[2].z-self.points[0].z,self.points[3].z-self.points[0].z]])
        b=np.array([p.x-self.points[0].x,p.y-self.points[0].y,p.z-self.points[0].z])
        try:
            tau=np.linalg.solve(A,b)
            if ((includeboundary
                 and -errtol<=tau[0] and -errtol<=tau[1] and -errtol<=tau[2]
                 and tau[0]+tau[1]+tau[2]<=1+errtol)
                or (not includeboundary
                    and errtol<tau[0] and errtol<tau[1] and errtol<tau[2]
                 and tau[0]+tau[1]+tau[2]<1-errtol)):
                return True
            return False
        except Exception:
            return False
    
    def edge_tetrahedron_intersect(self,e,includeboundary=True,errtol=1e-12):
        #This method returns WHETHER an edge intersects the tetrahedron object or not.
        #The method checks whether the objects intersect or not by sorting
        #through a number of different possibilities.
        if includeboundary:
            #If we include the boundary into the intersection, 
            #the method easily checks whether there is an intersection.
            if any([self.point_tetrahedron_intersect(p,errtol=errtol) for p in e.points]):
                return True
            elif any([e.edge_triangle_intersect(t,errtol=errtol) for t in self.triangles]):
                return True
        else:
            #If we don't include the boundary into the intersection,
            #the method runs through a few different scenarios to see
            #if there is an intersection or not.
            state=1
            for i in range(2):
                point_triangle_bool=any([t.point_triangle_intersect(e.points[i],errtol=errtol) for t in self.triangles])
                point_point_bool=any([p.is_point(e.points[i]) for p in self.points])
                if point_triangle_bool and not point_point_bool:
                    state*=1
                elif point_point_bool:
                    state*=2
                else:
                    state*=3
            
            if state==1 or state==2:
                return not any([t.point_triangle_intersect(e.points[0],errtol=errtol)
                                and t.point_triangle_intersect(e.points[1],errtol=errtol) for t in self.triangles])
            elif state==3 or state==6:
                #return any([self.point_tetrahedron_intersect(e.points[0],includeboundary=False,errtol=errtol)
                #            or self.point_tetrahedron_intersect(e.points[1],includeboundary=False,errtol=errtol)])
                if (any([self.point_tetrahedron_intersect(e.points[0],includeboundary=False,errtol=errtol)
                    or self.point_tetrahedron_intersect(e.points[1],includeboundary=False,errtol=errtol)])):
                    return True
                else:
                    for s_triangle in self.triangles:
                        if (s_triangle.edge_triangle_intersect(e,includeboundary=False,errtol=errtol)
                            and (not s_triangle.point_triangle_intersect(e.points[0],errtol=errtol)
                            or not s_triangle.point_triangle_intersect(e.points[1],errtol=errtol))):
                            return True
            elif state==9:
                return any([t.edge_triangle_intersect(e,includeboundary=False,errtol=errtol) for t in self.triangles])
        return False
    
    def triangle_tetrahedron_intersect(self,t,includeboundary=True,errtol=1e-12):
        #This method returns whether an inputted triangle intersects
        #the orginal tetrahedron object or not.
        #Distinction is made whether to include the boundaries or not.
        return (any([t.edge_triangle_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in self.edges])
             or any([self.edge_tetrahedron_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in t.edges]))
    
    def tetrahedron_tetrahedron_intersect(self,tet,includeboundary=True,errtol=1e-12):
        #This method returns whether an inputted tetrahedron intersects
        #the orginal tetrahedron object or not.
        #Distinction is made whether to include the boundaries or not.
        return (any([tet.triangle_tetrahedron_intersect(t,includeboundary=includeboundary,errtol=errtol) for t in self.triangles])
             or any([self.triangle_tetrahedron_intersect(t,includeboundary=includeboundary,errtol=errtol) for t in tet.triangles]))
    
    def circumsphere(self):
        #This method computes the circumsphere of the tetrahedron object and
        #makes the circumsphere variable a singleton.
        if self.circumSphere is None:
            try:
                tau=npla.solve(np.array([[self.points[0].x,self.points[0].y,self.points[0].z,1],
                                         [self.points[1].x,self.points[1].y,self.points[1].z,1],
                                         [self.points[2].x,self.points[2].y,self.points[2].z,1],
                                         [self.points[3].x,self.points[3].y,self.points[3].z,1]]),
                               np.array([self.points[0].x**2+self.points[0].y**2+self.points[0].z**2,
                                         self.points[1].x**2+self.points[1].y**2+self.points[1].z**2,
                                         self.points[2].x**2+self.points[2].y**2+self.points[2].z**2,
                                         self.points[3].x**2+self.points[3].y**2+self.points[3].z**2]))
                center=point(tau[0]/2.0,tau[1]/2.0,tau[2]/2.0)
                self.circumSphere=sphere(center,np.sqrt(tau[0]**2+tau[1]**2+tau[2]**2+4*tau[3])/2.0)
            except Exception:
                pass
        return self.circumSphere

    def inCircumsphere(self,p,includeboundary=True,errtol=1e-12):
        #This method calculates whether an inputted point lies inside of the
        #tetrahedron object's circumcircle.
        if includeboundary:
            return self.circumsphere().center.distance(p)<=self.circumsphere().radius+abs(errtol)
        else:
            return self.circumsphere().center.distance(p)<self.circumsphere().radius-abs(errtol)

    def __repr__(self):
        return "Tetrahedron "+str(self.index)+": {"+self.points[0].__repr__()+", "+self.points[1].__repr__()+", "+self.points[2].__repr__()+", "+self.points[3].__repr__()+"}"
    
    def copy(self):
        #This method returns a copy of the tetrahedron object.
        #Whether the copy contains all variables of the original used for
        #constructing tetrahedralizations is user-designated.
        #The tetrahedron object's edge and triangles list are not copied.
        tet=tetrahedron(self.points[0].copy(),self.points[1].copy(),
                        self.points[2].copy(),self.points[3].copy())
        if self.tetraAv is not None:
            tet.tetraAv=self.tetraAv.copy()
        if self.tetraVolume is not None:
            tet.tetraVolume=self.tetraVolume
        if self.circumSphere is not None:    
            tet.circumSphere=self.circumSphere.copy() 
        return tet
    
    def draw(self,plotaxis,points=True,edges=True,fill=True,elev=10,azim=-75,color="black",alpha=1):    
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
            plotaxis.add_collection3d(Poly3DCollection([[[self.points[i].x,self.points[i].y,self.points[i].z] for i in [0,1,2]],
                                                        [[self.points[i].x,self.points[i].y,self.points[i].z] for i in [0,1,3]],
                                                        [[self.points[i].x,self.points[i].y,self.points[i].z] for i in [0,2,3]],
                                                        [[self.points[i].x,self.points[i].y,self.points[i].z] for i in [1,2,3]]],
                                                       facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
        else:
            if edges:
                plotaxis.plot3D([self.points[i].x for i in [0,1,2,0,3,1,2,3]],
                                [self.points[i].y for i in [0,1,2,0,3,1,2,3]],
                                [self.points[i].z for i in [0,1,2,0,3,1,2,3]],
                                color=color,alpha=alpha,zorder=0)

        if points:
            if fill:
                plotaxis.scatter3D([p.x for p in self.points],
                                   [p.y for p in self.points],
                                   [p.z for p in self.points],
                                   facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
            else:
                plotaxis.scatter3D([p.x for p in self.points],
                                   [p.y for p in self.points],
                                   [p.z for p in self.points],
                                   facecolor=color,edgecolor=color,alpha=alpha,zorder=1)
        plotaxis.view_init(elev=elev,azim=azim)
