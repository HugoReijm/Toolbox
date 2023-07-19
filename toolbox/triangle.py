import numpy as np
from toolbox.point import point
from toolbox.edge import edge
from toolbox.sphere import sphere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as pltcolors

#This file contains the triangle class and its various methods.
class triangle(object):
    def __init__(self,p1,p2,p3,e1,e2,e3):
        #This list keeps track of which points constitute the end-points of the triangle object.
        self.points=[p1,p2,p3]
        self.triagAv=None
        #This list keeps track of which edge constitute the boundaries of the triangle object.
        self.edges=[e1,e2,e3]
        self.tetrahedra=[]
        self.order_edges()
        self.neighbors=[None,None,None]
        self.interiorAngles=None
        self.triagArea=None
        self.enclosed=True
        self.constraint=False
        self.index=-1
        self.circumSphere=None
            
    def order_edges(self):
        #This method sorts edges to the triangle object in such a way that
        #point 0 does not intersect with edge 0, point 1 does not intersect
        #with edge 1, and point 2 does not intersect with edge 2.
        temp=[]
        
        if not self.edges[0].point_edge_intersect(self.points[0]):
            temp.append(self.edges[0])
        elif not self.edges[1].point_edge_intersect(self.points[0]):
            temp.append(self.edges[1])
        else:
            temp.append(self.edges[2])

        if not self.edges[0].point_edge_intersect(self.points[1]):
            temp.append(self.edges[0])
        elif not self.edges[1].point_edge_intersect(self.points[1]):
            temp.append(self.edges[1])
        else:
            temp.append(self.edges[2])

        if not self.edges[0].point_edge_intersect(self.points[2]):
            temp.append(self.edges[0])
        elif not self.edges[1].point_edge_intersect(self.points[2]):
            temp.append(self.edges[1])
        else:
            temp.append(self.edges[2])
        
        self.edges=temp
    
    def is_triangle(self,t,errtol=1e-12):
        #This method compares whether two triangles are equal or not.
        for i in range(3):
            if self.points[0].is_point(t.points[i],errtol=errtol):
                for j in range(3):
                    if j!=i and self.points[1].is_point(t.points[j],errtol=errtol):
                        for k in range(3):
                            if k!=i and k!=j and self.points[2].is_point(t.points[k],errtol=errtol):
                                return True
        return False
                
    def update_network(self):
        #For each end-point of the triangle object, this method adds the triangle to the point's list of triangles it is a constituent of.
        #For each boundary edge of the triangle object, this method adds the triangle to the edge's list of triangles it is a constituent of.
        for p in self.points:
            p.triangles.append(self)
        for i,e_i in enumerate(self.edges):
            e_i.triangles.append(self)
            for t in e_i.triangles:
                if not t.is_triangle(self):
                    self.neighbors[i]=t
                    for j,e_j in enumerate(t.edges):
                        if e_j.is_edge(e_i):
                            t.neighbors[j]=self
                            break
                break
    
    def average(self):
        #This method computes the centroid of the triangle object
        #and makes the centroid variable a singleton.
        if self.triagAv is None:
            self.triagAv=(self.points[0]+self.points[1]+self.points[2]).scalar_div(3.0)
        return self.triagAv
    
    def area(self):
        #This method computes the area of the triangle object
        #and makes the area variable a singleton.
        if self.triagArea is None:
            self.triagArea=self.edges[0].cross(self.edges[1]).length()/2.0
        return self.triagArea
    
    def angles(self):
        #This method computes the interior angles of the triangle object and makes the interior angles object a singleton.
        if self.interiorAngles is None:
            angle0=self.edges[1].angle(self.edges[2])
            angle1=self.edges[0].angle(self.edges[2])
            self.interiorAngles=[angle0,angle1,np.pi-angle0-angle1]
        return self.interiorAngles
    
    def point_triangle_intersect(self,p,includeboundary=True,errtol=1e-6):
        #This method returns whether a point is inside the triangle object or not.
        #Distinction is made whether to include the boundary edges or not.
        
        errtol=abs(errtol)
        #The triangle uniquely defines a plane, one that the inputted point
        #could fall into. The method sets up the equations for the plane to
        #contain the piont, and rearranges them into a 3x2 system of equations...
        A=np.array([[self.points[1].x-self.points[0].x,self.points[2].x-self.points[0].x],
                    [self.points[1].y-self.points[0].y,self.points[2].y-self.points[0].y],
                    [self.points[1].z-self.points[0].z,self.points[2].z-self.points[0].z]])
        b=np.array([p.x-self.points[0].x,p.y-self.points[0].y,p.z-self.points[0].z])
        
        #...which it then tries to solve. If this is solvable, the method returns
        #the intersection point. If this isn't solvable, this means that the
        #point can not be in the triangle at all.        
        for i in range(0,2):
            for j in range(i+1,3):
                if abs(A[i,0]*A[j,1]-A[i,1]*A[j,0])>errtol:
                    tau=np.linalg.solve(A[[i,j]],b[[i,j]])
                    if max(abs(np.dot(A,tau)-b))<errtol:
                        #If the point is found in the plane, then the method
                        #needs to check if the point is also found in the triangle
                        #It does this by checking whether the solution of the
                        #linear equation satisfies certain requirements.
                        if ((includeboundary
                             and -errtol<=tau[0]
                             and -errtol<=tau[1]
                             and tau[0]+tau[1]<=1+errtol)
                            or (not includeboundary
                                and errtol<tau[0]
                                and errtol<tau[1]
                                and tau[0]+tau[1]<1-errtol)):
                            return True
                        else:
                            return False
                    else:
                        return False
        return False
        
    def edge_triangle_intersect(self,e,includeboundary=True,errtol=1e-12):
        #This method returns WHETHER an edge intersects the triangle object or not.
        #The method first checks whether the triangle and edge objects are coplanar...
        normal=self.edges[0].cross(self.edges[1])
        normal.points[0],normal.points[1]=normal.points[0]-normal.points[1],point(0,0,0)
        tau_1=normal.dot(edge(self.points[0]-e.points[0],point(0,0,0)))
        tau_2=normal.dot(edge(e.points[1]-e.points[0],point(0,0,0)))
        if abs(tau_2)<=errtol:
            if abs(tau_1)<=errtol:
                #If the triangle and edge objects are coplanar, then the method
                #checks whether the coplanar objects intersect or not by sorting
                #through a number of different possibilities.
                if includeboundary:
                    #If we include the boundary into the intersection, 
                    #the method easily checks whether there is an intersection.
                    if any([self.point_triangle_intersect(p,errtol=errtol) for p in e.points]):
                        return True
                    elif any([e.edge_edge_intersect(t_edge,errtol=errtol) for t_edge in self.edges]):
                        return True
                else:
                    #If we don't include the boundary into the intersection,
                    #the method runs through a few different scenarios to see
                    #if there is an intersection or not.
                    state=1
                    for i in range(2):
                        point_edge_bool=any([t_edge.point_edge_intersect(e.points[i],errtol=errtol) for t_edge in self.edges])
                        point_point_bool=any([p.is_point(e.points[i]) for p in self.points])
                        if point_edge_bool and not point_point_bool:
                            state*=1
                        elif point_point_bool:
                            state*=2
                        else:
                            state*=3
                    
                    if state==1 or state==2:
                        return not any([t_edge.point_edge_intersect(e.points[0],errtol=errtol)
                                        and t_edge.point_edge_intersect(e.points[1],errtol=errtol) for t_edge in self.edges])
                    elif state==3 or state==6:
                        #return any([self.point_triangle_intersect(e.points[0],includeboundary=False,errtol=errtol)
                        #            or self.point_triangle_intersect(e.points[1],includeboundary=False,errtol=errtol)])
                        if (any([self.point_triangle_intersect(e.points[0],includeboundary=False,errtol=errtol)
                            or self.point_triangle_intersect(e.points[1],includeboundary=False,errtol=errtol)])):
                            return True
                        else:
                            for t_edge in self.edges:
                                if (t_edge.edge_edge_intersect(e,includeboundary=False,errtol=errtol)
                                    and (not t_edge.point_edge_intersect(e.points[0],errtol=errtol)
                                    or not t_edge.point_edge_intersect(e.points[1],errtol=errtol))):
                                    return True
                    elif state==9:
                        return any([e.edge_edge_intersect(t_edge,includeboundary=False,errtol=errtol) for t_edge in self.edges])
                return False
        else:
            #If edge object is not coplanar to the triangle object, the edge
            #must intersect the plane that contains the triangle. The method
            #then checks if that point fits inside the triangle or not.
            tau=tau_1/tau_2
            if ((includeboundary and -errtol<=tau<=1+errtol)
                or (not includeboundary and errtol<tau<1-errtol)):
                if self.point_triangle_intersect(e.points[0]+(e.points[1]-e.points[0]).scalar_mul(tau),
                                                 includeboundary=includeboundary):
                    return True
        return False
        
    def triangle_triangle_intersect(self,t,includeboundary=True,errtol=1e-12):
        #This method returns whether an inputted triangle intersects
        #the original triangle object or not.
        #Distinction is made whether to include the boundaries or not.
        return (any([t.edge_triangle_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in self.edges])
             or any([self.edge_triangle_intersect(e,includeboundary=includeboundary,errtol=errtol) for e in t.edges]))
    
    def circumsphere(self):
        #This method computes the circumsphere of the triangle object and
        #makes the circumsphere variable a singleton. Obviously, for 3 points in 3D,
        #there are an infinite number of spheres that will count as circumspheres,
        #but this method focuses on the one with the circumcenter in the plane
        #uniquely defined by the triangle. 
        if self.circumSphere is None:
            #Here the method calculates the circumcenter of the triangle's 3 points.
            e2Xe1=self.edges[2].cross(self.edges[1])
            e2Xe1Xe2=e2Xe1.cross(self.edges[2])
            e1Xe2Xe1=self.edges[1].cross(e2Xe1)
            center=(e2Xe1Xe2.points[0]-e2Xe1Xe2.points[1]).scalar_mul(self.edges[1].length_squared())
            center+=(e1Xe2Xe1.points[0]-e1Xe2Xe1.points[1]).scalar_mul(self.edges[2].length_squared())
            center=center.scalar_div(2.0*e2Xe1.length_squared())
            center=center+self.points[0]
            #Here the method finds the radius of the sphere by calculating the
            #distance between any of the triangle's 3 points and the new circumcenter,
            #then returns the data as a sphere object.
            self.circumSphere=sphere(center,center.distance(self.points[0]))
        return self.circumSphere
    
    def inCircumsphere(self,p,includeboundary=True,errtol=1e-12):
        #This method calculates whether an inputted point lies inside of the
        #triangle object's circumcircle. Obviously, for 3 points in 3D,
        #there are an infinite number of spheres that will count as circumspheres,
        #but this method focuses on the one with the circumcenter in the plane
        #uniquely defined by the triangle.
        if self.circumSphere is None:
            e2Xe1=self.edges[2].cross(self.edges[1])
            e2Xe1Xe2=e2Xe1.cross(self.edges[2])
            e1Xe2Xe1=self.edges[1].cross(e2Xe1)
            center=(e2Xe1Xe2.points[0]-e2Xe1Xe2.points[1]).scalar_mul(self.edges[1].length_squared())
            center+=(e1Xe2Xe1.points[0]-e1Xe2Xe1.points[1]).scalar_mul(self.edges[2].length_squared())
            center=center.scalar_div(2.0*e2Xe1.length_squared())
            center=center+self.points[0]
            #The method now checks if the point is farther away from the circumcenter
            #than one of the vertices of the triangle object
            c_p_edge=edge(center,p)
            c_v_edge=edge(center,self.points[0])
            if includeboundary:
                return c_p_edge.length_squared()<=c_v_edge.length_squared()+abs(errtol)
            else:
                return c_p_edge.length_squared()<c_v_edge.length_squared()-abs(errtol)
        else:
            if includeboundary:
                return edge(self.circumsphere().center,p).is_shorter_than(self.circumsphere().radius+abs(errtol),or_equal_to=True)
            else:
                return edge(self.circumsphere().center,p).is_shorter_than(self.circumsphere().radius-abs(errtol),or_equal_to=False)

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
        if self.triagArea is not None:
            t.triagArea=self.triagArea
        if self.interiorAngles is not None:
            t.interiorAngles=self.interiorAngles
        if self.circumSphere is not None:    
            t.circumSphere=self.circumSphere.copy() 
        return t
    
    def draw(self,plotaxis,points=True,edges=True,fill=True,flat=True,elev=10,azim=-75,color="black",alpha=1):    
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
            if flat:
                plotaxis.add_patch(plt.Polygon([[self.points[i].x,self.points[i].y] for i in [0,1,2]],
                                               facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
            else:
                plotaxis.add_collection3d(Poly3DCollection([[[self.points[i].x,self.points[i].y,self.points[i].z] for i in [0,1,2]]],
                                                                 facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
        else:
            if edges:
                if flat:
                    plotaxis.plot([self.points[i].x for i in [0,1,2,0]],
                                  [self.points[i].y for i in [0,1,2,0]],
                                  color=color,alpha=alpha,zorder=0)
                else:
                    plotaxis.plot3D([self.points[i].x for i in [0,1,2,0]],
                                    [self.points[i].y for i in [0,1,2,0]],
                                    [self.points[i].z for i in [0,1,2,0]],
                                    color=color,alpha=alpha,zorder=0)

        if points:
            if fill:
                if flat:
                    plotaxis.scatter([p.x for p in self.points],
                                     [p.y for p in self.points],
                                     facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
                else:
                    plotaxis.scatter3D([p.x for p in self.points],
                                       [p.y for p in self.points],
                                       [p.z for p in self.points],
                                       facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
            else:
                if flat:
                    plotaxis.scatter([p.x for p in self.points],
                                     [p.y for p in self.points],
                                     facecolor=color,edgecolor=color,alpha=alpha,zorder=1)
                else:
                    plotaxis.scatter3D([p.x for p in self.points],
                                       [p.y for p in self.points],
                                       [p.z for p in self.points],
                                       facecolor=color,edgecolor=color,alpha=alpha,zorder=1)
        if not flat:
            plotaxis.view_init(elev=elev,azim=azim)
