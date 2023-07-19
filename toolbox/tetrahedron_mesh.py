import numpy as np
from toolbox.point import point
from toolbox.edge import edge
from toolbox.triangle import triangle
from toolbox.tetrahedron import tetrahedron
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as pltcolors

graphsize=9
font = {"family": "serif",
    "color": "black",
    "weight": "bold",
    "size": "20"}

class tetrahedron_mesh(object):
    def __init__(self,X,Y,Z):
        self.P=[]
        self.E=[]
        self.T=[]
        self.S=[]
        self.triangulate(X,Y,Z)
        
    def triangulate(self,X,Y,Z):    
        #This method performs the Bowyer-Watson tetrahedron mesh generation
        #from a given set of points P. It first sets up a beginner tetrahedron
        #that encompasses all points, and then iteratively adds all points one
        #by one, retriangulating locally as it goes.
        
        #Set up of initial tetrahedron that encompasses all the points
        maxX,minX=max(X),min(X)
        maxY,minY=max(Y),min(Y)
        maxZ,minZ=max(Z),min(Z)
        cx,cy,cz=(maxX+minX)/2,(maxY+minY)/2,(maxZ+minZ)/2
        
        r=1.1*np.sqrt((maxX-minX)**2+(maxY-minY)**2+(maxZ-minZ)**2)/2
        sqrt6r=np.sqrt(6)*r
        sqrt2r=np.sqrt(2)*r
        init_p1=point(cx+sqrt6r,cy-sqrt2r,cz-r)
        init_p1.index=0
        init_p2=point(cx-sqrt6r,cy-sqrt2r,cz-r)
        init_p2.index=1
        init_p3=point(cx,cy+2*sqrt2r,cz-r)
        init_p3.index=2
        init_p4=point(cx,cy,cz+3*r)
        init_p4.index=3
        self.P=[init_p1,init_p2,init_p3,init_p4]
        
        init_e1=edge(init_p1,init_p2)
        init_e1.index=0
        init_e1.update_network()
        init_e2=edge(init_p1,init_p3)
        init_e2.index=1
        init_e2.update_network()
        init_e3=edge(init_p2,init_p3)
        init_e3.index=2
        init_e3.update_network()
        init_e4=edge(init_p1,init_p4)
        init_e4.index=3
        init_e4.update_network()
        init_e5=edge(init_p2,init_p4)
        init_e5.index=4
        init_e5.update_network()
        init_e6=edge(init_p3,init_p4)
        init_e6.index=5
        init_e6.update_network()
        self.E=[init_e1,init_e2,init_e3,init_e4,init_e5,init_e6]
        
        init_t1=triangle(init_p1,init_p2,init_p3,
                         init_e1,init_e2,init_e3)
        init_t1.index=0
        init_t1.update_network()
        init_t2=triangle(init_p1,init_p2,init_p4,
                         init_e1,init_e4,init_e5)
        init_t2.index=1
        init_t2.update_network()
        init_t3=triangle(init_p1,init_p3,init_p4,
                         init_e2,init_e4,init_e6)
        init_t3.index=2
        init_t3.update_network()
        init_t4=triangle(init_p2,init_p3,init_p4,
                         init_e3,init_e5,init_e6)
        init_t4.index=3
        init_t4.update_network()
        self.T=[init_t1,init_t2,init_t3,init_t4]
        
        init_s=tetrahedron(init_p1,init_p2,init_p3,init_p4,
                             init_e1,init_e2,init_e3,init_e4,init_e5,init_e6,
                             init_t1,init_t2,init_t3,init_t4)
        init_s.index=0
        init_s.update_network()
        self.S=[init_s]
        
        #Iteratively calls upon the insert_Vertex_Delaunay method
        for i in range(min(len(X),len(Y),len(Z))):
            self.insert_Vertex_Delaunay(point(X[i],Y[i],Z[i]))
        
        #Removes the initial tetrahedron and performs any additional clean up
        self.delete_point(init_p1)
        self.delete_point(init_p2)
        self.delete_point(init_p3)
        self.delete_point(init_p4)
        
        for t in self.T:
            if len(t.tetrahedra)<2:
                t.enclosed=False
            
    def tetrahedron_search(self,p,start_tetrahedron=None,errtol=1e-6):
        #This method searches for the tetrahedron that emcompasses a given point.
        #This method assumes that this point is in at least one tetrahedron of our
        #tetrahedral mesh.
        errtol=abs(errtol)
        if len(self.S)>0:
            #The method starts with a randomly chosen tetrahedron in the mesh... 
            if start_tetrahedron is None:
                current_s=self.S[np.random.randint(len(self.S))]
            else:
                try:
                    current_s=start_tetrahedron
                except Exception:
                    current_s=self.S[np.random.randint(len(self.S))]
            prev_s=self.S[current_s.index]
            #...and checks if that tetrahedron encompasses the given point.
            found_bool=current_s.point_tetrahedron_intersect(p)
            counter=0
            while not found_bool and counter<len(self.S):
                #If the current tetrahedron does not encompass the given point,
                #it searches through all the neighbors of the current tetrahedron...
                next_s=None
                for i in range(4):
                    if edge(current_s.average(),p).point_edge_intersect(current_s.points[i]):
                        if current_s.neighbors[(i+1)%4] is not None and not current_s.neighbors[(i+1)%4].is_tetrahedron(prev_s):
                            if current_s.neighbors[(i+2)%4] is not None and not current_s.neighbors[(i+2)%4].is_tetrahedron(prev_s):
                                if current_s.neighbors[(i+3)%4] is not None and not current_s.neighbors[(i+3)%4].is_tetrahedron(prev_s):
                                    rand=np.random.rand()
                                    if rand<0.333:
                                        next_s=current_s.neighbors[(i+1)%4]
                                    elif rand<0.667:
                                        next_s=current_s.neighbors[(i+2)%4]
                                    else:
                                        next_s=current_s.neighbors[(i+3)%4]
                                else:
                                    if np.random.rand()<0.5:
                                        next_s=current_s.neighbors[(i+1)%4]
                                    else:
                                        next_s=current_s.neighbors[(i+2)%4]
                            else:
                                if current_s.neighbors[(i+3)%4] is not None and not current_s.neighbors[(i+3)%4].is_tetrahedron(prev_s):
                                    if np.random.rand()<0.5:
                                        next_s=current_s.neighbors[(i+1)%4]
                                    else:
                                        next_s=current_s.neighbors[(i+3)%4]
                                else:
                                    next_s=current_s.neighbors[(i+1)%4]
                        else:
                            if current_s.neighbors[(i+2)%4] is not None and not current_s.neighbors[(i+2)%4].is_tetrahedron(prev_s):
                                if current_s.neighbors[(i+3)%4] is not None and not current_s.neighbors[(i+3)%4].is_tetrahedron(prev_s):
                                    if np.random.rand()<0.5:
                                        next_s=current_s.neighbors[(i+2)%4]
                                    else:
                                        next_s=current_s.neighbors[(i+3)%4]
                                else:
                                    next_s=current_s.neighbors[(i+2)%4]
                            else:
                                if current_s.neighbors[(i+3)%4] is not None and not current_s.neighbors[(i+3)%4].is_tetrahedron(prev_s):
                                    next_s=current_s.neighbors[(i+3)%4]
                if next_s is None:
                    for i in range(4):
                        if current_s.neighbors[i] is not None:
                            if current_s.triangles[i].edge_triangle_intersect(edge(current_s.average(),p)):
                                if not current_s.neighbors[i].is_tetrahedron(prev_s):
                                    next_s=current_s.neighbors[i]
                                    break
                #The method then moves to the best neighboring tetrahedron and
                #checks again if that tetrahedron encompasses the given point,
                #repeating the cycle.
                if next_s is None:
                    temp=prev_s
                    prev_s=current_s
                    current_s=temp
                else:
                    prev_s=current_s
                    current_s=next_s
                found_bool=current_s.point_tetrahedron_intersect(p)
                counter+=1
            if found_bool:
                return current_s
        return None

    def insert_Vertex_Delaunay(self,p,return_changed_tetrahedra=False):
        #This method inserts a point into a tetrahedral mesh which is assumed
        #to fulfil the Delaunay requisite already. The point is then added
        #using the Bowyer-Watson Algorithm, which then results in a local
        #retriangulation of the mesh
        if all([not pointVar.is_point(p) for pointVar in self.P]):
            p.index=len(self.P)
            self.P.append(p)
            
            #First, the method first looks for the tetrahedron that contains
            #the point that is about to be inserted using an efficient
            #(hopefully O(log(N)) runtime) search algorithm. It then sets up
            #a beginning hole, surrounded by a front of appropriate tetrahedra
            tetrahedron_zero=self.tetrahedron_search(p)
            poly_triangles=[]
            constraint=None
            for t in tetrahedron_zero.triangles:
                if not t.point_triangle_intersect(p):
                    poly_triangles.append(t)
                else:
                    if t.constraint:
                        constraint=t
                    if len(t.tetrahedra)>1:
                        poly_triangles.append(t)
            bad_tetrahedra=[tetrahedron_zero]
            self.delete_tetrahedron(tetrahedron_zero)
            for i in range(len(tetrahedron_zero.edges)-1,-1,-1):
                if len(tetrahedron_zero.edges[i].triangles)==0:
                    self.delete_edge(tetrahedron_zero.edges[i])
            search_bool=True
            
            #Here the method performs a breadth-first search that expands
            #the polyhedral hole by... 
            while search_bool:
                search_bool=False
                #...investigating any triangle along the surface of
                #the polyhedral hole...
                for i in range(len(poly_triangles)-1,-1,-1):
                    #...and seeing if the triangle of interest is worth investigating.
                    constraint_bool=poly_triangles[i].constraint
                    constraint_intersect_bool=(constraint_bool
                                               and poly_triangles[i].point_triangle_intersect(p))
                    if not constraint_bool or constraint_intersect_bool:    
                        #The method then analyzes the (theoretically) only tetrahedron
                        #attached to the triangle and sees if it fails the Delaunay condition...
                        if (len(poly_triangles[i].tetrahedra)>0
                            and poly_triangles[i].tetrahedra[0].inCircumsphere(p)):
                            delete_poly_t_indices=[]
                            #...and if so, expands the polyhedral hole by adding
                            #the new triangles of the tetrahedron...
                            for t in poly_triangles[i].tetrahedra[0].triangles:
                                match_bool=False
                                for j in range(len(poly_triangles)):
                                    if t.is_triangle(poly_triangles[j]):
                                        match_bool=True
                                        delete_poly_t_indices.append(j)
                                if not match_bool:
                                    poly_triangles.append(t)
                            #...while keeping track of any constraints that
                            #the to-be-inserted points intersects.
                            if constraint_intersect_bool:
                                constraint=poly_triangles[i]    
                            #Just some book-keeping...
                            bad_tetrahedra.append(poly_triangles[i].tetrahedra[0])
                            tetrahedron_edges=[e for e in poly_triangles[i].tetrahedra[0].edges]
                            self.delete_triangle(poly_triangles[i])
                            for j in range(len(tetrahedron_edges)-1,-1,-1):
                                if len(tetrahedron_edges[j].triangles)==0:
                                    self.delete_edge(tetrahedron_edges[j])
                            search_bool=True
                            #And finally the method deletes any of the old
                            #infrastructure of the polyhedral hole
                            for index in sorted(delete_poly_t_indices,reverse=True):
                                del poly_triangles[index]
                            if len(delete_poly_t_indices)>1:
                                break
            
            #The polyhedral hole is retriangulated, using the triangles of the
            #polyhedral hole as a backbone. Care is taken to not include any
            #edges twice. All new edges, triangles, and tetrahedra are then
            #also networked into the mesh to conserve the global data structure.
            star_edges=[]
            star_triangles=[]
            good_tetrahedra=[]
            for poly_t in poly_triangles:
                e1=edge(poly_t.points[0],p)
                e2=edge(poly_t.points[1],p)
                e3=edge(poly_t.points[2],p)
                match_e1_bool=False
                match_e2_bool=False
                match_e3_bool=False
                for star_e in star_edges:
                    if not match_e1_bool and e1.is_edge(star_e):
                        e1=star_e
                        match_e1_bool=True
                    elif not match_e2_bool and e2.is_edge(star_e):
                        e2=star_e
                        match_e2_bool=True
                    elif not match_e3_bool and e3.is_edge(star_e):
                        e3=star_e
                        match_e3_bool=True
                    if match_e1_bool and match_e2_bool and match_e3_bool:
                        break
                if not match_e1_bool:
                    e1.index=len(self.E)
                    e1.update_network()
                    self.E.append(e1)
                    star_edges.append(e1)
                if not match_e2_bool:
                    e2.index=len(self.E)
                    e2.update_network()
                    self.E.append(e2)
                    star_edges.append(e2)
                if not match_e3_bool:
                    e3.index=len(self.E)
                    e3.update_network()
                    self.E.append(e3)
                    star_edges.append(e3)
                
                t1=triangle(poly_t.points[0],poly_t.points[1],p,
                            poly_t.edges[2],e1,e2)
                t2=triangle(poly_t.points[0],poly_t.points[2],p,
                            poly_t.edges[1],e1,e3)
                t3=triangle(poly_t.points[1],poly_t.points[2],p,
                            poly_t.edges[0],e2,e3)
                match_t1_bool=False
                match_t2_bool=False
                match_t3_bool=False
                for star_t in star_triangles:
                    if not match_t1_bool and t1.is_triangle(star_t):
                        t1=star_t
                        match_t1_bool=True
                    if not match_t2_bool and t2.is_triangle(star_t):
                        t2=star_t
                        match_t2_bool=True
                    if not match_t3_bool and t3.is_triangle(star_t):
                        t3=star_t
                        match_t3_bool=True
                if not match_t1_bool:    
                    t1.index=len(self.T)
                    t1.update_network()
                    self.T.append(t1)
                    star_triangles.append(t1)
                    #A quick check to transfer any constraint attributes
                    #to any newly constructed triangles
                    if constraint is not None:
                        if (constraint.point_triangle_intersect(poly_t.points[0])
                            and constraint.point_triangle_intersect(poly_t.points[1])):
                            t1.constraint=True
                if not match_t2_bool:    
                    t2.index=len(self.T)
                    t2.update_network()
                    self.T.append(t2)
                    star_triangles.append(t2)
                    #A quick check to transfer any constraint attributes
                    #to any newly constructed triangles
                    if constraint is not None:
                        if (constraint.point_triangle_intersect(poly_t.points[0])
                            and constraint.point_triangle_intersect(poly_t.points[2])):
                            t2.constraint=True
                if not match_t3_bool:    
                    t3.index=len(self.T)
                    t3.update_network()
                    self.T.append(t3)
                    star_triangles.append(t3)
                    #A quick check to transfer any constraint attributes
                    #to any newly constructed triangles
                    if constraint is not None:
                        if (constraint.point_triangle_intersect(poly_t.points[1])
                            and constraint.point_triangle_intersect(poly_t.points[2])):
                            t3.constraint=True

                s1=tetrahedron(poly_t.points[0],poly_t.points[1],poly_t.points[2],p,
                              poly_t.edges[0],poly_t.edges[1],poly_t.edges[2],e1,e2,e3,
                              poly_t,t1,t2,t3)
                s1.index=len(self.S)
                s1.update_network()
                self.S.append(s1)
                good_tetrahedra.append(s1)
            
            for good_s in good_tetrahedra:
                for good_t in good_s.triangles:
                    good_t.enclosed=(len(good_t.tetrahedra)>1)
            
            if return_changed_tetrahedra:
                return good_tetrahedra,bad_tetrahedra
        elif return_changed_tetrahedra:
            return [],[]

    def delete_point(self,p):
        #This method deletes a given point and in doing so deletes any edges,
        #triangles, and tetrahedra that are attached. Care is taken to also
        #update the global data structure.
        
        #Deleting the attached tetrahedra
        for i in range(len(p.tetrahedra)-1,-1,-1):
            self.delete_tetrahedron(p.tetrahedra[i])

        #Deleting the attached triangles...
        for i in range(len(p.triangles)-1,-1,-1):
            self.delete_triangle(p.triangles[i])

        #Deleting the attached edges...
        for i in range(len(p.edges)-1,-1,-1):
            self.delete_edge(p.edges[i])

        #Dealing with the point itself...
        for n in p.neighbors:
            for i in range(len(n.neighbors)-1,-1,-1):
                if n.neighbors[i].is_point(p):
                    del n.neighbors[i]
        for i in range(p.index+1,len(self.P)):
            self.P[i].index-=1
        del self.P[p.index]
        #self.P=self.P[:p.index]+self.P[p.index+1:]
        #del p

    def delete_edge(self,e):
        #This method deletes a given edge and in doing so deletes any triangles
        #tetrahedra that are attached. Care is taken to also update the global
        #data structure.
        
        #Deleting the attached tetrahedra
        for i in range(len(e.tetrahedra)-1,-1,-1):
            self.delete_tetrahedron(e.tetrahedra[i])

        #Deleting the attached triangles...
        for i in range(len(e.triangles)-1,-1,-1):
            self.delete_triangle(e.triangles[i])

        #Dealing with the points...
        for p in e.points:
            for i in range(len(p.edges)-1,-1,-1):
                if p.edges[i].is_edge(e):
                    del p.edges[i]
        for i in range(len(e.points[0].neighbors)-1,-1,-1):
            if e.points[0].neighbors[i].is_point(e.points[1]):
                del e.points[0].neighbors[i]
        for i in range(len(e.points[1].neighbors)-1,-1,-1):
            if e.points[1].neighbors[i].is_point(e.points[0]):
                del e.points[1].neighbors[i]
        
        #Dealing with the edge itself...
        for i in range(e.index+1,len(self.E)):
            self.E[i].index-=1
        del self.E[e.index]
        #self.E=self.E[:e.index]+self.E[e.index+1:]
        #del e
        
    def delete_triangle(self,t):
        #This method deletes a given triangle. Care is taken to also update
        #the global data structure.
        
        #Deleting the attached tetrahedra
        for i in range(len(t.tetrahedra)-1,-1,-1):
            self.delete_tetrahedron(t.tetrahedra[i])
        
        #Dealing with the points...
        for p in t.points:
            for i in range(len(p.triangles)-1,-1,-1):
                if p.triangles[i].is_triangle(t):
                    del p.triangles[i]
        
        #Dealing with the edges...
        for e in t.edges:
            for i in range(len(e.triangles)-1,-1,-1):
                if e.triangles[i].is_triangle(t):
                    del e.triangles[i]
        
        #Dealing with the triangle itself...
        for n in t.neighbors:
            if n is not None:
                for i in range(len(n.neighbors)-1,-1,-1):
                    if n.neighbors[i] is not None:
                        if n.neighbors[i].is_triangle(t):
                            n.neighbors[i]=None
        for i in range(t.index+1,len(self.T)):
            self.T[i].index-=1
        del self.T[t.index]
        #self.T=self.T[:t.index]+self.T[t.index+1:]
        #del t
        
    def delete_tetrahedron(self,s):
        #This method deletes a given tetrahedron. Care is taken to also update
        #the global data structure.
        
        #Dealing with the points...
        for p in s.points:
            for i in range(len(p.tetrahedra)-1,-1,-1):
                if p.tetrahedra[i].is_tetrahedron(s):
                    del p.tetrahedra[i]
        
        #Dealing with the edges...
        for e in s.edges:
            for i in range(len(e.tetrahedra)-1,-1,-1):
                if e.tetrahedra[i].is_tetrahedron(s):
                    del e.tetrahedra[i]
        
        #Dealing with the triangles...
        for t in s.triangles:
            for i in range(len(t.tetrahedra)-1,-1,-1):
                if t.tetrahedra[i].is_tetrahedron(s):
                    del t.tetrahedra[i]
        
        #Dealing with the tetrahedron itself...
        for n in s.neighbors:
            if n is not None:
                for i in range(len(n.neighbors)-1,-1,-1):
                    if n.neighbors[i] is not None:
                        if n.neighbors[i].is_tetrahedron(s):
                            n.neighbors[i]=None
        for i in range(s.index+1,len(self.S)):
            self.S[i].index-=1
        del self.S[s.index]
        #self.S=self.s[:s.index]+self.S[s.index+1:]
        #del s

    def __repr__(self):
        #This method returns a string representation of the tetrahedron mesh.
        string_rep="Number of Tetrahedra: "+str(len(self.S))+"\n"
        for s in self.S:
            string_rep+=str(s)+"\n"
        string_rep+="\nNumber of Triangles: "+str(len(self.T))+"\n"
        for t in self.T:
            string_rep+=str(t)+"\n"
        string_rep+="\nNumber of Edges: "+str(len(self.E))+"\n"
        for e in self.E:
            string_rep+=str(e)+"\n"
        string_rep+="\nNumber of Points: "+str(len(self.P))+"\n"
        for p in self.P:
            string_rep+=str(p)+"\n"
        return string_rep
        
    def draw(self,plotaxis=None,show_tetrahedra=True,show_edges=True,show_points=True,expand=False,elev=10,azim=-75,color="black",alpha=1):    
        #This method plots the triangle mesh object into an inputted
        #figure axis object.
        
        if expand and len(self.S)>1:
            center=[np.average([p.x for p in self.P]),
                    np.average([p.y for p in self.P]),
                    np.average([p.z for p in self.P])]
            expansion_scale=0.05*len(self.P)+1.75
            
        #This starts off the method and constructs a matplotlib figure and axis
        #that the triangulation will be drawn on if none was provided.
        plotshowbool=False
        if plotaxis is None:
            plotaxis=plt.figure(figsize=(graphsize,graphsize)).add_subplot(111,projection="3d")
            plotaxis.xaxis.set_rotate_label(False)
            plotaxis.yaxis.set_rotate_label(False)
            plotaxis.zaxis.set_rotate_label(False)
            plotaxis.set_title("Triangulation",fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
            plotaxis.set_zlabel("$\\mathbf{Z}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotaxis.zaxis.set_tick_params(labelsize=16)
            start=[min([p.x for p in self.P]),min([p.y for p in self.P]),min([p.z for p in self.P])]
            stop=[max([p.x for p in self.P]),max([p.y for p in self.P]),max([p.z for p in self.P])]
            diff=0.1*np.array([stop[0]-start[0],stop[1]-start[1],stop[2]-start[2]])
            if expand:
                plotaxis.set_xlim([start[0]+expansion_scale*(start[0]-center[0])-diff[0],
                                   stop[0]+expansion_scale*(stop[0]-center[0])+diff[0]])
                plotaxis.set_ylim([start[1]+expansion_scale*(start[1]-center[1])-diff[1],
                                   stop[1]+expansion_scale*(stop[1]-center[1])+diff[1]])
                plotaxis.set_zlim([start[2]+expansion_scale*(start[2]-center[2])-diff[2],
                                   stop[2]+expansion_scale*(stop[2]-center[2])+diff[2]])
            else:
                plotaxis.set_xlim([start[0]-diff[0],stop[0]+diff[0]])
                plotaxis.set_ylim([start[1]-diff[1],stop[1]+diff[1]])
                plotaxis.set_zlim([start[2]-diff[2],stop[2]+diff[2]])
            plotshowbool=True
        else:
            xlim=plotaxis.get_xlim()
            ylim=plotaxis.get_ylim()
            zlim=plotaxis.get_zlim()
            start=[min([p.x for p in self.P]),min([p.y for p in self.P]),min([p.z for p in self.P])]
            stop=[max([p.x for p in self.P]),max([p.y for p in self.P]),max([p.z for p in self.P])]
            diff=0.1*np.array([stop[0]-start[0],stop[1]-start[1],stop[2]-start[2]])
            if expand:
                plotaxis.set_xlim([min(xlim[0],start[0]+expansion_scale*(start[0]-center[0])-diff[0]),
                                   max(xlim[1],stop[0]+expansion_scale*(stop[0]-center[0])+diff[0])])
                plotaxis.set_ylim([min(ylim[0],start[1]+expansion_scale*(start[1]-center[1])-diff[1]),
                                   max(ylim[1],stop[1]+expansion_scale*(stop[1]-center[1])+diff[1])])
                plotaxis.set_zlim([min(zlim[0],start[2]+expansion_scale*(start[2]-center[2])-diff[2]),
                                   max(zlim[1],stop[2]+expansion_scale*(stop[2]-center[2])+diff[2])])
            else:
                plotaxis.set_xlim([min(xlim[0],start[0]-diff[0]),max(xlim[1],stop[0]+diff[0])])
                plotaxis.set_ylim([min(ylim[0],start[1]-diff[1]),max(ylim[1],stop[1]+diff[0])])
                plotaxis.set_zlim([min(zlim[0],start[2]-diff[2]),max(zlim[1],stop[2]+diff[0])])
        plotaxis.view_init(elev=elev,azim=azim)
        
        if sum(pltcolors.to_rgb(color))<=1.0:
            color_alt="white"
        else:
            color_alt="black"
            
        if expand and len(self.S)>1:
            if show_tetrahedra:
                face_color=color
                if show_edges:
                    edge_color=color_alt
                else:
                    edge_color=color
                for s in self.S:
                    plotaxis.add_collection3d(Poly3DCollection([[[p.x+expansion_scale*(s.average().x-center[0]),p.y+expansion_scale*(s.average().y-center[1]),p.z+expansion_scale*(s.average().z-center[2])] for p in s.triangles[0].points],
                                                                [[p.x+expansion_scale*(s.average().x-center[0]),p.y+expansion_scale*(s.average().y-center[1]),p.z+expansion_scale*(s.average().z-center[2])] for p in s.triangles[1].points],
                                                                [[p.x+expansion_scale*(s.average().x-center[0]),p.y+expansion_scale*(s.average().y-center[1]),p.z+expansion_scale*(s.average().z-center[2])] for p in s.triangles[2].points],
                                                                [[p.x+expansion_scale*(s.average().x-center[0]),p.y+expansion_scale*(s.average().y-center[1]),p.z+expansion_scale*(s.average().z-center[2])] for p in s.triangles[3].points]],
                                                                facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
            elif show_edges:
                for e in self.E:
                    if e.constraint:
                        linewidth=4
                    else:
                        linewidth=1
                    plotaxis.plot3D([p.x+expansion_scale*(e.average().x-center[0]) for p in e.points],
                                    [p.y+expansion_scale*(e.average().y-center[1]) for p in e.points],
                                    [p.z+expansion_scale*(e.average().z-center[2]) for p in e.points],
                                    linewidth=linewidth,color=color,alpha=alpha,zorder=0)
    
            if show_points:
                facecolor=color
                if show_tetrahedra:
                    edgecolor=color_alt
                else:
                    edgecolor=color
                X,Y,Z=[],[],[]
                if show_tetrahedra:
                    for s in self.S:
                        X=X+[p.x+expansion_scale*(s.average().x-center[0]) for p in s.points]
                        Y=Y+[p.y+expansion_scale*(s.average().y-center[1]) for p in s.points]
                        Z=Z+[p.z+expansion_scale*(s.average().z-center[2]) for p in s.points]
                else:
                    X=[p.x+expansion_scale*(p.x-center[0]) for p in self.P]
                    Y=[p.y+expansion_scale*(p.y-center[1]) for p in self.P]
                    Z=[p.z+expansion_scale*(p.z-center[2]) for p in self.P]
                plotaxis.scatter3D(X,Y,Z,
                                   facecolor=facecolor,edgecolor=edgecolor,alpha=alpha,zorder=1)
        else:
            if show_tetrahedra:
                face_color=color
                if show_edges:
                    edge_color=color_alt
                else:
                    edge_color=color
                #This is causing things to be weird because you are plotting the
                #TRIANGLES, not the tetrahedra
                plotaxis.add_collection3d(Poly3DCollection([[[p.x,p.y,p.z] for p in t.points] for t in self.T],
                                                            facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
            elif show_edges:
                for e in self.E:
                    if e.constraint:
                        linewidth=4
                    else:
                        linewidth=1
                    plotaxis.plot3D([p.x for p in e.points],
                                    [p.y for p in e.points],
                                    [p.z for p in e.points],
                                    linewidth=linewidth,color=color,alpha=alpha,zorder=0)
    
            if show_points:
                facecolor=color
                if show_tetrahedra:
                    edgecolor=color_alt
                else:
                    edgecolor=color
                plotaxis.scatter3D([p.x for p in self.P],
                                   [p.y for p in self.P],
                                   [p.z for p in self.P],
                                   facecolor=facecolor,edgecolor=edgecolor,alpha=alpha,zorder=1)
        if plotshowbool:
            plt.show()
