import numpy as np
from toolbox.point import point
from toolbox.edge import edge
from toolbox.triangle import triangle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as pltcolors

graphsize=9
font = {"family": "serif",
    "color": "black",
    "weight": "bold",
    "size": "20"}

class triangle_mesh(object):
    def __init__(self,X,Y):
        self.P=[]
        self.E=[]
        self.T=[]
        self.triangulate(X,Y)
    
    def triangulate(self,X,Y):    
        #This method performs the Bowyer-Watson triangle mesh generation
        #from a given set of points P. It first sets up a beginner triangle
        #that encompasses all points, and then iteratively adds all points one
        #by one, retriangulating locally as it goes.
        
        #Set up of initial triangle that encompasses all the points
        maxX,minX=max(X),min(X)
        maxY,minY=max(Y),min(Y)
        cx,cy=(maxX+minX)/2,(maxY+minY)/2
        r=1.1*np.sqrt((maxX-minX)**2+(maxY-minY)**2)/2
        sqrt3r=np.sqrt(3)*r
        init_p1=point(cx,cy+2*r,0)
        init_p1.index=0
        init_p2=point(cx+sqrt3r,cy-r,0)
        init_p2.index=1
        init_p3=point(cx-sqrt3r,cy-r,0)
        init_p3.index=2
        self.P=[init_p1,init_p2,init_p3]
        
        init_e1=edge(init_p1,init_p2)
        init_e1.index=0
        init_e1.update_network()
        init_e2=edge(init_p1,init_p3)
        init_e2.index=1
        init_e2.update_network()
        init_e3=edge(init_p2,init_p3)
        init_e3.index=2
        init_e3.update_network()
        self.E=[init_e1,init_e2,init_e3]
        
        init_t=triangle(init_p1,init_p2,init_p3,
                        init_e1,init_e2,init_e3)
        init_t.index=0
        init_t.update_network()
        self.T=[init_t]
        
        #Iteratively calls upon the insert_Vertex_Delaunay method
        for i in range(min(len(X),len(Y))):
            self.insert_Vertex_Delaunay(point(X[i],Y[i],0))
        
        #Removes the initial triangle and performs any additional clean up
        self.delete_point(init_p1)
        self.delete_point(init_p2)
        self.delete_point(init_p3)
        
        for e in self.E:
            if len(e.triangles)<2:
                e.enclosed=False

    def triangle_search(self,p,start_triangle=None,errtol=1e-6):
        #This method searches for the triangle that emcompasses a given point.
        #This method assumes that this point is in at least one triangle of our
        #triangular mesh.
        errtol=abs(errtol)
        if len(self.T)>0:
            #The method starts with a randomly chosen triangle in the mesh... 
            if start_triangle is None:
                current_t=self.T[np.random.randint(len(self.T))]
            else:
                try:
                    current_t=start_triangle
                except Exception:
                    current_t=self.T[np.random.randint(len(self.T))]
            prev_t=self.T[current_t.index]
            #...and checks if that triangle encompasses the given point.
            found_bool=current_t.point_triangle_intersect(p)
            counter=0
            while not found_bool and counter<len(self.T):
                #If the current triangle does not encompass the given point,
                #it searches through all the neighbors of the current triangle...
                next_t=None
                for i in range(3):
                    if edge(current_t.average(),p).point_edge_intersect(current_t.points[i]):
                        if current_t.neighbors[(i+1)%3] is not None and not current_t.neighbors[(i+1)%3].is_triangle(prev_t):
                            if current_t.neighbors[(i+2)%3] is not None and not current_t.neighbors[(i+2)%3].is_triangle(prev_t):
                                if np.random.rand()<0.5:
                                    next_t=current_t.neighbors[(i+1)%3]
                                else:
                                    next_t=current_t.neighbors[(i+2)%3]
                            else:
                                next_t=current_t.neighbors[(i+1)%3]
                        else:
                            if current_t.neighbors[(i+2)%3] is not None and not current_t.neighbors[(i+2)%3].is_triangle(prev_t):
                                next_t=current_t.neighbors[(i+2)%3]
                if next_t is None:
                    for i in range(3):
                        if current_t.neighbors[i] is not None:
                            if current_t.edges[i].edge_edge_intersect(edge(current_t.average(),p)):
                                if not current_t.neighbors[i].is_triangle(prev_t):
                                    next_t=current_t.neighbors[i]
                                    break
                #The method then moves to the best neighboring triangle and
                #checks again if that triangle encompasses the given point,
                #repeating the cycle.
                if next_t is None:
                    temp=prev_t
                    prev_t=current_t
                    current_t=temp
                else:
                    prev_t=current_t
                    current_t=next_t
                found_bool=current_t.point_triangle_intersect(p)
                counter+=1
            if found_bool:
                return current_t
        return None

    def insert_Vertex_Delaunay(self,p,return_changed_triangles=False):
        #This method inserts a point into a triangular mesh which is assumed
        #to fulfil the Delaunay requisite already. The point is then added
        #using the Bowyer-Watson Algorithm, which then results in a local
        #retriangulation of the mesh
        if all([not pointVar.is_point(p) for pointVar in self.P]):
            p.index=len(self.P)
            self.P.append(p)
            
            #First, the method first looks for the triangle that contains
            #the point that is about to be inserted using an efficient
            #(hopefully O(log(N)) runtime) search algorithm. It then sets up
            #a beginning hole, surrounded by a front of appropriate triangles
            triangle_zero=self.triangle_search(p)
            poly_edges=[]
            constraint=None
            for e in triangle_zero.edges:
                if not e.point_edge_intersect(p):
                    poly_edges.append(e)
                else:
                    if e.constraint:
                        constraint=e
                    if len(e.triangles)>1:
                        poly_edges.append(e)
            bad_triangles=[triangle_zero]
            self.delete_triangle(triangle_zero)
            search_bool=True
            
            #Here the method performs a breadth-first search that expands
            #the polygonal hole by... 
            while search_bool:
                search_bool=False
                #...investigating any triangles along the surface of
                #the polygonal hole...
                for i in range(len(poly_edges)-1,-1,-1):
                    #...and seeing if the edge of interest is worth investigating.
                    constraint_bool=poly_edges[i].constraint
                    constraint_intersect_bool=(constraint_bool
                                               and poly_edges[i].point_edge_intersect(p))
                    if not constraint_bool or constraint_intersect_bool:    
                        #The method then analyzes the triangle
                        #attached to the edge in question and sees if it fails
                        #the Delaunay condition...
                        if (len(poly_edges[i].triangles)>0
                            and poly_edges[i].triangles[0].inCircumsphere(p)):
                            delete_poly_e_indices=[]
                            #...and if so, destroys the triangle and
                            #appropriately expands the polygonal hole...
                            for e in poly_edges[i].triangles[0].edges:
                                match_bool=False
                                for j in range(len(poly_edges)):
                                    if e.is_edge(poly_edges[j]):
                                        match_bool=True
                                        delete_poly_e_indices.append(j)
                                if not match_bool:
                                    poly_edges.append(e)
                            #...while keeping track of any constraints that
                            #the to-be-inserted points intersects.
                            if constraint_intersect_bool:
                                constraint=poly_edges[i]    
                            #Just some book-keeping...
                            bad_triangles.append(poly_edges[i].triangles[0])
                            self.delete_edge(poly_edges[i])
                            search_bool=True
                            #And finally the method deletes any of the old
                            #infrastructure of the polygonal hole
                            for index in sorted(delete_poly_e_indices,reverse=True):
                                del poly_edges[index]
                            if len(delete_poly_e_indices)>1:
                                break

            #The polygonal hole is retriangulated, using the edges of the
            #polygonal hole as a backbone. Care is taken to not include any
            #edges twice. All new edges and triangles are then also networked
            #into the mesh to conserve the global data structure.
            star_edges=[]
            good_triangles=[]
            for poly_e in poly_edges:
                e1=edge(poly_e.points[0],p)
                e2=edge(poly_e.points[1],p)
                match_e1_bool=False
                match_e2_bool=False
                for star_e in star_edges:
                    if not match_e1_bool and e1.is_edge(star_e):
                        e1=star_e
                        match_e1_bool=True
                    elif not match_e2_bool and e2.is_edge(star_e):
                        e2=star_e
                        match_e2_bool=True
                    if match_e1_bool and match_e2_bool:
                        break
                if not match_e1_bool:
                    e1.index=len(self.E)
                    e1.update_network()
                    self.E.append(e1)
                    star_edges.append(e1)
                    #A quick check to transfer any constraint attributes
                    #to any newly constructed edges
                    if constraint is not None:
                        if constraint.point_edge_intersect(poly_e.points[0]):
                            e1.constraint=True
                if not match_e2_bool:
                    e2.index=len(self.E)
                    e2.update_network()
                    self.E.append(e2)
                    star_edges.append(e2)
                    #A quick check to transfer any constraint attributes
                    #to any newly constructed edges
                    if constraint is not None:
                        if constraint.point_edge_intersect(poly_e.points[1]):
                            e2.constraint=True
                t1=triangle(poly_e.points[0],poly_e.points[1],p,poly_e,e1,e2)
                t1.update_network()
                t1.index=len(self.T)
                self.T.append(t1)
                good_triangles.append(t1)
            
            #A quick wrap-up to enclose any edges
            for good_t in good_triangles:
                for good_e in good_t.edges:
                    good_e.enclosed=(len(good_e.triangles)>1)
                    
            if return_changed_triangles:
                return good_triangles,bad_triangles
        elif return_changed_triangles:
            return [],[]

    def flip(self,t1,t2):
        #This method flips two triangles in the mesh around if they touch via
        #an edge. The method first searches for this diagonal edge, then
        #reconstructs the triangulation locally with the flipped triangles.
        #All old objects are replaced directly with the new ones.
        
        #Find the original diagonal (while also checking if the two
        #inputted triangles touch in the first place)
        diagonal=None
        for e1 in t1.edges:
            for e2 in t2.edges:
                if e1.is_edge(e2):
                    diagonal=e1
                    break
            if diagonal is not None:
                break
        if diagonal is not None:
            #Finds and labels the 4 points of the triangles
            polygon_points=[diagonal.points[0],diagonal.points[1]]
            for t in diagonal.triangles:
                for p in t.points:
                    if all([not p.is_point(elem) for elem in polygon_points]):
                        polygon_points.append(p)
                        break
            
            #Deletes the diagonal and the associated triangles, while still
            #preserving their location in the global data structure
            temp_e_index=diagonal.index
            temp_t1_index=t1.index
            temp_t2_index=t2.index
            self.delete_edge(diagonal)
            
            #Constructs a new diagonal and inserts it into the global data
            #structure in the same place as the original
            cross_edge=edge(polygon_points[2],polygon_points[3])
            cross_edge.index=temp_e_index
            cross_edge.update_network()
            for e in self.E[temp_e_index:]:
                e.index+=1
            self.E.insert(temp_e_index,cross_edge)
            
            #Constructs a new triangle and inserts it into the global data
            #structure in the same place as the original. It could be done
            #easier, but triangles objects specifically require the edges to be
            #found first before constructing the triangle...
            cross_triangle_edges=[]
            for j in range(2,4):
                match_bool=False
                for e1 in polygon_points[0].edges:
                    for e2 in polygon_points[j].edges:
                        if e1.is_edge(e2):
                            cross_triangle_edges.append(e1)
                            match_bool=True
                            break
                    if match_bool:
                        break
            match_bool=False
            for e1 in polygon_points[2].edges:
                for e2 in polygon_points[3].edges:
                    if e1.is_edge(e2):
                        cross_triangle_edges.append(e1)
                        match_bool=True
                        break
                if match_bool:
                    break
            
            #...and now we can construct the first new triangle
            cross_triangle_1=triangle(polygon_points[0],polygon_points[2],polygon_points[3],
                                      cross_triangle_edges[0],cross_triangle_edges[1],cross_triangle_edges[2])
            cross_triangle_1.index=temp_t1_index
            cross_triangle_1.update_network()
            for t in self.T[temp_t1_index:]:
                t.index+=1
            self.T.insert(temp_t1_index,cross_triangle_1)
            
            #Constructs a second triangle and inserts it into the global data
            #structure in the same place as the original. It could be done
            #easier, but triangles objects specifically require the edges to be
            #found first before constructing the triangle...
            cross_triangle_edges=[]
            for i in range(1,3):
                for j in range(i+1,4):
                    match_bool=False
                    for e1 in polygon_points[i].edges:
                        for e2 in polygon_points[j].edges:
                            if e1.is_edge(e2):
                                cross_triangle_edges.append(e1)
                                match_bool=True
                                break
                        if match_bool:
                            break
            #...and now we can construct the second triangle
            cross_triangle_2=triangle(polygon_points[1],polygon_points[2],polygon_points[3],
                                      cross_triangle_edges[0],cross_triangle_edges[1],cross_triangle_edges[2])
            cross_triangle_2.index=temp_t2_index
            cross_triangle_2.update_network()
            for t in self.T[temp_t2_index:]:
                t.index+=1
            self.T.insert(temp_t2_index,cross_triangle_2)
    
    def add_constraint(self,constraint):
        #Search for first bad triangle
        search_tri=None
        for t in constraint.points[0].triangles:
            if any([elem.edge_edge_intersect(constraint,includeboundary=False) for elem in t.edges]):
                search_tri=t
                break
        #Search for all bad triangles iteratively
        bad_triangles=[search_tri]
        prev_tri=search_tri
        search_bool=True
        while search_bool:
            search_bool=False
            for nt in search_tri.neighbors:
                if (nt is not None
                    and not nt.is_triangle(prev_tri)
                    and any([constraint.edge_edge_intersect(e,includeboundary=False) for e in nt.edges])):
                    prev_tri=search_tri
                    search_tri=nt
                    bad_triangles.append(search_tri)
                    search_bool=True
                    break
        #Find all the edges that intersect the constraint
        bad_edges_index=[]
        for bt in bad_triangles[::-1]:
            for e in bt.edges:
                e.enclosed=False
                if constraint.edge_edge_intersect(e,includeboundary=False):
                    if all([not e.is_edge(self.E[index]) for index in bad_edges_index]):
                        bad_edges_index.append(e.index)
        #Rearranging all the triangles by flipping to include the constraint
        delaunay_edges_index=[]
        while len(bad_edges_index)>0:
            for i,index in reversed(list(enumerate(bad_edges_index))):
                if constraint.is_edge(self.E[index]):
                    self.E[bad_edges_index[i]].constraint=True
                    del bad_edges_index[i]
                else:
                    p1=None
                    for p in self.E[index].triangles[0].points:
                        if not self.E[index].point_edge_intersect(p):
                            p1=p
                            break
                    p2=None
                    for p in self.E[index].triangles[1].points:
                        if not self.E[index].point_edge_intersect(p):
                            p2=p
                            break
                    if edge(p1,p2).edge_edge_intersect(self.E[index],includeboundary=False):
                        self.flip(self.E[index].triangles[0],self.E[index].triangles[1])
                        if not constraint.edge_edge_intersect(self.E[index],includeboundary=False):
                            delaunay_edges_index.append(index)
                            del bad_edges_index[i]
        
        #Rearrange new triangles to reintroduce constrainted delaunay properties
        done_bool=False
        while not done_bool:
            done_bool=True
            for i,index in enumerate(delaunay_edges_index):
                if not constraint.is_edge(self.E[index]):
                    delaunay_bool=True
                    p1=None
                    for p in self.E[index].triangles[0].points:
                        if not self.E[index].point_edge_intersect(p):
                            p1=p
                            break
                    if self.E[index].triangles[1].inCircumsphere(p1,includeboundary=False):
                        delaunay_bool=False
                        done_bool=False
                    if delaunay_bool:
                        p2=None
                        for p in self.E[index].triangles[1].points:
                            if not self.E[index].point_edge_intersect(p):
                                p2=p
                                break
                        if self.E[index].triangles[0].inCircumsphere(p2,includeboundary=False):
                            delaunay_bool=False
                            done_bool=False
                    if not delaunay_bool:
                        self.flip(self.E[index].triangles[0],self.E[index].triangles[1])
                        for e in self.E[index].triangles[0].edges:
                            if any([self.E[index].triangles[1].is_triangle(nt) for nt in e.triangles]):
                                delaunay_edges_index[i]=e.index
                                break
                else:
                    self.E[index].constraint=True
                            
    def constrain_triangulation(self,constraints):
        #This method prepares the triangle mesh for the addition of constraints
        #by filtering through any unnecessary situations. It then feeds the
        #constraints one by one to the add_constraint method.
        for i in range(len(constraints)-1,-1,-1):
            if not isinstance(constraints[i],edge):
                del constraints[i]
            elif any([constraints[i].is_edge(e) for e in constraints[i].points[0].edges]):
                del constraints[i]
        
        for i in range(len(constraints)-1,-1,-1):
            for j in range(len(constraints)-1,i,-1):
                if constraints[i].edge_edge_intersect(constraints[j],includeboundary=False):
                    del constraints[j]
                    
        for c in constraints:
            self.add_constraint(c)
    
    def refine(self,min_angle=20,max_area=1.0,min_length=0.5,errtol=1e-6):
        #This method implements Ruppert's refinement algorithm. It's goal is to
        #retriangulates the mesh so that it looks cleaner. It will prioritize
        #bad edges (edges whose circumsphere contains a non-end-point), splitting
        #them in half (repeatedly if necessary). It then cleans up the triangles,
        #inserting the circumspheres of bad triangles unless it causes another
        #edge to be encroached. 
        
        #Cleans up the inputted parameters of the method
        min_angle=abs((min_angle*np.pi/180)%np.pi)
        max_area=abs(max_area)
        min_length=2*abs(min_length)
        errtol=abs(errtol)
        
        #Starts by first finding all the initially encroached edges
        encroached_edges=[]
        for e in self.E:
            encroached_bool=False
            for t in e.triangles:
                for p in t.points:
                    if (not e.point_edge_intersect(p)
                        and e.inCircumsphere(p)
                        and e.length()>min_length):
                        encroached_bool=True
                        encroached_edges.append(e)
                        break
                if encroached_bool:
                    break
        
        #Then also finds all the initial bad triangles
        bad_triangles=[]
        for t in self.T:
            if t.area()>max_area or any([elem<min_angle for elem in t.angles()]):
                bad_triangles.append(t)

        #While there are still encroached edges or bad triangles...
        while len(encroached_edges)>0 or len(bad_triangles)>0:
            update_bool=False
            
            #...the method prioritizes inserting new points into the middle of
            #any encroached edge...
            if len(encroached_edges)>0:
                inserted_point=encroached_edges[0].average()
                inserted_triangles,deleted_triangles=self.insert_Vertex_Delaunay(inserted_point,return_changed_triangles=True)
                del encroached_edges[0]
                update_bool=True
            #...or otherwise inserts the circumsphere of a bad triangle into
            #the triangulation, so long as it doesn't fall outside of the
            #triangulation already present.
            else:
                inserted_point=bad_triangles[0].circumsphere().center
                if self.triangle_search(inserted_point,start_triangle=bad_triangles[0]) is not None:
                    inserted_triangles,deleted_triangles=self.insert_Vertex_Delaunay(inserted_point,return_changed_triangles=True)
                    update_bool=True
                del bad_triangles[0]
            
            #The method also keeps track if the list of encroached edges and
            #bad triangles needs to be updated, and does so by locally searching 
            #through every new triangle that was inserted or every old triangle
            #that was deleted.
            if update_bool:
                #Removing any no-longer-encroached edges from the list...
                for i in range(len(encroached_edges)-1,-1,-1):
                    if all([not elem.point_edge_intersect(encroached_edges[i].points[1]) for elem in encroached_edges[i].points[0].edges]):
                        del encroached_edges[i]
                #Removing any no-longer-bad triangles from the list...
                for i in range(len(bad_triangles)-1,-1,-1):
                    if any([elem.is_triangle(bad_triangles[i]) for elem in deleted_triangles]):
                        del bad_triangles[i]
                #Searching through all the new triangles and edges to see if
                #any are bad or encroached
                for good_t in inserted_triangles:
                    for good_e in good_t.edges:
                        if all([not good_e.is_edge(elem) for elem in encroached_edges]) and good_e.length()>min_length:
                            encroached_bool=False
                            for t in good_e.triangles:
                                for p in t.points:
                                    if not good_e.point_edge_intersect(p) and good_e.inCircumsphere(p):
                                        encroached_bool=True
                                        encroached_edges.append(good_e)
                                        break
                            if encroached_bool:
                                break
                    if good_t.area()>max_area or any([elem<min_angle for elem in good_t.angles()]):
                        bad_triangles.append(good_t)

    def delete_point(self,p):
        #This method deletes a given point and in doing so deletes any edges or
        #triangles that are attached. Care is taken to also update the global
        #data structure.
        
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
        #that are attached. Care is taken to also update the global
        #data structure.
        
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
    
    def __repr__(self):
        #This method returns a string representation of the triangle mesh.
        string_rep="Number of Triangles: "+str(len(self.T))+"\n"
        for t in self.T:
            string_rep+=str(t)+"\n"
        string_rep+="\nNumber of Edges: "+str(len(self.E))+"\n"
        for e in self.E:
            string_rep+=str(e)+"\n"
        string_rep+="\nNumber of Points: "+str(len(self.P))+"\n"
        for p in self.P:
            string_rep+=str(p)+"\n"
        return string_rep
        
    def draw(self,plotaxis=None,show_triangles=True,show_edges=True,show_points=True,show_values=False,color="black",alpha=1):    
        #This method plots the triangle mesh object into an inputted
        #figure axis object.
        
        #This starts off the method and constructs a matplotlib figure and axis
        #that the triangulation will be drawn on if none was provided.
        plotshowbool=False
        if plotaxis is None:
            if show_values:
                plotaxis=plt.figure(figsize=(graphsize,graphsize)).add_subplot(111,projection="3d")
                plotaxis.xaxis.set_rotate_label(False)
                plotaxis.yaxis.set_rotate_label(False)
                plotaxis.zaxis.set_rotate_label(False)
                plotaxis.set_zlabel("$\\mathbf{Value}$",fontsize=16,rotation=0)
                plotaxis.zaxis.set_tick_params(labelsize=16)
                plotaxis.set_xlim([min(self.P,key=lambda p:p.x).x-1,max(self.P,key=lambda p:p.x).x+1])
                plotaxis.set_ylim([min(self.P,key=lambda p:p.y).y-1,max(self.P,key=lambda p:p.y).y+1])
                plotaxis.set_zlim([min(self.P,key=lambda p:p.value).value-1,max(self.P,key=lambda p:p.value).value+1])
            else:
                plotfig=plt.figure(figsize=(graphsize,graphsize))
                plotaxis=plotfig.add_subplot(111)
                plotaxis.set_xlim([min(self.P,key=lambda p:p.x).x-1,max(self.P,key=lambda p:p.x).x+1])
                plotaxis.set_ylim([min(self.P,key=lambda p:p.y).y-1,max(self.P,key=lambda p:p.y).y+1])
            plotaxis.set_title("Triangulation",fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotshowbool=True
        
        if show_values:
            xlim=plotaxis.get_xlim()
            ylim=plotaxis.get_ylim()
            zlim=plotaxis.get_zlim()
            start=[min([p.x for p in self.P]),min([p.y for p in self.P]),min([p.value for p in self.P])]
            stop=[max([p.x for p in self.P]),max([p.y for p in self.P]),max([p.value for p in self.P])]
            
            if xlim!=(0.0,1.0):
                plotaxis.set_xlim([min(xlim[0],start[0]),max(xlim[1],stop[0])])
            else:
                plotaxis.set_xlim([start[0],stop[0]])
            if ylim!=(0.0,1.0):
                plotaxis.set_ylim([min(ylim[0],start[1]),max(ylim[1],stop[1])])
            else:
                plotaxis.set_ylim([start[1],stop[1]])
            if zlim!=(0.0,1.0):
                plotaxis.set_zlim([min(zlim[0],start[2]),max(zlim[1],stop[2])])
            else:
                plotaxis.set_zlim([start[2],stop[2]])
                
        if sum(pltcolors.to_rgb(color))<=1.0:
            color_alt="white"
        else:
            color_alt="black"
            
        if show_triangles:
            face_color=color
            if show_edges:
                edge_color=color_alt
            else:
                edge_color=color
            if show_values:
                plotaxis.add_collection3d(Poly3DCollection([[[t.points[i].x,t.points[i].y,t.points[i].value]
                                                            for i in range(3)] for t in self.T],
                                                            facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
            else:
                for t in self.T:
                    plotaxis.add_patch(plt.Polygon([[p.x,p.y] for p in t.points],facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
        elif show_edges:
            if show_values:
                for e in self.E:
                    if e.constraint:
                        plotaxis.plot3D([p.x for p in e.points],[p.y for p in e.points],[p.value for p in e.points],linewidth=4,color=color,alpha=alpha,zorder=0)
                    else:
                        plotaxis.plot3D([p.x for p in e.points],[p.y for p in e.points],[p.value for p in e.points],color=color,alpha=alpha,zorder=0)
            else:
                for e in self.E:
                    if e.constraint:
                        plotaxis.plot([p.x for p in e.points],[p.y for p in e.points],linewidth=4,color=color,alpha=alpha,zorder=0)
                    else:
                        plotaxis.plot([p.x for p in e.points],[p.y for p in e.points],color=color,alpha=alpha,zorder=0)

        if show_points:
            if show_values:
                if show_triangles:
                    plotaxis.scatter3D([p.x for p in self.P],[p.y for p in self.P],[p.value for p in self.P],facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
                else:
                    plotaxis.scatter3D([p.x for p in self.P],[p.y for p in self.P],[p.value for p in self.P],facecolor=color,edgecolor=color,alpha=alpha,zorder=1)
            else:
                if show_triangles:
                    plotaxis.scatter([p.x for p in self.P],[p.y for p in self.P],facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
                else:
                    plotaxis.scatter([p.x for p in self.P],[p.y for p in self.P],facecolor=color,edgecolor=color,alpha=alpha,zorder=1)

        if plotshowbool:
            plt.show()
