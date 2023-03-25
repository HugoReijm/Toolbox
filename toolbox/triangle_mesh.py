import numpy as np
from toolbox.point import point
from toolbox.edge import edge
from toolbox.triangle import triangle
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

graphsize=9
font = {"family": "serif",
    "color": "black",
    "weight": "bold",
    "size": "20"}

class triangle_mesh(object):
    def __init__(self,P):
        self.P=[]
        self.E=[]
        self.T=[]
        self.triangulate(P)
    
    def delete_point(self,point_index):
        #This method deletes a given point and in doing so deletes any edges or
        #triangles that are attached. Care is taken to also update the global
        #data structure.
        #Dealing with the attached triangles...
        for t in self.P[point_index].triangles:
            for nt in self.T[t.index].neighbors:
                try:
                    nt.neighbors.remove(self.T[t.index])
                except Exception:
                    pass
            for e in self.T[t.index].edges:
                try:
                    e.triangles.remove(self.T[t.index])
                except Exception:
                    pass
            for p in self.T[t.index].points:
                if not self.P[point_index].is_point(p):
                    try:
                        p.triangles.remove(self.T[t.index])
                    except Exception:
                        pass
            try:
                t_index=t.index
                del self.T[t_index]
                for i in range(t_index,len(self.T)):
                    self.T[i].index-=1
            except Exception:
                pass
        
        #Dealing with the attached edges...
        for e in self.P[point_index].edges:
            for t in self.E[e.index].triangles:
                try:
                    t.edges.remove(self.E[e.index])
                except Exception:
                    pass
            for p in self.E[e.index].points:
                if not self.P[point_index].is_point(p):
                    try:
                        p.edges.remove(self.E[e.index])
                    except Exception:
                        pass
            try:
                e_index=e.index
                del self.E[e_index]
                for i in range(e_index,len(self.E)):
                    self.E[i].index-=1
            except Exception:
                pass
        
        #Dealing with the point itself...
        try:
            del self.P[point_index]
            for i in range(point_index,len(self.P)):
                self.P[i].index-=1
        except Exception:
            pass
        
    def delete_edge(self,edge_index):
        #This method deletes a given edge and in doing so deletes any triangles
        #that are attached. Care is taken to also update the global
        #data structure.
        #Dealing with the points...
        for p in self.E[edge_index].points:
            try:
                p.edges.remove(self.E[edge_index])
            except Exception:
                pass
        
        #Dealing with the triangles...
        for t in self.E[edge_index].triangles:
            for nt in self.T[t.index].neighbors:
                try:
                    nt.neighbors.remove(self.T[t.index])
                except Exception:
                    pass
            for e in self.T[t.index].edges:
                if not self.E[edge_index].is_edge(e):
                    try:
                        e.triangles.remove(self.T[t.index])
                    except Exception:
                        pass
            for p in self.T[t.index].points:
                try:
                    p.triangles.remove(self.T[t.index])
                except Exception:
                    pass
            try:
                t_index=t.index
                del self.T[t_index]
                for i in range(t_index,len(self.T)):
                    self.T[i].index-=1
            except Exception:
                pass
        
        #Dealing with the edge itself...
        try:
            del self.E[edge_index]
            for i in range(edge_index,len(self.E)):
                self.E[i].index-=1
        except Exception:
            pass
        
    def delete_triangle(self,tri_index):
        #This method deletes a given triangle. Care is taken to also update
        #the global data structure.
        #Dealing with the neighboring triangles...
        for nt in self.T[tri_index].neighbors:
            try:
                nt.neighbors.remove(self.T[tri_index])
            except Exception:
                pass
        #Dealing with the edges...
        for e in self.T[tri_index].edges:
            try:
                e.triangles.remove(self.T[tri_index])
            except Exception:
                pass
        #Dealing with the points...
        for p in self.T[tri_index].points:
            try:
                p.triangles.remove(self.T[tri_index])
            except Exception:
                pass
        #Dealing with the triangle itself...
        try:
            del self.T[tri_index]
            for i in range(tri_index,len(self.T)):
                self.T[i].index-=1
        except Exception:
            pass
    
    def triangle_search(self,p,start_triangle=None,errtol=1e-3):
        #This method searches for the triangle that emcompasses a given point.
        #This method assumes that this point is in at least one triangle of our
        #triangular mesh.
        errtol=abs(errtol)
        if len(self.T)>0:
            #The method starts with a randomly chosen triangle in the mesh... 
            if start_triangle is None:
                t_index=np.random.randint(len(self.T))
            else:
                try:
                    t_index=start_triangle.index
                except Exception:
                    t_index=np.random.randint(len(self.T))
            prev_t_index=t_index
            #...and checks if that triangle encompasses the given point.
            found_bool=self.T[t_index].point_triangle_intersect(p)
            counter=0
            while not found_bool and counter<len(self.T):
                #If the current triangle does not encompass the given point...
                nt_index=-1
                bar=-np.inf
                #...it searches through all the neighbors
                #of the current triangle...
                for nt in self.T[t_index].neighbors:
                    for e in nt.edges:
                        if e in self.T[t_index].edges:
                            #...and checks which of the neighboring triangles
                            #is pointing the most towards the given point
                            #through the use of cross products.
                            #Loops are avoided by remembering which triangle
                            #was visited last.
                            dir_match=edge(p,e.points[0]).cross(e)
                            dir_match*=np.sign(edge(nt.average(),e.points[0]).cross(e))
                            if nt.index!=prev_t_index:
                                if dir_match > bar+errtol:
                                    nt_index=nt.index
                                    bar=dir_match
                                elif dir_match >= bar-errtol and np.random.rand()<0.5:
                                    nt_index=nt.index
                                    bar=dir_match
                            break
                #The method then moves to the best neighboring triangle and
                #checks again if that triangle encompasses the given point,
                #repeating the cycle.
                prev_t_index=t_index
                t_index=nt_index
                found_bool=self.T[t_index].point_triangle_intersect(p)
                counter+=1
            if found_bool:
                return self.T[t_index]
        return None
    
    def insert_Vertex_Delaunay(self,p,return_changed_triangles=False):
        #This method inserts a point into a triangular mesh which is assumed
        #to fulfil the Delaunay requisite already. The point is then added
        #using the Bowyer-Watson Algorithm, which then results in a local
        #retriangulation of the mesh
        if all([not pointVar.is_point(p) for pointVar in self.P]):
            p.index=len(self.P)
            self.P.append(p)
            poly_edges=[]
            star_edges=[]
            
            #First, the method searches for the triangles that need to
            #be changed. It first looks for the triangle that contains
            #the point that is about to be inserted using an efficient
            #(hopefully O(log(N)) runtime) search algorithm. Then it performs
            #a breadth-first search to find all the triangles whose circumcircle
            #encompasses the given point.
            #bad_triangles=[t for t in self.T if t.inCircumcircle(p)]
            triangle_zero=self.triangle_search(p)
            bad_triangles=[triangle_zero]
            search_triangles=[triangle_zero]
            search_bool=True
            while search_bool:
                search_bool=False
                for i in range(len(search_triangles)-1,-1,-1):
                    for nt in search_triangles[i].neighbors:
                        if (all([not nt.is_triangle(elem) for elem in bad_triangles])
                            and nt.inCircumcircle(p)):
                            search_triangles.append(nt)
                            bad_triangles.append(nt)
                            search_bool=True
                    search_triangles=search_triangles[:i]+search_triangles[i+1:]

            #From there, the bad triangles are removed from the mesh and their
            #edges are added to the polygonal hole that needs to be
            #retriangulated
            for i in range(len(bad_triangles)-1,-1,-1):
                for v in bad_triangles[i].points:
                    star_e=edge(p,v)
                    if all([not star_e.is_edge(elem) for elem in star_edges]):
                        star_edges.append(star_e)
                for e in bad_triangles[i].edges:
                    if all([not e.is_edge(elem) for elem in poly_edges]):
                        poly_edges.append(e)
                self.delete_triangle(bad_triangles[i].index)
            
            #The polygonal hole is cleaned to remove any edges from the interior
            #of the hole
            for star_e in star_edges:
                restart=True
                while restart:
                    restart=False
                    for i,poly_e in enumerate(poly_edges):
                        if poly_e.edge_edge_intersect(star_e,includeboundary=False):
                            del poly_edges[i]
                            self.delete_edge(poly_e.index)
                            restart=True
                            break
            
            #The polygonal hole is retriangulated, using the edges of the
            #polygonal hole as a backbone. Care is taken to not include any
            #edges twice. All new edges and triangles are then also networked
            #into the mesh to conserve the global data structure.
            star_edges.clear()
            good_triangles=[]
            for e in poly_edges:
                e1=edge(e.points[0],p)
                match_bool=False
                for elem in star_edges:
                    if e1.is_edge(elem):
                        e1=elem
                        match_bool=True
                        break
                if not match_bool:
                    e1.index=len(self.E)
                    e1.update()
                    self.E.append(e1)
                    star_edges.append(e1)
                
                e2=edge(e.points[1],p)
                match_bool=False
                for elem in star_edges:
                    if e2.is_edge(elem):
                        e2=elem
                        match_bool=True
                        break
                if not match_bool:
                    e2.index=len(self.E)
                    e2.update()
                    self.E.append(e2)
                    star_edges.append(e2)
                
                t=triangle(e.points[0],e.points[1],p,e,e1,e2)
                t.update()
                t.index=len(self.T)
                self.T.append(t)
                good_triangles.append(t)
            
            for good_t in good_triangles:
                for good_e in good_t.edges:
                    if len(good_e.triangles)==1:
                        good_e.enclosed=False
                    elif len(good_e.triangles)>1:
                        good_e.enclosed=True
            
            if return_changed_triangles:
                return good_triangles,bad_triangles
        elif return_changed_triangles:
            return [],[]
        
    def triangulate(self,P,constraints=[]):    
        #This method performs the Bowyer-Watson triangle mesh generation
        #from a given set of points P. It first sets up a beginner triangle
        #that encompasses all points, and then iteratively adds all points one
        #by one, retriangulating locally as it goes.
        
        #Set up of initial triangle that encompasses all the points
        maxX=max(P,key=lambda p:p.x).x
        minX=min(P,key=lambda p:p.x).x
        maxY=max(P,key=lambda p:p.y).y
        minY=min(P,key=lambda p:p.y).y
        cx,cy=(maxX+minX)/2,(maxY+minY)/2
        r=np.sqrt((maxX-minX)**2+(maxY-minY)**2)/2+1
        init_p1=point(cx,cy+2*r)
        init_p1.index=0
        init_p2=point(cx+np.sqrt(3)*r,cy-r)
        init_p2.index=1
        init_p3=point(cx-np.sqrt(3)*r,cy-r)
        init_p3.index=2
        self.P.clear()
        self.P.append(init_p1)
        self.P.append(init_p2)
        self.P.append(init_p3)
        
        init_e1=edge(init_p1,init_p2)
        init_e1.index=0
        init_e1.update()
        self.E.append(init_e1)
        init_e2=edge(init_p1,init_p3)
        init_e2.index=1
        init_e2.update()
        self.E.append(init_e2)
        init_e3=edge(init_p2,init_p3)
        init_e3.index=2
        init_e3.update()
        self.E.append(init_e3)
        
        init_t=triangle(init_p1,init_p2,init_p3,init_e1,init_e2,init_e3)
        init_t.index=0
        init_t.update()
        self.T=[init_t]
        
        #Iteratively calls upon the insert_Vertex_Delaunay method
        for p in P:
            if isinstance(p,point):
                self.insert_Vertex_Delaunay(p)
        
        #Adds any constraints to the mesh
        for c in constraints:
            if isinstance(c,edge):
                self.add_constraint(c)
        
        #Removes the initial triangle and performs any additional clean up
        self.delete_point(2)
        self.delete_point(1)
        self.delete_point(0)
        
        for e in self.E:
            if len(e.triangles)<2:
                e.enclosed=False

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
            self.delete_edge(diagonal.index)
            
            #Constructs a new diagonal and inserts it into the global data
            #structure in the same place as the original
            cross_edge=edge(polygon_points[2],polygon_points[3])
            cross_edge.index=temp_e_index
            cross_edge.update()
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
            cross_triangle_1.update()
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
            cross_triangle_2.update()
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
                if not nt.is_triangle(prev_tri) and any([constraint.edge_edge_intersect(e,includeboundary=False) for e in nt.edges]):
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
                    if self.E[index].triangles[1].inCircumcircle(p1,includeboundary=False):
                        delaunay_bool=False
                        done_bool=False
                    if delaunay_bool:
                        p2=None
                        for p in self.E[index].triangles[1].points:
                            if not self.E[index].point_edge_intersect(p):
                                p2=p
                                break
                        if self.E[index].triangles[0].inCircumcircle(p2,includeboundary=False):
                            delaunay_bool=False
                            done_bool=False
                    if not delaunay_bool:
                        self.flip(self.E[index].triangles[0],self.E[index].triangles[1])
                        for e in self.E[index].triangles[0].edges:
                            if any([self.E[index].triangles[1].is_triangle(nt) for nt in e.triangles]):
                                delaunay_edges_index[i]=e.index
                                break
                            
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
        #bad edges (edges whose circumcircle contains a non-end-point), splitting
        #them in half (repeatedly if necessary). It then cleans up the triangles,
        #inserting the circumcircles of bad triangles unless it causes another
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
                    if not e.point_edge_intersect(p) and e.inCircumcircle(p) and e.length()>min_length:
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
            #...or otherwise inserts the circumcircle of a bad triangle into
            #the triangulation, so long as it doesn't fall outside of the
            #triangulation already present.
            else:
                inserted_point=bad_triangles[0].circumcircle().center
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
                                    if not good_e.point_edge_intersect(p) and good_e.inCircumcircle(p):
                                        encroached_bool=True
                                        encroached_edges.append(good_e)
                                        break
                            if encroached_bool:
                                break
                    if good_t.area()>max_area or any([elem<min_angle for elem in good_t.angles()]):
                        bad_triangles.append(good_t)

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
        
    def draw(self,plotaxis=None,show_triangles=True,show_edges=True,show_points=True,show_circumcircle=False,color="black",alpha=1):    
        #This method plots the triangle mesh object into an inputted
        #figure axis object.
        
        #This starts off the method and constructs a matplotlib figure and axis
        #that the triangulation will be drawn on if none was provided.
        plotshowbool=False
        if plotaxis is None:
            plotfig=plt.figure(figsize=(graphsize,graphsize))
            plotaxis=plotfig.add_subplot(111)
            plotaxis.set_title("Function Point",fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotshowbool=True
        
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
            for t in self.T:
                plotaxis.add_patch(plt.Polygon([[p.x,p.y] for p in t.points],facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
        elif show_edges:
            for e in self.E:
                if e.constraint:
                    plotaxis.plot([p.x for p in e.points],[p.y for p in e.points],linewidth=4,color=color,alpha=alpha,zorder=0)
                else:
                    plotaxis.plot([p.x for p in e.points],[p.y for p in e.points],color=color,alpha=alpha,zorder=0)
    
        if show_circumcircle:
            theta=np.linspace(0,2*np.pi,100)
            for t in self.T:
                plotaxis.plot(t.circumcircle().radius*np.cos(theta)+t.circumcircle().center.x,
                              t.circumcircle().radius*np.sin(theta)+t.circumcircle().center.y,
                              color="red",alpha=alpha)
        if show_points:
            if show_triangles:
                plotaxis.scatter([p.x for p in self.P],[p.y for p in self.P],facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
            else:
                plotaxis.scatter([p.x for p in self.P],[p.y for p in self.P],facecolor=color,edgecolor=color,alpha=alpha,zorder=1)
        
        if plotshowbool:
            plt.show()
