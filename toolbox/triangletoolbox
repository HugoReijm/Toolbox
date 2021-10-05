import math
import numpy as np
from point import point
from edge import edge
from triangle import triangle
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt

def draw(P,E,T,points=True,edges=True,triangles=True,plotaxis=None,color="black",alpha=1):
    #Draws a triangulation
    plotshowbool=False
    if plotaxis is None:
        graphsize=9
        font = {"family": "serif",
            "color": "black",
            "weight": "bold",
            "size": "20"}
        plotfig=plt.figure(figsize=(graphsize,graphsize))
        plotaxis=plotfig.add_subplot(111)
        plotaxis.set_title("Triangulation",fontdict=font)
        plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
        plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
        plotaxis.xaxis.set_tick_params(labelsize=16)
        plotaxis.yaxis.set_tick_params(labelsize=16)
        plotshowbool=True
        
    xlim = plotaxis.get_xlim()
    ylim = plotaxis.get_ylim()
    minX,maxX=min([elem.x for elem in P]),max([elem.x for elem in P])
    minY,maxY=min([elem.y for elem in P]),max([elem.y for elem in P])
    minX,maxX=1.1*minX-0.1*maxX,1.1*maxX-0.1*minX
    minY,maxY=1.1*minY-0.1*maxY,1.1*maxY-0.1*minY
    plotaxis.set_xlim([min(xlim[0],minX),max(xlim[1],maxX)])
    plotaxis.set_ylim([min(ylim[0],minY),max(ylim[1],maxY)])
    
    if sum(pltcolors.to_rgb(color))<=1.0:
        color_alt="white"
    else:
        color_alt="black"
        
    if triangles:
        face_color=color
        if edges:
            edge_color=color_alt
        else:
            edge_color=color
        for t in T:
            plotaxis.add_patch(plt.Polygon([[p.x,p.y] for p in t.points],facecolor=face_color,edgecolor=edge_color,alpha=alpha,zorder=0))
    else:
        if edges:
            for e in E:
                plotaxis.plot([p.x for p in e.points],[p.y for p in e.points],color=color,alpha=alpha,zorder=0)
    
    if points:
        if triangles:
            plotaxis.scatter([p.x for p in P],[p.y for p in P],facecolor=color,edgecolor=color_alt,alpha=alpha,zorder=1)
        else:
            plotaxis.scatter([p.x for p in P],[p.y for p in P],facecolor=color,edgecolor=color,alpha=alpha,zorder=1)
    
    if plotshowbool:
        plt.show()
    
def convex_hull(P,errtol=1e-12):
    #Quickhull algorithm that computes the convex hull of a number of points  
    N=len(P)
    if N<=1:
        print("Error: Unable to make a convex hull due to lack of sufficient points")
        return []
    elif N==2:
        return [edge(P[0],P[1])]
    else:
        #Checks if set of points is colinear
        e0=edge(P[0],P[1])
        colinearbool=True
        for i in range(2,N):
            if abs(e0.cross(edge(P[0],P[i])))>abs(errtol):
                colinearbool=False
                break
        if colinearbool:
            print("Error: point set is colinear, unable to make convex hull")
            return []
       
    #Finds the left_most and down_most point of the set
    left_anchor=P[0]
    for i in range(1,N):
        if P[i].x<left_anchor.x:
            left_anchor=P[i]
        elif P[i].x==left_anchor.x and P[i].y<left_anchor.y:
            left_anchor=P[i]
    
    #Finds the right_most and up_most point of the set
    right_anchor=P[0]
    for i in range(1,N):
        if P[i].x>right_anchor.x:
            right_anchor=P[i]
        elif P[i].x==right_anchor.x and P[i].y>right_anchor.y:
            right_anchor=P[i]
    
    #Finds the farthest points above the line between left_anchor and right_anchor
    farthest_anchor=left_anchor
    height=0
    for p in P:
        res=-(right_anchor.y-left_anchor.y)*p.x+(right_anchor.x-left_anchor.x)*p.y-right_anchor.x*left_anchor.y+right_anchor.y*left_anchor.x
        if (right_anchor.x-left_anchor.x)*res>abs(errtol) and res>height+abs(errtol):
            farthest_anchor=p
            height=res
        
    #Sorting points that into an upper half, but only focussing on those that dont yet fall into the contructed convex hull
    e0=edge(left_anchor,right_anchor)
    upper_hull=[edge(left_anchor,farthest_anchor),edge(farthest_anchor,right_anchor)]
    exterior=[]
    tri=triangle(left_anchor,farthest_anchor,right_anchor)
    tri.edges=[e0]+upper_hull
    for p in P:
        if not p.is_point(left_anchor) and not p.is_point(farthest_anchor) and not p.is_point(right_anchor):
            res=-(right_anchor.y-left_anchor.y)*p.x+(right_anchor.x-left_anchor.x)*p.y-right_anchor.x*left_anchor.y+right_anchor.y*left_anchor.x
            if (right_anchor.x-left_anchor.x)*res>abs(errtol) and not tri.point_triangle_intersect(p,includeboundary=False):
                exterior.append(p)
            
    #Fully constructing the upper convex hull iteratively
    counter=0
    while len(exterior)>0 and counter<N:
        for i in range(len(upper_hull)):
            res1=upper_hull[i].points[1].x-upper_hull[i].points[0].x
            if res1!=0:
                #Again finds the farthest points below line between two connected points in the convex hull
                farthest_anchor=upper_hull[i].points[0]
                height=0
                for p in exterior:
                    res2=-(upper_hull[i].points[1].y-upper_hull[i].points[0].y)*p.x+(upper_hull[i].points[1].x-upper_hull[i].points[0].x)*p.y-upper_hull[i].points[1].x*upper_hull[i].points[0].y+upper_hull[i].points[1].y*upper_hull[i].points[0].x
                    if res1*res2>abs(errtol) and res2>height+abs(errtol):
                        farthest_anchor=p
                        height=res2

                #If there exists a point outside the convex hull, change the hull to include it
                if height>abs(errtol) or not farthest_anchor.is_point(upper_hull[i].points[0]):
                    e1=edge(upper_hull[i].points[0],farthest_anchor)
                    e2=edge(farthest_anchor,upper_hull[i].points[1])
                    upper_hull+=[e1,e2]
                    
                    if height>abs(errtol):
                        tri=triangle(upper_hull[i].points[0],upper_hull[i].points[1],farthest_anchor)
                        for j in range(len(exterior)-1,-1,-1):
                            if tri.point_triangle_intersect(exterior[j],includeboundary=False):
                                del exterior[j]

                    exterior.remove(farthest_anchor)
                    del upper_hull[i]
                    break
        counter+=1
    
    #Finds the farthest points below the line between left_anchor and right_anchor
    farthest_anchor=right_anchor
    height=0
    for p in P:
        res=-(right_anchor.y-left_anchor.y)*p.x+(right_anchor.x-left_anchor.x)*p.y-right_anchor.x*left_anchor.y+right_anchor.y*left_anchor.x
        if (right_anchor.x-left_anchor.x)*res<-abs(errtol) and res+abs(errtol)<height:
            farthest_anchor=p
            height=res
    
    #Sorting points that into a lower half, but only focussing on those that dont yet fall into the contructed convex hull
    lower_hull=[edge(left_anchor,farthest_anchor),edge(farthest_anchor,right_anchor)]
    exterior=[]
    tri=triangle(left_anchor,farthest_anchor,right_anchor)
    tri.edges=[e0]+lower_hull
    for p in P:
        if not p.is_point(left_anchor) and not p.is_point(farthest_anchor) and not p.is_point(right_anchor):
            res=-(right_anchor.y-left_anchor.y)*p.x+(right_anchor.x-left_anchor.x)*p.y-right_anchor.x*left_anchor.y+right_anchor.y*left_anchor.x
            if (right_anchor.x-left_anchor.x)*res<-abs(errtol) and not tri.point_triangle_intersect(p,includeboundary=False):
                exterior.append(p)    
    
    #Fully constructing the lower convex hull iteratively
    counter=0
    while len(exterior)>0 and counter<N:
        for i in range(len(lower_hull)):
            res1=lower_hull[i].points[1].x-lower_hull[i].points[0].x
            if res1!=0:
                #Again finds the farthest points below line between two connected points in the convex hull
                farthest_anchor=lower_hull[i].points[0]
                height=0
                for p in exterior:
                    res2=-(lower_hull[i].points[1].y-lower_hull[i].points[0].y)*p.x+(lower_hull[i].points[1].x-lower_hull[i].points[0].x)*p.y-lower_hull[i].points[1].x*lower_hull[i].points[0].y+lower_hull[i].points[1].y*lower_hull[i].points[0].x
                    if res1*res2<-abs(errtol) and res2+abs(errtol)<height:
                        farthest_anchor=p
                        height=res2
                
                #If there exists a point outside the convex hull, change the hull to include it
                if height<-abs(errtol) or not farthest_anchor.is_point(lower_hull[i].points[0]):
                    e1=edge(lower_hull[i].points[0],farthest_anchor)
                    e2=edge(farthest_anchor,lower_hull[i].points[1])
                    lower_hull+=[e1,e2]
                    
                    if height<-abs(errtol):
                        tri=triangle(lower_hull[i].points[0],lower_hull[i].points[1],farthest_anchor)
                        for j in range(len(exterior)-1,-1,-1):
                            if tri.point_triangle_intersect(exterior[j],includeboundary=False):
                                del exterior[j]    
                    
                    exterior.remove(farthest_anchor)
                    del lower_hull[i]
                    break
        counter+=1
    return upper_hull+lower_hull

def scan_triangulate(P,errtol=1e-12):
    #Scan triangulation algorithm that triangulates points by simply connecting them haphazardly
    N=len(P)
    P.sort(key=lambda p:p.x)
    E=[]
    T=[]
    
    if N<2:
        return E,T
    elif N==2:
        e=edge(P[0],P[1])
        e.index=0
        e.update()
        E.append(e)
        return E,T
    else:
        #Checks if set of points is colinear
        e0=edge(P[0],P[1])
        colinearbool=True
        for i in range(2,N):
            if abs(e0.cross(edge(P[0],P[i])))>abs(errtol):
                colinearbool=False
                break
        if colinearbool:
            print("Error: point set is colinear, unable to make convex hull")
            E=[edge(P[i],P[i+1]) for i in range(N-1)]
            for i in range(N-1):
                E[i].index=i
            return P,E,T
    
    #Makes the first triangle
    startindex=-1
    for i in range(2,N):
        t0=triangle(P[0],P[1],P[i])
        t0.index=0
        if t0.area()>abs(errtol):
            T.append(t0)
            e1=edge(P[0],P[1])
            e1.index=0
            e1.update()
            e2=edge(P[0],P[i])
            e2.index=1
            e2.update()
            e3=edge(P[1],P[i])
            e3.index=2
            e3.update()
            t0.edges=[e1,e2,e3]
            t0.update()
            E=[e1,e2,e3]
            startindex=i
            P[0].index=0
            P[1].index=1
            P[i].index=2
            break
    
    if startindex==-1:
        print("All points are colinear")
    else:
        point_index=3
        edge_index=3
        tri_index=1
        #Adds more triangles to the triangulation whenever possible
        for i in range(2,N):
            if i!=startindex:
                #Every point is added incrementally to the triangulation
                P[i].index=point_index
                point_index+=1
                newE=[]
                newEbool=[]
                for e in E:
                    if not e.enclosed:
                        #An edge is made to the added point and tested to be unique and not crossing any other edge   
                        e1=edge(e.points[0],P[i])
                        match1bool=False
                        for j in range(len(newE)):
                            if e1.is_edge(newE[j]):
                                e1=newE[j]
                                e1bool=newEbool[j]
                                match1bool=True
                                break
                        if not match1bool:
                            e1bool=any([elem.edge_edge_intersect(e1,includeboundary=False) for elem in E])
                        
                        #An edge is made to the added point and tested to be uniue and not crossing any other edge
                        e2=edge(e.points[1],P[i])
                        match2bool=False
                        for j in range(len(newE)):
                            if e2.is_edge(newE[j]):
                                e2=newE[j]
                                e2bool=newEbool[j]
                                match2bool=True
                                break
                        if not match2bool:
                            e2bool=any([elem.edge_edge_intersect(e2,includeboundary=False) for elem in E])
                        
                        #If both edges are viable, a new triangle is made
                        if not e1bool and not e2bool:
                            t=triangle(e.points[0],e.points[1],P[i])
                            t.edges=[e,e1,e2]
                            if all([not t.triangle_triangle_intersect(tri,includeboundary=False) for tri in T]):
                                t.index=tri_index
                                tri_index+=1
                                e.enclosed=True
                                if not match1bool:
                                    e1.update()
                                    newE.append(e1)
                                    newEbool.append(e1bool)
                                    e1.index=edge_index
                                    edge_index+=1
                                else:
                                    e1.enclosed=True
                                if not match2bool:
                                    e2.update()
                                    newE.append(e2)
                                    newEbool.append(e2bool)
                                    e2.index=edge_index
                                    edge_index+=1
                                else:
                                    e2.enclosed=True
                                t.update()
                                T.append(t)
                E=E+newE
    return P,E,T

def delaunize(P,E,T,errtol=1e-12):
    delaunaybool=False
    counter=0
    N=len(E)
    while not delaunaybool and counter<N*(N-1)/2+1:
        delaunaybool=True
        for e in E:
            if e.enclosed:
                p1=e.triangles[0].points[0]
                if p1 in e.points:
                    p1=e.triangles[0].points[1]
                    if p1 in e.points:
                        p1=e.triangles[0].points[2]
                
                p2=e.triangles[1].points[0]
                if p2 in e.points:
                    p2=e.triangles[1].points[1]
                    if p2 in e.points:
                        p2=e.triangles[1].points[2]
                
                if e.triangles[0].inCircumcircle(p2):
                    if all([abs(edge(p1,e.points[i]).cross(edge(p2,e.points[i])))>abs(errtol) for i in range(2)]):
                        t1=e.triangles[0]
                        t2=e.triangles[1]
                                              
                        for elem in t1.edges:
                            elem.triangles.remove(t1)
                        for elem in t1.points:
                            elem.triangles.remove(t1)
                        for elem in t2.edges:
                            elem.triangles.remove(t2)
                        for elem in t2.points:
                            elem.triangles.remove(t2)
                        for elem in e.points:
                            elem.edges.remove(e)
                        
                        temppoints=[elem for elem in e.points]
                        e.points=[p1,p2]
                        
                        tempt1edges=[elem for elem in t1.edges if not e.is_edge(elem)]
                        tempt2edges=[elem for elem in t2.edges if not e.is_edge(elem)]
                        if tempt1edges[0].edge_edge_intersect(tempt2edges[0],includeboundary=True):
                            t1.edges=[tempt1edges[0],tempt2edges[0],e]
                            t2.edges=[tempt1edges[1],tempt2edges[1],e]
                        else:
                            t1.edges=[tempt1edges[0],tempt2edges[1],e]
                            t2.edges=[tempt1edges[1],tempt2edges[0],e]
                        
                        if temppoints[0] in tempt1edges[0].points:
                            t1.points=[p1,p2,temppoints[0]]
                            t2.points=[p1,p2,temppoints[1]]
                        else:
                            t1.points=[p1,p2,temppoints[1]]
                            t2.points=[p1,p2,temppoints[0]]
                            
                        t1.update()
                        t2.update()
                        e.update()
                        t1.triagarea=None
                        t2.triagarea=None
                        e.edgelength=None
                        delaunaybool=False
            counter+=1
            
def _insert_Vertex_Delaunay_(point,P,E,T):
    polyedges=[]
    staredges=[]
    constraintpolypoints=[]
    for i in range(len(T)-1,-1,-1):
        if T[i].inCircumcircle(point,includeboundary=True):
            if T[i].point_triangle_intersect(point):
                for j in range(len(T[i].edges)):
                    if T[i].edges[j].point_edge_intersect(point,includeboundary=False):
                        if T[i].edges[j].constraint:
                            constraintpolypoints+=T[i].edges[j].points
                        break
                for e in T[i].edges:
                    if e not in polyedges:
                        polyedges.append(e)
                for p in T[i].points:
                    starE=edge(point,p)
                    if all([not starE.is_edge(elem) for elem in staredges]):
                        staredges.append(starE)
                
                for p in T[i].points:
                    p.triangles.remove(T[i])
                for e in T[i].edges:
                    e.triangles.remove(T[i])
                del T[i]
            else:
                intersectbool=False
                for j,e in enumerate(T[i].edges):
                    for p in T[i].points:
                        if not p.is_point(e.points[0]) and not p.is_point(e.points[1]):
                            starE=edge(point,p)
                            if starE.edge_edge_intersect(e,includeboundary=False):
                                #if all([not starE.edge_edge_intersect(E[k],includeboundary=False) for k in constraintIndex]):
                                if all([not starE.edge_edge_intersect(elem,includeboundary=False) for elem in E if elem.constraint]):
                                    for k in range(j):
                                        if T[i].edges[k] not in polyedges:
                                            polyedges.append(T[i].edges[k])
                                    for k in range(j+1,3):
                                        if T[i].edges[k] not in polyedges:
                                            polyedges.append(T[i].edges[k])
                                    for elemP in T[i].points:    
                                        starE=edge(point,elemP)
                                        if all([not starE.is_edge(elem) for elem in staredges]):
                                            staredges.append(starE)
                                    intersectbool=True
                                    for elemP in T[i].points:
                                        elemP.triangles.remove(T[i])
                                    for elemE in T[i].edges:
                                        elemE.triangles.remove(T[i])
                                    del T[i]
                            break
                    if intersectbool:
                        break
                    
    for i in range(len(polyedges)-1,-1,-1):
        if any([polyedges[i].edge_edge_intersect(starE,includeboundary=False) for starE in staredges]):
            polyedges[i].points[0].edges.remove(polyedges[i])
            polyedges[i].points[1].edges.remove(polyedges[i])
            E.remove(polyedges[i])
            del polyedges[i]
            #indexE=E.index(polyedges[i])
            #del polyedges[i]
            #del E[indexE]
            #for j in range(len(constraintIndex)-1,-1,-1):
            #    if constraintIndex[j]>indexE:
            #        constraintIndex[j]-=1
            #    elif constraintIndex[j]==indexE:
            #        del constraintIndex[j]
    
    for e in polyedges:
        e1=edge(e.points[0],point)
        e2=edge(e.points[1],point)
        
        t=triangle(e.points[0],e.points[1],point)
        t.edges=[e]
        matchbool=False
        for elem in E:
            if e1.is_edge(elem):
                matchbool=True
                t.edges.append(elem)
                break
        if not matchbool:
            if e.points[0] in constraintpolypoints:
                e1.constraint=True
                #constraintIndex.append(len(E))
            E.append(e1)
            t.edges.append(e1)
            e1.update()
            e1.enclosed=True
        matchbool=False
        for elem in E:
            if e2.is_edge(elem):
                matchbool=True
                t.edges.append(elem)
                break
        if not matchbool:
            if e.points[1] in constraintpolypoints:
                e2.constraint=True
                #constraintIndex.append(len(E))
            E.append(e2)
            t.edges.append(e2)
            e2.update()
            e2.enclosed=True
        t.update()
        T.append(t)
        
    matchbool=False
    for p in P:
        if p.is_point(point):
            matchbool=True
            break
    if not matchbool:
        P.append(point)

def _patch_(P,E,T,constr,anchor,cavityPoints,involvedE):
    while not anchor.points[1].is_point(constr.points[0]):
        paths=[]
        for e in anchor.points[1].edges:
            if not e.enclosed and not e.is_edge(anchor):
                if e.points[1].is_point(anchor.points[1]):
                    e.swap()
                paths.append(e)    
        if len(paths)==0:
            anchor.swap()
            cavityPoints.append(anchor.points[1])
        elif len(paths)==1:
            anchor=paths[0]
            cavityPoints.append(anchor.points[1])
        else:
            cwIndex=0
            thetaprime=math.atan2(anchor.points[0].y-anchor.points[1].y,
                              anchor.points[0].x-anchor.points[1].x)
            if thetaprime<0:
                thetaprime+=2*math.pi
            minAngle=math.atan2(paths[0].points[1].y-paths[0].points[0].y,paths[0].points[1].x-paths[0].points[0].x)-thetaprime
            while minAngle<=0:
                minAngle+=2*math.pi
            for i in range(1,len(paths)):
                res=math.atan2(paths[i].points[1].y-paths[i].points[0].y,paths[i].points[1].x-paths[i].points[0].x)-thetaprime
                while res<=0:
                    res+=2*math.pi
                if res<minAngle:
                    minAngle=res
                    cwIndex=i
            anchor=paths[cwIndex]
            cavityPoints.append(anchor.points[1])
    
    segments=[cavityPoints]
    segmentE=[constr]
    while len(segments)>0:
        for i in range(len(segments)-1,-1,-1):
            if len(segments[i])==3:
                e1=segments[i][1].edges[0]
                e2=segments[i][1].edges[1]
                ref1=edge(segments[i][0],segments[i][1])
                ref2=edge(segments[i][-1],segments[i][1])
                for j in range(len(segments[i][1].edges)):
                    if segments[i][1].edges[j].is_edge(ref1):
                        e1=segments[i][1].edges[j]
                    elif segments[i][1].edges[j].is_edge(ref2):
                        e2=segments[i][1].edges[j]
                involvedE+=[e1,e2]
                tri=triangle(segments[i][0],segments[i][1],segments[i][-1])
                tri.edges=[e1,e2,segmentE[i]]
                tri.update()
                T.append(tri)
                del segments[i]
                del segmentE[i]
            else:
                for j in range(1,len(segments[i])-1):
                    tri=triangle(segments[i][0],segments[i][j],segments[i][-1])
                    validbool=True
                    for k in range(1,len(segments[i])-1):
                        if k!=j and tri.inCircumcircle(segments[i][k],includeboundary=False):
                            validbool=False
                            break
                    if validbool:
                        if j==1:
                            e1=segments[i][1].edges[0]
                            ref1=edge(segments[i][0],segments[i][1])
                            for k in range(1,len(segments[i][1].edges)):
                                if segments[i][1].edges[k].is_edge(ref1):
                                    e1=segments[i][1].edges[k]
                                    break
                            e2=edge(segments[i][1],segments[i][-1])
                            e2.update()
                            E.append(e2)
                            involvedE+=[e1,e2]
                            tri.edges=[e1,e2,segmentE[i]]
                            tri.update()
                            T.append(tri)
                            segments[i]=segments[i][1:]
                            segmentE[i]=e2
                        elif j==len(segments[i])-2:
                            e1=edge(segments[i][0],segments[i][-2])
                            e1.update()
                            e2=segments[i][-2].edges[0]
                            ref2=edge(segments[i][-2],segments[i][-1])
                            for k in range(1,len(segments[i][-2].edges)):
                                if segments[i][-2].edges[k].is_edge(ref2):
                                    e2=segments[i][-2].edges[k]
                                    break
                            E.append(e1)
                            involvedE+=[e1,e2]
                            tri.edges=[e1,e2,segmentE[i]]
                            tri.update()
                            T.append(tri)
                            segments[i]=segments[i][0:-1]
                            segmentE[i]=e1
                        else:
                            e1=edge(segments[i][0],segments[i][j])
                            e1.update()
                            e2=edge(segments[i][j],segments[i][-1])
                            e2.update()
                            E+=[e1,e2]
                            involvedE+=[e1,e2]
                            tri.edges=[segmentE[i],e1,e2]
                            tri.update()
                            T.append(tri)
                            segments.append(segments[i][j:])
                            segmentE.append(e2)
                            segments[i]=segments[i][0:j+1]
                            segmentE[i]=e1
                        break

def _constrained_(P,E,T,constraints):    
    for constr in constraints:
        for i in range(len(T)-1,-1,-1):
            if T[i].edge_triangle_intersect(constr,includeboundary=False) or any([elem.edge_edge_intersect(constr,includeboundary=False) for elem in T[i].edges]):
                for elem in T[i].points:
                    elem.triangles.remove(T[i])
                for elem in T[i].edges:
                    if elem.edge_edge_intersect(constr,includeboundary=False):
                        try:
                            elem.points[0].edges.remove(elem)
                            elem.points[1].edges.remove(elem)
                            E.remove(elem)
                            #indexE=E.index(elem)
                            #del E[indexE]
                            #for j in range(len(constraintIndex)):
                            #    if constraintIndex[j]>indexE:
                            #        constraintIndex[j]-=1
                            #    elif constraintIndex[j]==indexE:
                            #        del constraintIndex[j]
                        except:
                            pass
                    else:
                        elem.triangles.remove(T[i])
                        elem.enclosed=False
                del T[i]
        
        #constr.enclosed=True
        constr.update()
        #constraintIndex.append(len(E))
        E.append(constr)
        
        involvedE=[constr]
        
        anchor=constr
        cavityPoints=[constr.points[1]]
        _patch_(P,E,T,constr,anchor,cavityPoints,involvedE)
        
        anchor=constr
        anchor.swap()
        cavityPoints=[constr.points[1]]
        _patch_(P,E,T,constr,anchor,cavityPoints,involvedE)
        
        for e in involvedE:
            if len(e.triangles)<2:
                e.enclosed=False
            else:
                e.enclosed=True

def delaunay_triangulate(P,constraints=[]):    
    maxX=P[0].x
    minX=P[0].x
    maxY=P[0].y
    minY=P[0].y
    for i in range(1,len(P)):
        if P[i].x>maxX:
            maxX=P[i].x
        elif P[i].x<minX:
            minX=P[i].x
        if P[i].y>maxY:
            maxY=P[i].y
        elif P[i].y<minY:
            minY=P[i].y
    cx=(maxX+minX)/2
    cy=(maxY+minY)/2
    r=math.sqrt((maxX-minX)**2+(maxY-minY)**2)/2+1
    p01=point(cx,cy+2*r)
    p02=point(cx+math.sqrt(3)*r,cy-r)
    p03=point(cx-math.sqrt(3)*r,cy-r)
    P.insert(0,p03)
    P.insert(0,p02)
    P.insert(0,p01)
    
    e01=edge(p01,p02)
    e01.update()
    e02=edge(p01,p03)
    e02.update()
    e03=edge(p02,p03)
    e03.update()
    E=[e01,e02,e03]
    
    t0=triangle(p01,p02,p03)
    t0.edges=[e01,e02,e03]
    t0.update()
    T=[t0]
    
    for p in P[3:]:
        _insert_Vertex_Delaunay_(p,P,E,T)
    
    for i in range(len(T)-1,-1,-1):
        if (p01 in T[i].points) or (p02 in T[i].points) or (p03 in T[i].points):
            for elem in T[i].points:
                elem.triangles.remove(T[i])
            for elem in T[i].edges:
                elem.triangles.remove(T[i])
                elem.enclosed=False
            del T[i]
    
    for i in range(len(E)-1,-1,-1):
        if (p01 in E[i].points) or (p02 in E[i].points) or (p03 in E[i].points):
            for elem in E[i].points:
                elem.edges.remove(E[i])
            del E[i]
    
    del P[0]
    del P[0]
    del P[0]
    
    if constraints is not None:
        #constraintIndex=[]
        if isinstance(constraints,list) or isinstance(constraints,tuple) or isinstance(constraints,np.ndarray):
            if all([isinstance(constr,edge) for constr in constraints]):
                for i in range(len(constraints)-1,-1,-1):
                    matchbool=False
                    for j in range(len(E)):
                        if constraints[i].is_edge(E[j]):
                            matchbool=True
                            E[j].constraint=True
                            #constraintIndex.append(j)
                            break
                    if matchbool:
                        #print("Constraint "+constraints[i].to_string()+" is already included in the triangulation")
                        del constraints[i]
                for i in range(len(constraints)-1):
                    for j in range(len(constraints)-1,i,-1):
                        if constraints[i].is_edge(constraints[j]):
                            print("Constraint "+constraints[j].to_string()+" is already included in constraints set")
                            del constraints[j]
                        elif constraints[i].edge_edge_intersect(constraints[j],includeboundary=False):
                            print("Constraint "+constraints[j].to_string()+" intersects with constraint "+constraints[i].to_string()+"; the former has been deleted")
                            del constraints[j]
                for i in range(len(constraints)-1,-1,-1):
                    p1bool=False
                    p2bool=False
                    for p in P:
                        if constraints[i].points[0].is_point(p):
                            p1bool=True
                        elif constraints[i].points[1].is_point(p):
                            p2bool=True
                        if p1bool and p2bool:
                            break
                    if not p1bool or not p2bool:
                        print("One of the points in Constraint "+constraints[i].to_string()+" is not included in the triangulation")
                        del constraints[i]
                
                for constr in constraints:
                    constr.constraint=True
                
                #_constrained_(P,E,T,constraints,constraintIndex)
                _constrained_(P,E,T,constraints)
    return P,E,T
                
def killvirus(point,P,E,T):
    infectedT=[]
    for t in T:
        if t.point_triangle_intersect(point):
           infectedT=[t]
           break
    while len(infectedT)>0:
        newInfectedT=[]
        for i in range(len(infectedT)-1,-1,-1):
            infectedP=[p for p in infectedT[i].points]
            infectedE=[e for e in infectedT[i].edges]
            for e in infectedT[i].edges:
                if not e.constraint:
                    if len(e.triangles)>1:
                        if not e.triangles[0].is_triangle(infectedT[i]):
                            if e.triangles[0] not in newInfectedT:
                                newInfectedT.append(e.triangles[0])
                        elif not e.triangles[1].is_triangle(infectedT[i]):
                            if e.triangles[0] not in newInfectedT:
                                newInfectedT.append(e.triangles[1])
            infectedT[i].kill()
            for j in range(len(infectedE)-1,-1,-1):
                if len(infectedE[j].triangles)==0:
                    if not infectedE[j].constraint:
                        infectedE[j].kill()
                    else:
                        infectedE[j].enclosed=False
                else:
                    infectedE[j].enclosed=False
            for j in range(len(infectedP)-1,-1,-1):
                if len(infectedP[j].triangles)==0 and len(infectedP[j].edges)==0:
                    infectedP[j].kill()
                
def delaunay_refine(P,E,T,theta=20,areatol=None,errtol=1e-12):
    theta*=math.pi/180
    count=0
    while theta<0 and count<100:
        theta+=2*math.pi
        count+=1
    if count>=100:
        print("Parameter theta too small. Resorting to theta=20 degrees")
        theta=math.pi/9
    count=0
    while theta>2*math.pi and count<100:
        theta-=2*math.pi
        count+=1
    if count>=100:
        print("Parameter theta too large. Resorting to theta=20 degrees")
        theta=math.pi/9
        
    center=[]
    for p in P:
        constraintTotal=sum([1 for e in p.edges if e.constraint])
        if constraintTotal>1:
            cascadebool=False
            minEdgeLength=p.edges[0].length()
            minAbsCosTheta=1
            for i in range(len(p.edges)-1):
                if p.edges[i].constraint:
                    for j in range(1,len(p.edges)):
                        if p.edges[j].constraint:
                            a=max(p.edges[i].length(),p.edges[j].length())
                            b=min(p.edges[i].length(),p.edges[j].length())
                            if b<minEdgeLength:
                                minEdgeLength=b
                            if p.edges[i].points[0].is_point(p) and p.edges[j].points[0].is_point(p):
                                c2=(p.edges[i].points[1].x-p.edges[j].points[1].x)**2+(p.edges[i].points[1].y-p.edges[j].points[1].y)**2
                            elif p.edges[i].points[0].is_point(p) and p.edges[j].points[1].is_point(p):
                                c2=(p.edges[i].points[1].x-p.edges[j].points[0].x)**2+(p.edges[i].points[1].y-p.edges[j].points[0].y)**2
                            elif p.edges[i].points[1].is_point(p) and p.edges[j].points[0].is_point(p):
                                c2=(p.edges[i].points[0].x-p.edges[j].points[1].x)**2+(p.edges[i].points[0].y-p.edges[j].points[1].y)**2
                            else:
                                c2=(p.edges[i].points[0].x-p.edges[j].points[0].x)**2+(p.edges[i].points[0].y-p.edges[j].points[0].y)**2
                            costheta=(a**2+b**2-c2)/(2*a*b)
                            if abs(costheta)<minAbsCosTheta:
                                minAbsCosTheta=abs(costheta)
                            if costheta>abs(1e-12):
                                n=math.floor(math.log(a,2)+math.log(costheta,2)-math.log(b,2))
                                if b*2**n<a*costheta-abs(errtol) and a<b*costheta*2**(n+1)-abs(errtol):
                                    cascadebool=True
                if cascadebool:
                    break
            if cascadebool:
                sintheta=math.sqrt(1-minAbsCosTheta**2)
                if areatol is not None and abs(areatol)<minEdgeLength**2*sintheta/2-abs(errtol):
                    minEdgeLength=math.sqrt(2*abs(areatol)/sintheta)
                for e in p.edges:
                    if e.constraint:
                        if e.points[0].is_point(p):
                            center.append(point(p.x+0.5*(e.points[1].x-p.x)*minEdgeLength/e.length(),
                                                 p.y+0.5*(e.points[1].y-p.y)*minEdgeLength/e.length()))
                        else:
                            center.append(point(p.x+0.5*(e.points[0].x-p.x)*minEdgeLength/e.length(),
                                                 p.y+0.5*(e.points[0].y-p.y)*minEdgeLength/e.length()))
    for c in center:
        _insert_Vertex_Delaunay_(c,P,E,T)
    
    donebool=False
    count=0
    while not donebool and count<500:
        donebool=True
        for e in E:
            if e.constraint:
                encroachedbool=False
                for elem in e.points[0].edges:
                    if (elem.points[0].is_point(e.points[0]) and e.inCircumcircle(elem.points[1],includeboundary=False)) or (elem.points[1].is_point(e.points[0]) and e.inCircumcircle(elem.points[0],includeboundary=False)):
                        center=point((e.points[1].x+e.points[0].x)/2,(e.points[1].y+e.points[0].y)/2)
                        _insert_Vertex_Delaunay_(center,P,E,T)
                        encroachedbool=True
                        donebool=False
                        break
                if not encroachedbool:
                    for elem in e.points[1].edges:
                        if (elem.points[0].is_point(e.points[1]) and e.inCircumcircle(elem.points[1],includeboundary=False)) or (elem.points[1].is_point(e.points[1]) and e.inCircumcircle(elem.points[0],includeboundary=False)):
                            center=point((e.points[1].x+e.points[0].x)/2,(e.points[1].y+e.points[0].y)/2)
                            _insert_Vertex_Delaunay_(center,P,E,T)
                            encroachedbool=True
                            donebool=False
                            break
            if not donebool:
                break
        count+=1
    
    donebool=False
    count=0
    while not donebool and count<500:
        donebool=True
        for t in T:
            a=t.edges[0].length()
            b=t.edges[1].length()
            c=t.edges[2].length()
            minLengthIndex=0
            if a<=b and a<=c:
                minAngle=math.acos((b**2+c**2-a**2)/(2*b*c))
            elif b<=a and b<=c:
                minLengthIndex=1
                minAngle=math.acos((a**2+c**2-b**2)/(2*a*c))
            else:
                minLengthIndex=2
                minAngle=math.acos((a**2+b**2-c**2)/(2*a*b))
            if minAngle<theta-abs(errtol) or (areatol is not None and t.area()>abs(areatol)+abs(errtol)):
                if sum([1 for e in t.edges if e.constraint])<2:
                    tcircle=t.circumcircle()
                    center=tcircle.center
                    if 2*math.asin(t.edges[minLengthIndex].length()/(2*tcircle.radius))<theta-abs(errtol):
                        edgeCenter=point((t.edges[minLengthIndex].points[0].x+t.edges[minLengthIndex].points[1].x)/2,
                                         (t.edges[minLengthIndex].points[0].y+t.edges[minLengthIndex].points[1].y)/2)
                        dist=center.distance(edgeCenter)
                        distprime=t.edges[minLengthIndex].length()/(2*math.tan(theta/2))
                        center.x=edgeCenter.x+0.99*(center.x-edgeCenter.x)*distprime/dist
                        center.y=edgeCenter.y+0.99*(center.y-edgeCenter.y)*distprime/dist
                    encroachedbool=False
                    for e in E:
                        if e.constraint and e.inCircumcircle(center,includeboundary=False):
                            _insert_Vertex_Delaunay_(point((e.points[1].x+e.points[0].x)/2,(e.points[1].y+e.points[0].y)/2),P,E,T)
                            encroachedbool=True
                            break
                    if not encroachedbool:
                        _insert_Vertex_Delaunay_(center,P,E,T)
                    donebool=False
            if not donebool:
                break
        count+=1
