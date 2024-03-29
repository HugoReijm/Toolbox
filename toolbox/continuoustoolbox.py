import math
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse as sparsepy
import scipy.sparse.linalg as sparsela
from itertools import product
import toolbox.matrixtoolbox as mtb
import toolbox.generaltoolbox as gtb
import toolbox.plottoolbox as ptb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def poincareSection(sol,planeCoords,planeParams):
    #This method calculates a Poincare Section of a solution of a system of
    #differential equations onto a plane discribed by 
    #sum([planeParams[i]*(x_i-planeCoords[i]) for i in range(dim)])=0. Including 
    #time is not necessary here. Notice that the inputted solution must as 
    #organized first by dimension, then by chronology.
    #That is, sol=[[x_1,x_2,...x_n],[y_1,y_2,...y_n],...]
    if isinstance(planeCoords,list):
        planeCoords=np.array(planeCoords)
    if isinstance(planeParams,list):
        planeParams=np.array(planeParams)
    dim=len(sol)
    planeParams=planeParams/npla.norm(planeParams)
    poincareCoords=[[] for i in range(dim)]
    for i in range(sol[0].shape[0]):
        coords1=np.array([s[i] for s in sol])
        res1=planeParams.dot(coords1-planeCoords)
        if res1==0:
            for l in range(dim):
                poincareCoords[l].append(coords1[l])
        else:
            if i!=sol[0].shape[0]-1:
                coords2=np.array([s[i+1] for s in sol])
                res2=planeParams.dot(coords2-planeCoords)
                if res1*res2<0:
                    for l in range(dim):
                        poincareCoords[l].append((res2*coords1[l]-res1*coords2[l])/(res2-res1))
    return poincareCoords
    
def lyapunovSpectrum(sol,t,f,args=[],kwargs={},dist=1e-6,K=1,easy=True,plotbool=False,plotaxis=None,savefigName=None):
    #This method approximates the first K values of the Lyapunov Spectrum of a 
    #solution of a system of differential equations. Variable sol is the inputted
    #solution (organized first by dimension, then by chronology), variable t is
    #the corresponding time array, and variable f is the corresponding system of
    #differential equations. Variable dist is the distance used between the 
    #initial conditions for each trajectory in each iteration. Variable easy 
    #controls whether the RK4 numerical integration method (easy=True) or the 
    #RK14(12) numerical integration method (easy=False) is used. Infrastructure
    #for plotting the convergence of each exponent is available (for visual 
    #verification). Variable savefigName allows for tge resulting 
    #graph to be plotted under the name <savefigName>.png.
    dim=len(sol)
    solcopy=[sol[i][:-1] for i in range(dim)]
    tcopy=t[:-1]
    n=len(tcopy)
    if any([len(solcopy[i])!=n for i in range(dim)]):
        print("Error in inputted solution. Terminating process")
        return 0.0
    
    K=max(min(K,dim),1)
    vect0=np.array([float(solcopy[i][0]) for i in range(dim)])
    vects=[np.copy(vect0) for i in range(K)]
    for i in range(K):
        vects[i][i]+=dist
    
    def GramDet(vectors):
        size=len(vectors)
        Gramian=[[np.inner(vectors[i],vectors[j]) for j in range(size)] for i in range(size)]
        return npla.det(Gramian)
    
    Lambda=[0.0 for i in range(K)]
    if plotbool:
        plotLambda=[[] for i in range(K)]
    counts=[0 for i in range(K)]
    
    vols=[0.0 for i in range(K)]
    for i in range(1,n):
        vect0=np.array([float(solcopy[j][i]) for j in range(dim)])
        
        for j in range(K):
            if easy:
                tempsol,tempt=rk4(f,vects[j],vects[j]-np.array([1e3 for i in range(dim)]),
                                  vects[j]+np.array([1e3 for i in range(dim)]),tcopy[i-1],tcopy[i],tcopy[i]-tcopy[i-1],
                                  args=args,kwargs=kwargs,adapt=False,inform=False)
            else:
                tempsol,tempt=rk14(f,vects[j],vects[j]-np.array([1e3 for i in range(dim)]),
                                   vects[j]+np.array([1e3 for i in range(dim)]),tcopy[i-1],tcopy[i],tcopy[i]-tcopy[i-1],
                                   args=args,kwargs=kwargs,adapt=False,inform=False)
            vects[j]=np.array([tempsol[l][len(tempt)-1] for l in range(dim)])
            vols[j]=math.sqrt(GramDet([vects[l]-vect0 for l in range(j+1)]))

        for j in range(K):
            if vols[j]!=0:
                counts[j]+=1
                Lambda[j]+=math.log(vols[j])-(j+1)*math.log(dist)
                if plotbool:
                    if j==0:
                        plotLambda[j].append(Lambda[j]/(tcopy[i]-tcopy[0]))
                    else:
                        if counts[j-1]==0:
                            plotLambda[j].append(Lambda[j]/(tcopy[i]-tcopy[0]))
                        else:
                            plotLambda[j].append(Lambda[j]/(tcopy[i]-tcopy[0])-Lambda[j-1]/(tcopy[i]-tcopy[0]))
            else:
                if plotbool:
                    counts[j]+=1
                    plotLambda[j].append(plotLambda[j][len(plotLambda[j])-1])
                    
        orthonormals=mtb.grammSchmidt([vects[j]-vect0 for j in range(K)],normalize=True,tol=min(dist*1e-3,1e-3),clean=False)
        adjusted=[]
        voided=[]
        for j in range(K):
            if any([elem!=0.0 for elem in orthonormals[j]]):
                adjusted.append(j)
                vects[j]=dist*orthonormals[j]+vect0
            else:
                voided.append(j)
        if len(voided)>0:
            if len(adjusted)>0:
                nullspace=np.transpose(spla.null_space(np.array([vects[a] for a in adjusted])))
                for j in range(len(voided)):
                    vects[voided[j]]=dist*nullspace[j]/npla.norm(nullspace[j])+vect0
            else:
                for j in range(K):
                    vects[j]=np.array([0 for k in range(j)]+[dist]+[0 for k in range(j+1,dim)])+vect0
    
    lmbda=[]
    if counts[0]!=0:
        lmbda.append(Lambda[0]/(tcopy[-1]-tcopy[0]))
    for i in range(1,K):
        if counts[i]!=0:
            if counts[i-1]==0:
                lmbda.append(Lambda[i]/(tcopy[-1]-tcopy[0]))
            else:
                lmbda.append(Lambda[i]/(tcopy[-1]-tcopy[0])-Lambda[i-1]/(tcopy[-1]-tcopy[0]))
    
    if plotbool:
        showbool=False
        if plotaxis is None:
            graphsize=9
            font = {"family": "serif",
                "color": "black",
                "weight": "bold",
                "size": "20"}
            plotfig=plt.figure(figsize=(graphsize,graphsize))
            plotaxis=plotfig.add_subplot(111)
            if K==1:
                plotaxis.set_title("Maximal Lyapunov Exponent Convergence",fontdict=font)
                plotaxis.set_xlabel("Iterations",fontsize=16,rotation=0)
                plotaxis.set_ylabel("$\\mathbf{\\lambda_{max}}$",fontsize=16,rotation=0)
            else:
                plotaxis.set_title("Lyapunov Spectrum Convergence",fontdict=font)
                plotaxis.set_xlabel("Iterations",fontsize=16,rotation=0)
                plotaxis.set_ylabel("$\\mathbf{\\lambda 's}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            showbool=True
        colors=ptb.colors(K)
        for i in range(K):
            plotaxis.plot([j for j in range(counts[i])],plotLambda[i],color=colors[i])
        if showbool:
            if savefigName is not None and isinstance(savefigName,str):
                plt.savefig(savefigName+".png",bbox_inches="tight")
                plt.close()
            else:
                plt.show()
    return lmbda
    
def scm(f,start,stop,delta,args=[],kwargs={},plotbool=True,plotaxis=None,color="black",alpha=1.0):
    #This method provides a simple cell mapping analysis to determine all 
    #persistent cells and all transient cells of the phase space between lower
    #bound vector start and upper bound vector stop, where each cell has dimensions
    #vector delta. Infrastructure for plotting the cells is available. Notice 
    #that this method can only be applied to 1D, 2D, and 3D systems of differential
    #equations.
    if len(f(start,*args,**kwargs))==len(start) and len(start)==len(stop) and len(stop)==len(delta):
        dim=len(start)
        if 0<dim<=3:
            N=[int(round((stop[i]-start[i])/delta[i])) for i in range(dim)]
            plotshowbool=False
            if plotbool and plotaxis is None:
                graphsize=9
                font = {"family": "serif",
                    "color": "black",
                    "weight": "bold",
                    "size": "20"}
                plotfig=plt.figure(figsize=(graphsize,graphsize))
                if dim<=2:
                    plotaxis=plotfig.add_subplot(111)
                    plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
                    if dim==2:
                        plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
                else:
                    plotaxis=Axes3D(plotfig)
                    plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
                    plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
                    plotaxis.set_zlabel("$\\mathbf{Z}$",fontsize=16,rotation=0)
                plotaxis.set_title("Approximate System Behavior",fontdict=font)
                plotaxis.set_xlim([start[0],stop[0]])
                plotaxis.xaxis.set_tick_params(labelsize=16)
                if dim>=2:
                    plotaxis.set_ylim([start[1],stop[1]])
                    plotaxis.yaxis.set_tick_params(labelsize=16)
                if dim==3:
                    plotaxis.set_zlim([start[2],stop[2]])
                    plotaxis.zaxis.set_tick_params(labelsize=16)
                plotshowbool=True
            
            class cell:
                def __init__(self,v,index):
                    if not isinstance(v,np.ndarray):
                        v=np.array(v)
                    self.v=v
                    self.index=index
                    self.center=self.v+np.array(delta)/2
                    
                def trajectory(self,C):
                    self.trajIndex=cellIndex(f(self.center,*args,**kwargs),C)
                
            grids=[[start[i]+j*delta[i] for j in range(N[i])] for i in range(dim)]
            C=[]
            size=0
            for g in product(*grids):
                C.append(cell(g,size))
                size+=1
                
            def cellIndex(point,C):
                if all([start[i]<=point[i]<=stop[i] for i in range(dim)]):
                    res=[math.floor((point[i]-start[i])/delta[i]) for i in range(dim)]
                    for c in C[sum([res[i]*np.product(N[i+1:]) for i in range(dim-1)])+res[dim-1]:sum([res[i]*np.product(N[i+1:]) for i in range(dim-1)])+res[dim-1]+dim]:
                        if all([c.v[i]<=point[i]<=c.v[i]+delta[i] for i in range(dim)]):
                            return c.index
                    for c in C:
                        if all([c.v[i]<=point[i]<=c.v[i]+delta[i] for i in range(dim)]):
                            return c.index
                else:
                    return len(C)
                
            for c in C:
                c.trajectory(C)
                
            persistentC=[]
            transientC=[]
            sinkC=[]
            for c in C:
                if (c.index not in persistentC) and (c.index not in transientC) and (c.index not in sinkC):
                    linkIndex=c.index
                    linkC=[]
                    while (linkIndex!=size) and (linkIndex not in linkC) and (linkIndex not in persistentC) and (linkIndex not in transientC) and (linkIndex not in sinkC):
                        linkC.append(linkIndex)
                        linkIndex=C[linkIndex].trajIndex
                    if (linkIndex==size) or (linkIndex in sinkC):
                        sinkC+=linkC
                    elif (linkIndex in transientC) or (linkIndex in persistentC):
                        transientC+=linkC    
                    else:
                        for i in range(len(linkC)-1,-1,-1):
                            if linkC[i]==linkIndex:
                                transientC+=linkC[:i]
                                persistentC+=linkC[i:]
                                break
                            
            if plotbool:
                if dim==1:                    
                    plotaxis.scatter([C[pc].center[0] for pc in persistentC],[0 for pc in persistentC],color=color,alpha=alpha)
                    plotaxis.scatter([C[tc].center[0] for tc in transientC],[0 for tc in transientC],s=min([10*max(delta),10]),color=color,alpha=alpha)
                elif dim==2:
                    plotaxis.scatter([C[pc].center[0] for pc in persistentC],[C[pc].center[1] for pc in persistentC],color=color,alpha=alpha)
                    plotaxis.scatter([C[tc].center[0] for tc in transientC],[C[tc].center[1] for tc in transientC],s=min([10*max(delta),10]),color=color,alpha=alpha)
                else:
                    plotaxis.scatter([C[pc].center[0] for pc in persistentC],[C[pc].center[1] for pc in persistentC],[C[pc].center[2] for pc in persistentC],color=color,alpha=alpha)
                    plotaxis.scatter([C[tc].center[0] for tc in transientC],[C[tc].center[1] for tc in transientC],[C[tc].center[2] for tc in transientC],s=min([10*max(delta),10]),color=color,alpha=alpha)
                if plotshowbool:
                    plt.show()
            else:
                return [C[pc].center for pc in persistentC], [C[tc].center for tc in transientC]
        else:
            print("Unable to plot that number of dimensions; restrict the dimension of the phase space")
    else:
        print("Error in Cell Mapping: check the length of your inputted vectors")
        print("Length of output of f: %s"%len(f(start,*args,**kwargs)))
        print("Length of start: %i"%len(start))
        print("Length of stop: %i"%len(stop))
        print("Length of delta: %i"%len(delta))

def gcm(f,start,stop,delta,args=[],kwargs={},tol=1e-6,plotbool=True,plotaxis=None,color="black",alpha=1.0):
    #This method provides a generalized cell mapping analysis to determine all 
    #persistent, transient, and border cells of the phase space between lower
    #bound vector start and upper bound vector stop, where each cell has dimensions
    #vector delta. The variable tol controls the tolerance of what is to be 
    #considered zero or not (useful when applying linear algebraic methods)
    #Infrastructure for plotting the cells is available. Notice that this method
    #can only be applied to 1D, 2D, and 3D systems of differential equations.
    if len(f(start,*args,**kwargs))==len(start) and len(start)==len(stop) and len(stop)==len(delta):
        dim=len(start)
        if 0<dim<=3:
            N=[int(round((stop[i]-start[i])/delta[i])) for i in range(dim)]
            plotshowbool=False
            if plotbool and plotaxis is None:
                graphsize=9
                font = {"family": "serif",
                    "color": "black",
                    "weight": "bold",
                    "size": "20"}
                plotfig=plt.figure(figsize=(graphsize,graphsize))
                if dim<=2:
                    plotaxis=plotfig.add_subplot(111)
                    plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
                    if dim==2:
                        plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
                else:
                    plotaxis=Axes3D(plotfig)
                    plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
                    plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
                    plotaxis.set_zlabel("$\\mathbf{Z}$",fontsize=16,rotation=0)
                plotaxis.set_title("Approximate System Behavior",fontdict=font)
                plotaxis.set_xlim([start[0],stop[0]])
                plotaxis.xaxis.set_tick_params(labelsize=16)
                plotaxis.yaxis.set_tick_params(labelsize=16)
                if dim>=2:
                    plotaxis.set_ylim([start[1],stop[1]])
                if dim==3:
                    plotaxis.set_zlim([start[2],stop[2]])
                    plotaxis.zaxis.set_tick_params(labelsize=16)
                plotshowbool=True
            
            class cell:
                def __init__(self,v,index):
                    if not isinstance(v,np.ndarray):
                        v=np.array(v)
                    self.v=v
                    self.index=index
                    self.center=self.v+np.array(delta)/2
                    
                def probs(self,C,data,rows,cols,count=3):
                    if count==1:
                        grids=[[self.v[i]+delta[i]/2] for i in range(dim)]
                    else:
                        grids=[[self.v[i]+j*delta[i]/(count-1) for j in range(count)] for i in range(dim)]
                    self.probs=[]
                    for g in product(*grids):
                        res=cellIndex(f(g,*args,**kwargs),C)
                        uniquebool=True
                        for i in range(len(self.probs)):
                            if res==self.probs[i][0]:
                                uniquebool=False
                                self.probs[i][1]+=count**(-dim)
                                break
                        if uniquebool:
                            self.probs.append([res,count**(-dim)])
                    data+=[p[1] for p in self.probs]
                    rows+=[self.index for p in self.probs]
                    cols+=[p[0] for p in self.probs]
                    
            grids=[[start[i]+j*delta[i] for j in range(N[i])] for i in range(dim)]
            C=[]
            size=0
            for g in product(*grids):
                C.append(cell(g,size))
                size+=1
                
            def cellIndex(point,C):
                if all([start[i]<=point[i]<=stop[i] for i in range(dim)]):
                    res=[math.floor((point[i]-start[i])/delta[i]) for i in range(dim)]
                    for c in C[sum([res[i]*np.product(N[i+1:]) for i in range(dim-1)])+res[dim-1]:sum([res[i]*np.product(N[i+1:]) for i in range(dim-1)])+res[dim-1]+dim]:
                        if all([c.v[i]<=point[i]<=c.v[i]+delta[i] for i in range(dim)]):
                            return c.index
                    for c in C:
                        if all([c.v[i]<=point[i]<=c.v[i]+delta[i] for i in range(dim)]):
                            return c.index
                else:
                    return len(C)
                
            data,rows,cols=[1],[size],[size]
            for c in C:
                c.probs(C,data,rows,cols)
            P=sparsepy.coo_matrix((data,(rows,cols)),shape=(size+1,size+1)).tolil()-sparsepy.identity(size+1)
            
            u,s,vt=sparsela.svds(P.transpose(),k=size,tol=tol)
            if s[len(s)-1]==0:
                pivot=len(s)
                for i in range(len(s)-2,-1,-1):
                    if abs(s[i])>tol:
                        pivot=i+1
                        break
                Sol=vt[pivot:].tolist()
            else:
                Sol=[]
            for sol in Sol:
                sol[size]=0.0
            sol=np.zeros(size+1)
            sol[size]=1.0
            Sol.append(sol)

            persistentGroups=[[i for i in range(len(sol)) if abs(sol[i])>tol] for sol in Sol]
            if plotbool:
                for i in range(len(persistentGroups)-1):
                    if dim==1:                    
                        plotaxis.scatter([C[pc].center[0] for pc in persistentGroups[i]],
                                         [0 for pc in persistentGroups[i]],color=color,alpha=alpha)
                    elif dim==2:
                        plotaxis.scatter([C[pc].center[0] for pc in persistentGroups[i]],
                                         [C[pc].center[1] for pc in persistentGroups[i]],color=color,alpha=alpha)
                    else:
                        plotaxis.scatter([C[pc].center[0] for pc in persistentGroups[i]],
                                         [C[pc].center[1] for pc in persistentGroups[i]],
                                         [C[pc].center[2] for pc in persistentGroups[i]],color=color,alpha=alpha)
                        
            transientC=[i for i in range(size) if i not in sum(persistentGroups,[])]
            if len(transientC)>0:
                T=P[np.ix_(transientC,transientC)]
                B=[-np.array([sum([P[ti,elem] for elem in pgroup]) for ti in transientC]) for pgroup in persistentGroups]
                Absorb=[sparsela.spsolve(T,b) for b in B]
                
                transientGroups=[[transientC[j] for j in range(len(transientC)) if abs(Absorb[i][j])>0.5-tol] for i in range(len(persistentGroups))]
                borderGroup=[i for i in transientC if i not in sum(transientGroups,[])]
            
                if plotbool:
                    for i in range(len(transientGroups)-1):
                        if dim==1:                    
                            plotaxis.scatter([C[tc].center[0] for tc in transientGroups[i]],
                                             [0 for pc in transientGroups[i]],s=min([10*max(delta),10]),color=color,alpha=alpha)
                        elif dim==2:
                            plotaxis.scatter([C[tc].center[0] for tc in transientGroups[i]],
                                             [C[tc].center[1] for tc in transientGroups[i]],s=min([10*max(delta),10]),color=color,alpha=alpha)
                        else:
                            plotaxis.scatter([C[tc].center[0] for tc in transientGroups[i]],
                                             [C[tc].center[1] for tc in transientGroups[i]],
                                             [C[tc].center[2] for tc in transientGroups[i]],s=min([10*max(delta),10]),color=color,alpha=alpha)
                    if dim==1:
                        plotaxis.scatter([C[bc].center[0] for bc in borderGroup],[0 for bc in borderGroup],facecolor="none",edgecolor=color,linewidth=2,alpha=alpha)
                    elif dim==2:
                        plotaxis.scatter([C[bc].center[0] for bc in borderGroup],[C[bc].center[1] for bc in borderGroup],facecolor="none",edgecolor=color,linewidth=2,alpha=alpha)
                    else:
                        plotaxis.scatter([C[bc].center[0] for bc in borderGroup],[C[bc].center[1] for bc in borderGroup],[C[bc].center[2] for bc in borderGroup],facecolor="none",edgecolor=color,linewidth=2,alpha=alpha)
                    if plotshowbool:
                            plt.show()
                            
            if not plotbool:
                if len(transientC)>0:
                    return [[C[pc].center for pc in pgroup] for pgroup in persistentGroups[:-1]],[[C[tc].center for tc in tgroup] for tgroup in transientGroups[:-1]],[C[bc].center for bc in borderGroup]
                else:
                    return [[C[pc].center for pc in pgroup] for pgroup in persistentGroups[:-1]]
        else:
            print("Unable to plot that number of dimensions; restrict the dimension of the phase space")
    else:
        print("Error in Cell Mapping: check the length of your inputted vectors")
        print("Length of output of f: %s"%len(f(start,*args,**kwargs)))
        print("Length of start: %i"%len(start))
        print("Length of stop: %i"%len(stop))
        print("Length of delta: %i"%len(delta))

def manifold(f,point,start,stop,dist,maxlevel,args=[],kwargs={},stable=False,easy=False,plotaxis=None,colormap=None,savefigName=None,color="black",alpha=1.0):
    #This method approximates the manifolds of an equilibrium point of a system
    #of differential equations between lower bound vector start and upper bound
    #vector stop. Variable f represents the system, while variable point is the 
    #equilibrium point. The method iteratively adds a front to the manifold, and
    #variable dist determines how far the new front is from the old front. Variable
    #stable determines whether the manifold is stable (stable=True) or unstable
    #(stable=False). Variable easy controls whether the RK4 numerical
    #integration method (easy=True) or the the RK14(12) numerical integration 
    #method (easy=False) is used. Infrastructure for plotting the manifold 
    #is available. A colormap option is available to when plotting 
    #(colormap values are based on the z-coordinate).
    dim=len(point)
    if not isinstance(point,np.ndarray):
        point=np.array(point)
    plotshowbool=False
    maxlevel=max(round(maxlevel),0)
    if plotaxis is None:
        graphsize=9
        font = {"family": "serif",
            "color": "black",
            "weight": "bold",
            "size": "20"}
        labelfont = {"family": "serif",
                "color": "black",
                "weight": "bold",
                "size": "16"}
        plotfig=plt.figure(figsize=(graphsize,graphsize))
        if dim==1:
            plotaxis=plotfig.add_subplot(111)
            if stable:
                plotaxis.set_title("Stable Manifold of (%0.2f)"%(point[0]),fontdict=font)
            else:
                plotaxis.set_title("Unstable Manifold of (%0.2f)"%(point[0]),fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$Iteration$",fontdict=labelfont,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotshowbool=True
        elif dim==2:
            plotaxis=plotfig.add_subplot(111)
            if stable:
                plotaxis.set_title("Stable Manifold of (%0.2f,%0.2f)"%(point[0],point[1]),fontdict=font)
            else:
                plotaxis.set_title("Unstable Manifold of (%0.2f,%0.2f)"%(point[0],point[1]),fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotshowbool=True
        elif dim==3:
            plotaxis=Axes3D(plotfig)
            if stable:
                plotaxis.set_title("Stable Manifold of (%0.2f,%0.2f,%0.2f)"%(point[0],point[1],point[2]),fontdict=font)
            else:
                plotaxis.set_title("Unstable Manifold of (%0.2f,%0.2f,%0.2f)"%(point[0],point[1],point[2]),fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
            plotaxis.set_zlabel("$\\mathbf{Z}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotaxis.zaxis.set_tick_params(labelsize=16)
            plotshowbool=True
        
    Lin=mtb.jacobian(f,point)
    eigval,eigvect=npla.eig(Lin)
    eigvect=eigvect.transpose()
    
    if not stable:
        seed=[eigvect[i].real for i in range(len(eigval)) if eigval[i].real>0]
        seedDim=len(seed)
    else:
        seed=[eigvect[i].real for i in range(len(eigval)) if eigval[i].real<0]
        seedDim=len(seed)
        
    for i in range(len(seed)):
        seed[i]=max(1e-4*dist,1e-4)*seed[i]/npla.norm(seed[i])
    
    filtered=False
    while not filtered and len(seed)>0:
        filtered=True
        for i in range(len(seed)):
            for j in range(i+1,len(seed)):
                if np.array_equal(seed[i],seed[j]):
                    del seed[j]
                    filtered=False
                    break
            if not filtered:
                break
    
    def completeSeed(seed,seedDim,thetalimit):
        if len(seed)<dim:
            if dim==2:
                if thetalimit==math.pi/2:
                    theta=np.array([math.pi/2,3*math.pi/2])
                else:
                    theta=gtb.hammersley(50,1,points=False)[0]
                    for i in range(len(theta)):
                        if theta[i]<=0.5:
                            theta[i]=2*(math.pi-2*thetalimit)*theta[i]+thetalimit
                        else:
                            theta[i]=2*(math.pi-2*thetalimit)*(theta[i]-0.5)+thetalimit+math.pi
                radius=npla.norm(seed[0])
                thetaShift=math.atan2(seed[0][1],seed[0][0])
                best=seed[0]
                X=point[0]+radius*np.cos(theta)*math.cos(thetaShift)-radius*np.sin(theta)*math.sin(thetaShift)
                Y=point[1]+radius*np.cos(theta)*math.sin(thetaShift)+radius*np.sin(theta)*math.cos(thetaShift)
                bestcorrelation=0
                for i in range(len(X)):
                    flow=f([X[i],Y[i]],*args,**kwargs)
                    correlation=(X[i]-point[0])*flow[0]+(Y[i]-point[1])*flow[1]
                    if stable:
                        if correlation<bestcorrelation:
                            best=np.array([X[i],Y[i]])
                            bestcorrelation=correlation
                    else:
                        if correlation>bestcorrelation:
                            best=np.array([X[i],Y[i]])
                            bestcorrelation=correlation
                if bestcorrelation!=0:
                    seed.append(best-point)
            if dim==3:
                theta,phi=gtb.hammersley(500,2,points=False)
                if len(seed)==1:
                    radius=npla.norm(seed[0])
                    theta=(math.pi-2*thetalimit)*theta+thetalimit
                    thetaShift=math.acos(seed[0][2]/npla.norm(seed[0]))
                    phiShift=math.atan2(seed[0][1],seed[0][0])
                    best=seed[0]
                elif len(seed)==2:
                    radius=npla.norm(seed[0])
                    for i in range(len(theta)):
                        if theta[i]<=0.5:
                            theta[i]=2*(math.pi/2-thetalimit)*theta[i]
                        elif theta[i]>0.5:
                            theta[i]=2*(math.pi/2-thetalimit)*theta[i]+math.pi/2+thetalimit
                    res=np.cross(seed[0],seed[1])
                    thetaShift=math.acos(res[2]/npla.norm(res))
                    phiShift=math.atan2(res[1],res[0])
                    best=res
                    
                phi=2*math.pi*phi
                X=point[0]+radius*np.sin(theta)*np.cos(phi)*math.cos(thetaShift)*math.cos(phiShift)-radius*np.sin(theta)*np.sin(phi)*math.sin(phiShift)+radius*np.cos(theta)*math.sin(thetaShift)*math.cos(phiShift)
                Y=point[1]+radius*np.sin(theta)*np.cos(phi)*math.cos(thetaShift)*math.sin(phiShift)+radius*np.sin(theta)*np.sin(phi)*math.cos(phiShift)+radius*np.cos(theta)*math.sin(thetaShift)*math.sin(phiShift)
                Z=point[2]-radius*np.sin(theta)*np.cos(phi)*math.sin(thetaShift)+radius*np.cos(theta)*math.cos(thetaShift)
                
                bestcorrelation=0
                for i in range(len(X)):
                    flow=f([X[i],Y[i],Z[i]],*args,**kwargs)
                    correlation=(X[i]-point[0])*flow[0]+(Y[i]-point[1])*flow[1]+(Z[i]-point[2])*flow[2]
                    if stable:
                        if correlation<bestcorrelation:
                            best=np.array([X[i],Y[i],Z[i]])
                            bestcorrelation=correlation
                    else:
                        if correlation>bestcorrelation:
                            best=np.array([X[i],Y[i],Z[i]])
                            bestcorrelation=correlation
                if bestcorrelation!=0:
                    seed.append(best-point)
    
    count=0
    countlimit=20
    while (seedDim==2 and len(seed)==1) and count<=countlimit:
        completeSeed(seed,2,math.pi/(count+2))
        count+=1
    count=0
    while (seedDim==3 and len(seed)==1) and count<=countlimit:
        completeSeed(seed,3,math.pi/(count+2))
        count+=1
    count=0
    while (seedDim==3 and len(seed)==2) and count<=countlimit:
        completeSeed(seed,3,math.pi/(count+2))
        count+=1
    
    class vertex:
        def __init__(self,coord):
            if not isinstance(coord,np.ndarray):
                coord=np.array(coord)
            self.coord=coord
            self.traj=None
            self.staticbool=False
            self.expandbool=True
            
        def trajectory(self):
            if self.traj is None:
                d=abs(dist)
                if stable:
                    d*=-1.0 
                zero=1e-10
                
                def step(vect,d):
                    flow=np.array(f(vect.tolist(),*args,**kwargs))
                    fnorm=npla.norm(flow)
                    if not easy and fnorm>2.5e-6:
                        k2=np.array(f((vect+d*flow/(2*fnorm)).tolist(),*args,**kwargs))
                        normk2=npla.norm(k2)
                        if normk2>zero:
                            k3=np.array(f((vect+d*k2/(2*npla.norm(k2))).tolist(),*args,**kwargs))
                        else:
                            k3=flow.copy()
                        normk3=npla.norm(k3)
                        if normk3>zero:
                            k4=np.array(f((vect+d*k3/npla.norm(k3)).tolist(),*args,**kwargs))
                        else:
                            k4=flow.copy()
                        k1234=flow+2*k2+2*k3+k4
                        normk1234=npla.norm(k1234)
                        if normk1234>zero:
                            return vect+d*k1234/normk1234
                        else:
                            return vect
                    elif fnorm>zero:
                        return vect+d*flow/fnorm
                    else:
                        return vect
                    
                if not easy:
                    trajold=self.coord.copy()
                    traj=step(self.coord.copy(),d)
                    count=1
                    while sum([(traj[i]-trajold[i])**2 for i in range(dim)])>(1e-2*dist)**2 and count<=20:
                        count+=1
                        trajold=traj.copy()
                        traj=self.coord.copy()
                        for i in range(count):
                            traj=step(traj,d/count)
                else:
                    traj=step(self.coord.copy(),d)
                            
                for i in range(len(traj)):
                    if traj[i]<=start[i]:
                        traj[i]=start[i]
                    elif traj[i]>=stop[i]:
                        traj[i]=stop[i]
                        
                self.traj=vertex(traj)
                self.staticbool=all([self.coord[i]==self.traj.coord[i] for i in range(len(self.coord))])
            
    class simplex:
        def __init__(self,V,level):          
            self.V=V
            self.expandbool=[True for v in self.V]
            self.level=level
            
        def expand(self,dist):
            res=[]
            for i in range(len(self.V)):
                if not self.expandbool[i] and all([self.expandbool[j] for j in range(len(self.V)) if j!=i]):
                    index=[j for j in range(len(self.V)) if j!=i]
                    for ind in index:
                        self.V[ind].trajectory()
                    if any([not self.V[ind].staticbool for ind in index]):
                        splitbool=[]
                        for j in range(len(index)):
                            for k in range(j+1,len(index)):
                                if sum([(self.V[index[j]].traj.coord[l]-self.V[index[k]].traj.coord[l])**2 for l in range(dim)])>=2*dist**2:
                                    splitbool.append(True)
                                else:
                                    splitbool.append(False)
                        if all(splitbool):
                            vertices=[]
                            vIndices=[]
                            for j in range(len(index)):
                                self.expandbool[index[j]]=False
                                for k in range(j+1,len(index)):
                                    v=vertex((self.V[index[j]].coord+self.V[index[k]].coord)/2)
                                    v.trajectory()
                                    vertices.append(v)
                                    vIndices.append([j,k])
                            for j in range(len(index)):
                                s=simplex([self.V[index[j]],self.V[index[j]].traj]+[vertices[k].traj for k in range(len(vertices)) if j in vIndices[k]],self.level+1)
                                s.expandbool[0]=False
                                res.append(s)
                            if len(self.V)==4:
                                s=simplex([self.V[index[0]]]+[vertices[j].traj for j in range(len(vertices))],self.level+1)
                                s.expandbool[0]=False
                                res.append(s)
                        else:
                            if len(self.V)==3:
                                for j in range(len(index)):
                                    self.expandbool[index[j]]=False
                                if sum([(self.V[index[0]].coord[j]-self.V[index[1]].traj.coord[j])**2 for j in range(dim)])<=sum([(self.V[index[1]].coord[j]-self.V[index[0]].traj.coord[j])**2 for j in range(dim)]):
                                    s=simplex([self.V[index[0]],self.V[index[0]].traj,self.V[index[1]].traj],self.level+1)
                                else:
                                    s=simplex([self.V[index[1]],self.V[index[0]].traj,self.V[index[1]].traj],self.level+1)
                                s.expandbool[0]=False
                                res.append(s)
                            elif len(self.V)==4:
                                for ind in index:
                                        self.V[ind]=self.V[ind].traj
                break
            return res
                
        def draw(self,plotaxis,color,alpha):
            if dim==1:
                if len(self.V)==2:
                    plotaxis.plot([self.V[i].coord[0] for i in range(2)],
                                   [self.level,self.level+1],color=color,alpha=alpha)
            elif dim==2:
                if len(self.V)==2:
                    plotaxis.plot([self.V[i].coord[0] for i in range(2)],
                                  [self.V[i].coord[1] for i in range(2)],color=color,alpha=alpha)
                elif len(self.V)==3:
                    plotaxis.plot([self.V[i].coord[0] for i in range(3)]+[self.V[0].coord[0]],
                                  [self.V[i].coord[1] for i in range(3)]+[self.V[0].coord[1]],color=color,alpha=alpha)
            elif dim==3:
                viewWeights=[math.sin((90-plotaxis.elev)*math.pi/180)*math.cos(plotaxis.azim*math.pi/180),
                             math.sin((90-plotaxis.elev)*math.pi/180)*math.sin(plotaxis.azim*math.pi/180),
                             math.cos((90-plotaxis.elev)*math.pi/180)]
                if len(self.V)==2:
                    plotaxis.plot([self.V[i].coord[0] for i in range(2)],
                                  [self.V[i].coord[1] for i in range(2)],
                                  [self.V[i].coord[2] for i in range(2)],
                                  zorder=sum([(self.V[0].coord[i]-start[i]+self.V[1].coord[i]-stop[i])/2*viewWeights[i] for i in range(3)]),
                                  color=color,alpha=alpha)
                elif len(self.V)==3:
                    plotaxis.plot([self.V[i].coord[0] for i in range(3)]+[self.V[0].coord[0]],
                                  [self.V[i].coord[1] for i in range(3)]+[self.V[0].coord[1]],
                                  [self.V[i].coord[2] for i in range(3)]+[self.V[0].coord[2]],
                                  zorder=sum([((self.V[0].coord[i]+self.V[1].coord[i]+self.V[2].coord[i])/3-(start[i]+stop[i])/2)*viewWeights[i] for i in range(3)]),
                                  color=color,alpha=alpha)
                elif len(self.V)==4:
                    plotaxis.plot([self.V[i].coord[0] for i in range(1,4)]+[self.V[1].coord[0]],
                                  [self.V[i].coord[1] for i in range(1,4)]+[self.V[1].coord[1]],
                                  [self.V[i].coord[2] for i in range(1,4)]+[self.V[1].coord[2]],
                                  zorder=sum([((self.V[1].coord[i]+self.V[2].coord[i]+self.V[3].coord[i])/3-(start[i]+stop[i])/2)*viewWeights[i] for i in range(3)]),
                                  color=color,alpha=alpha)

    if len(seed) not in [1,2,3]:
        if stable:
            print("The stable manifold of your inputted point does not exist")
        else:
            
            print("The unstable manifold of your inputted point does not exist")
    elif maxlevel>0:
        vertex0=vertex(point)
        Simplices=[]
        V=[[vertex(point+seed[i]),vertex(point-seed[i])] for i in range(len(seed))]
        for v in V:
            v[0].trajectory()
            v[1].trajectory()
        Vtraj=[[V[i][0].traj,V[i][1].traj] for i in range(len(seed))]
        for vset in product(*Vtraj):
            Simplices.append(simplex([vertex0]+list(vset),1))
        for i in range(len(Simplices)):
            Simplices[i].expandbool[0]=False
        
        for i in range(maxlevel):
            baseSimplices=[s for s in Simplices]
            for bs in baseSimplices:
                if any(bs.expandbool):
                    Simplices+=bs.expand(dist)
    
        for s in Simplices:
            if colormap is not None:
                s.draw(plotaxis,colormap(s.level/maxlevel),alpha)
            else:
                s.draw(plotaxis,color,alpha)
    if plotshowbool:
        if savefigName is not None and isinstance(savefigName,str):
            plt.savefig(savefigName+".png",bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def rungeKutta(a,b,c,f,vect0,args,kwargs,start,stop,tstart,tstop,deltat,inform,rev,autonomous,adapt):
    #This method is a universal explicit Runge Kutta numerical integration 
    #technique using Butcher Tableau a in R^2, b in R, and C in R. Variable f
    #represents the system of differential equations while variable vect0 is the
    #initial position of the solution. The solution is restricted by lower bound
    #vector start and upper bound vector stop, while time is restricted between
    #tstart and tstop, with a time step variable of deltat. Variable inform allows
    #the user to receive information about the integration process. Variable rev
    #reverses the direction of the solution, as if the solution is being plotted
    #over negative time. Variable autonomous dictates whether f is an autonomous
    #system of differential equations. Variable adapt allows for adaptive step sizes
    #in the integration process, allowing for potential increased accuracy and 
    #potential increased speed.
    if len(vect0)==len(start) and len(start)==len(stop):
        dim=len(vect0)
        res = np.array([float(vect0[i]) for i in range(dim)])
        restau = float(tstart)
        sol = [[res[i]] for i in range(dim)]
        tau = [restau]
        alpha=1.0
        if adapt:
            relErrTolmax=(1e-1*deltat)**2
            relErrTolmin=(1e-2*deltat)**2
        count=0
        N=round((tstop-tstart)/deltat)
        maxcount=32*N
        while restau<tstop and count<maxcount:
            if adapt:                
                adaptedbool=False
                halvedbool=False
                doubledbool=False
                while not adaptedbool:
                    adaptedbool=True
                    
                    cRes=np.array([sol[i][len(tau)-1] for i in range(dim)])
                    K1=np.array(f(cRes,*args,**kwargs))
                    
                    k2=np.array(f(cRes+alpha*deltat*K1,*args,**kwargs))
                    Res1=cRes+0.5*alpha*deltat*(K1+k2)
                    k1=np.array(f(Res1,*args,**kwargs))
                    k2=np.array(f(Res1+alpha*deltat*k1,*args,**kwargs))
                    Res2=Res1+0.5*alpha*deltat*(k1+k2)
                    
                    k2=np.array(f(cRes+0.5*alpha*deltat*K1,*args,**kwargs))
                    Reshalf=cRes+0.25*alpha*deltat*(K1+k2)
                    k1=np.array(f(Reshalf,*args,**kwargs))
                    k2=np.array(f(Reshalf+0.5*alpha*deltat*k1,*args,**kwargs))
                    Reshalf=Reshalf+0.25*alpha*deltat*(k1+k2)
                    
                    k2=np.array(f(cRes+2.0*alpha*deltat*K1,*args,**kwargs))
                    Resdouble=cRes+alpha*deltat*(K1+k2)
                    
                    if sum([(Reshalf[i]-Res1[i])**2 for i in range(dim)])>relErrTolmax and not doubledbool and alpha>=1.0/32.0:
                        alpha*=0.5
                        adaptedbool=False
                        halvedbool=True
                    elif sum([(Resdouble[i]-Res2[i])**2 for i in range(dim)])<relErrTolmin and not halvedbool:
                        try:
                            J=mtb.jacobian(f,res,args=args,kwargs=kwargs)
                            Lmbdas=npla.eigvals(J)
                            Lmbda=Lmbdas[0]
                            for i in range(1,Lmbdas.shape[0]):
                                if (Lmbdas[i].real)**2+(Lmbdas[i].imag)**2>(Lmbda.real)**2+(Lmbda.imag)**2:
                                    Lmbda=Lmbdas[i]
                            if (1+2.0*alpha*deltat*Lmbda.real)**2+(2.0*alpha*deltat*Lmbda.imag)**2<1:
                                alpha*=2.0
                                adaptedbool=False
                                doubledbool=True
                        except:
                            if alpha*deltat<=0.25:
                                alpha*=2.0
                                adaptedbool=False
                                doubledbool=True
                    
            K = []
            for i in range(len(b)):
                if autonomous:
                    if rev:
                        K.append(f([res[k] - alpha * deltat * sum([float(a[i][j]) * K[j][k] for j in range(i)]) for k in range(dim)],*args,**kwargs))
                    else:
                        K.append(f([res[k] + alpha * deltat * sum([float(a[i][j]) * K[j][k] for j in range(i)]) for k in range(dim)],*args,**kwargs))
                else:
                    if rev:
                        K.append(f([res[k] - alpha * deltat * sum([float(a[i][j]) * K[j][k] for j in range(i)]) for k in range(dim)],
                                    restau + float(c[i]) * alpha * deltat,*args,**kwargs))
                    else:
                        K.append(f([res[k] + alpha * deltat * sum([float(a[i][j]) * K[j][k] for j in range(i)]) for k in range(dim)],
                                    restau + float(c[i]) * alpha * deltat,*args,**kwargs))                  
            if rev:
                for i in range(dim):
                    res[i]-=alpha * deltat * sum([float(b[j]) * K[j][i] for j in range(len(K))])
            else:
                for i in range(dim):
                    res[i]+=alpha * deltat * sum([float(b[j]) * K[j][i] for j in range(len(K))])            
            
            restau+=alpha*deltat
            if restau>=tstop:
                if abs(tstop-restau)<=abs(tstop-restau-alpha*deltat):
                    for i in range(dim):
                        if count>=3:
                            res[i]=(sol[i][count-3]*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-tau[count])*(tstop-restau)/((tau[count-3]-tau[count-2])*(tau[count-3]-tau[count-1])*(tau[count-3]-tau[count])*(tau[count-3]-restau))
                                    +sol[i][count-2]*(tstop-tau[count-3])*(tstop-tau[count-1])*(tstop-tau[count])*(tstop-restau)/((tau[count-2]-tau[count-3])*(tau[count-2]-tau[count-1])*(tau[count-2]-tau[count])*(tau[count-2]-restau))
                                    +sol[i][count-1]*(tstop-tau[count-3])*(tstop-tau[count-2])*(tstop-tau[count])*(tstop-restau)/((tau[count-1]-tau[count-3])*(tau[count-1]-tau[count-2])*(tau[count-1]-tau[count])*(tau[count-1]-restau))
                                    +sol[i][count]*(tstop-tau[count-3])*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-restau)/((tau[count]-tau[count-3])*(tau[count]-tau[count-2])*(tau[count]-tau[count-1])*(tau[count]-restau))
                                    +res[i]*(tstop-tau[count-3])*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-tau[count])/((restau-tau[count-3])*(restau-tau[count-2])*(restau-tau[count-1])*(restau-tau[count])))
                        elif count>=2:
                            res[i]=(sol[i][count-2]*(tstop-tau[count-1])*(tstop-tau[count])*(tstop-restau)/((tau[count-2]-tau[count-1])*(tau[count-2]-tau[count])*(tau[count-2]-restau))
                                    +sol[i][count-1]*(tstop-tau[count-2])*(tstop-tau[count])*(tstop-restau)/((tau[count-1]-tau[count-2])*(tau[count-1]-tau[count])*(tau[count-1]-restau))
                                    +sol[i][count]*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-restau)/((tau[count]-tau[count-2])*(tau[count]-tau[count-1])*(tau[count]-restau))
                                    +res[i]*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-tau[count])/((restau-tau[count-2])*(restau-tau[count-1])*(restau-tau[count])))
                        elif count>=1:
                            res[i]=(sol[i][count-1]*(tstop-tau[count])*(tstop-restau)/((tau[count-1]-tau[count])*(tau[count-1]-restau))
                                    +sol[i][count]*(tstop-tau[count-1])*(tstop-restau)/((tau[count]-tau[count-1])*(tau[count]-restau))
                                    +res[i]*(tstop-tau[count-1])*(tstop-tau[count])/((restau-tau[count-1])*(restau-tau[count])))
                        else:
                            res[i]=(sol[i][count]*(tstop-restau)/(tau[count]-restau)
                                    +res[i]*(tstop-tau[count])/(restau-tau[count]))
                        sol[i][count]=(min(max(float(res[i]),start[i]),stop[i]))
                    restau=tstop
                    tau[count]=tstop
                else:
                    for i in range(dim):
                        if count>=3:
                            res[i]=(sol[i][count-3]*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-tau[count])*(tstop-restau)/((tau[count-3]-tau[count-2])*(tau[count-3]-tau[count-1])*(tau[count-3]-tau[count])*(tau[count-3]-restau))
                                    +sol[i][count-2]*(tstop-tau[count-3])*(tstop-tau[count-1])*(tstop-tau[count])*(tstop-restau)/((tau[count-2]-tau[count-3])*(tau[count-2]-tau[count-1])*(tau[count-2]-tau[count])*(tau[count-2]-restau))
                                    +sol[i][count-1]*(tstop-tau[count-3])*(tstop-tau[count-2])*(tstop-tau[count])*(tstop-restau)/((tau[count-1]-tau[count-3])*(tau[count-1]-tau[count-2])*(tau[count-1]-tau[count])*(tau[count-1]-restau))
                                    +sol[i][count]*(tstop-tau[count-3])*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-restau)/((tau[count]-tau[count-3])*(tau[count]-tau[count-2])*(tau[count]-tau[count-1])*(tau[count]-restau))
                                    +res[i]*(tstop-tau[count-3])*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-tau[count])/((restau-tau[count-3])*(restau-tau[count-2])*(restau-tau[count-1])*(restau-tau[count])))
                        elif count>=2:
                            res[i]=(sol[i][count-2]*(tstop-tau[count-1])*(tstop-tau[count])*(tstop-restau)/((tau[count-2]-tau[count-1])*(tau[count-2]-tau[count])*(tau[count-2]-restau))
                                    +sol[i][count-1]*(tstop-tau[count-2])*(tstop-tau[count])*(tstop-restau)/((tau[count-1]-tau[count-2])*(tau[count-1]-tau[count])*(tau[count-1]-restau))
                                    +sol[i][count]*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-restau)/((tau[count]-tau[count-2])*(tau[count]-tau[count-1])*(tau[count]-restau))
                                    +res[i]*(tstop-tau[count-2])*(tstop-tau[count-1])*(tstop-tau[count])/((restau-tau[count-2])*(restau-tau[count-1])*(restau-tau[count])))
                        elif count>=1:
                            res[i]=(sol[i][count-1]*(tstop-tau[count])*(tstop-restau)/((tau[count-1]-tau[count])*(tau[count-1]-restau))
                                    +sol[i][count]*(tstop-tau[count-1])*(tstop-restau)/((tau[count]-tau[count-1])*(tau[count]-restau))
                                    +res[i]*(tstop-tau[count-1])*(tstop-tau[count])/((restau-tau[count-1])*(restau-tau[count])))
                        else:
                            res[i]=(sol[i][count]*(tstop-restau)/(tau[count]-restau)
                                    +res[i]*(tstop-tau[count])/(restau-tau[count]))
                        sol[i].append(min(max(float(res[i]),start[i]),stop[i]))
                    restau=tstop
                    tau.append(tstop)
                    count+=1
            else:
                for i in range(dim):
                    sol[i].append(min(max(float(res[i]),start[i]),stop[i]))
                tau.append(restau)
                count+=1
        if inform:
            if count<maxcount:
                if adapt:
                    print("The adaptive Runge-Kutta process speed-up: %0.3f"%(N/count))
            elif count==maxcount:
                if autonomous:
                    print("The Runge-Kutta process had to be terminated early. We were able to approximate up to t=%0.3f"%restau)
                else:
                    print("The Runge-Kutta process had to be terminated early. We were able to approximate up to tau=%0.3f"%restau)
        for i in range(dim):
            sol[i]=np.array(sol[i])
        tau=np.array(tau)
        if rev:
            tau = np.flip(tau,0)
        #if adapt:
        #    sol,tau=gtb.interpolate(sol,tau,N)
        return sol,tau
    elif inform:
        print("Error in Runge Kutta: check the length of your inputted vectors")
        print("Length of vect0: %i"%len(vect0))
        print("Length of start: %i"%len(start))
        print("Length of stop: %i"%len(stop))
        return [], []
    
def euler(f,vect0,start,stop,tstart,tstop,deltat,args=[],kwargs={},inform=True,rev=False,autonomous=True,adapt=False):
    #Explicit Euler Forward Numerical Integration Algorithm
    a,b,c=eulerTableau()
    return rungeKutta(a,b,c,f,vect0,args,kwargs,start,stop,tstart,tstop,deltat,inform,rev,autonomous,adapt)

def rk2(f,vect0,start,stop,tstart,tstop,deltat,args=[],kwargs={},inform=True,rev=False,autonomous=True,adapt=False):
    #Explicit Midpoint Method Numerical Integration Algorithm
    a,b,c=rk2Tableau()
    return rungeKutta(a,b,c,f,vect0,args,kwargs,start,stop,tstart,tstop,deltat,inform,rev,autonomous,adapt)

def rk4(f,vect0,start,stop,tstart,tstop,deltat,args=[],kwargs={},inform=True,rev=False,autonomous=True,adapt=False):
    #Explicit RK4 Numerical Integration Algorithm
    a,b,c=rk4Tableau()
    return rungeKutta(a,b,c,f,vect0,args,kwargs,start,stop,tstart,tstop,deltat,inform,rev,autonomous,adapt)

def rk12(f,vect0,start,stop,tstart,tstop,deltat,args=[],kwargs={},inform=True,rev=False,autonomous=True,adapt=False):
    #Explicit RK12(10) Numerical Integration Algorithm
    a,b,c=rk12Tableau()
    return rungeKutta(a,b,c,f,vect0,args,kwargs,start,stop,tstart,tstop,deltat,inform,rev,autonomous,adapt)

def rk14(f,vect0,start,stop,tstart,tstop,deltat,args=[],kwargs={},inform=True,rev=False,autonomous=True,adapt=False):
    #Explicit RK14(12) Numerical Integration Algorithm
    a,b,c=rk14Tableau()
    return rungeKutta(a,b,c,f,vect0,args,kwargs,start,stop,tstart,tstop,deltat,inform,rev,autonomous,adapt)

def verlet(f,vect0,start,stop,tstart,tstop,deltat,args=[],kwargs={},inform=True,rev=False,adapt=False):
    #This method is a Verlet numerical double integration algorithm. Variable f
    #represents the system of differential equations while variable vect0 is the
    #initial position of the solution, organized as [x_1(0),x_2(0),...x_n(0),
    #dx_1(0)/dt,dx_2(0)/dt,...dx_n(0)/dt]. The solution is restricted by lower 
    #bound vector start and upper bound vector stop, while time is restricted 
    #between tstart and tstop, with a time step variable of deltat. Variable 
    #inform allows the user to receive information about the integration process.
    #Variable rev reverses the direction of the solution, as if the solution is
    #being plotted over negative time. Variable adapt allows for adaptive step 
    #sizes in the integration process, allowing for potential increased accuracy
    #and potential increased speed.
    if len(vect0)%2==0 and int(len(vect0)/2)==len(start) and len(start)==len(stop):
        dim = int(len(vect0)/2)
        vect0=np.array(vect0[0:len(vect0):2])
        vectprime0=np.array(vect0[1:len(vect0):2])
        res1=vect0.copy()
        res2=vect0.copy()
        restau = float(tstart)
        sol = [[res1[i]] for i in range(dim)]
        tau = [restau]
        alpha1 = 1.0
        alpha2 = 1.0
        if adapt:
            relErrTolmax=1e-1*(deltat)**2
            relErrTolmin=1e-3*(deltat)**2
        count = 0
        while restau<tstop and count<10*round((tstop-tstart)/deltat):
            if adapt:
                alpha2=alpha1
                adaptedbool=False
                halvedbool=False
                doubledbool=False
                while not adaptedbool:
                    adaptedbool=True
                    if count==0:
                        Res1=res1+alpha1*deltat*vectprime0+0.5*(alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                        Reshalf=res1+0.5*alpha1*deltat*vectprime0+0.5*(0.5*alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                        Reshalf=2*Reshalf-res1+(0.5*alpha1*deltat)**2*np.array(f(Reshalf,*args,**kwargs))
                        Res2=2*Res1-res1+(alpha1*deltat)**2*np.array(f(Res1,*args,**kwargs))
                        Resdouble=res1+2*alpha1*deltat*vectprime0+0.5*(2*alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                    else:
                        Res1=2*res1-res2+(alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                        Reshalf=2*res1-res2+(0.5*alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                        Reshalf=2*Reshalf-res1+(0.5*alpha1*deltat)**2*np.array(f(Reshalf,*args,**kwargs))
                        Res2=2*Res1-res1+(alpha1*deltat)**2*np.array(f(Res1,*args,**kwargs))
                        Resdouble=2*res1-res2+(2*alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                    
                    if sum([(Reshalf[i]-Res1[i])**2 for i in range(dim)])>relErrTolmax and not doubledbool and alpha1>=1.0/32.0:
                        alpha1*=0.5
                        adaptedbool=False
                        halvedbool=True
                    elif sum([(Resdouble[i]-Res2[i])**2 for i in range(dim)])<relErrTolmin and not halvedbool and alpha1*deltat<=0.25:
                        alpha1*=2.0
                        adaptedbool=False
                        doubledbool=True
            if count==0:
                if rev:
                    res1=res1-alpha1*deltat*vectprime0+0.5*(alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                else:
                    res1=res1+alpha1*deltat*vectprime0+0.5*(alpha1*deltat)**2*np.array(f(res1,*args,**kwargs))
                restau+=alpha1*deltat
            else:
                tempres=res1.copy()
                res1=res1+alpha1*(res1-res2)/alpha2+0.5*alpha1*(alpha1+alpha2)*deltat**2*np.array(f(res1,*args,**kwargs))
                res2=tempres.copy()
                restau+=alpha1*deltat
            for i in range(dim):
                sol[i].append(min(max(float(res1[i]),start[i]),stop[i]))
            tau.append(restau)
            #if all([start[i]<=float(res1[i]) for i in range(dim)]) and all([float(res1[i])<=stop[i] for i in range(dim)]):
            #    for i in range(dim):
            #        sol[i].append(float(res1[i]))
            #    tau.append(restau)
            count+=1
        if inform:
            if count<10*round((tstop-tstart)/deltat):
                if adapt:
                    print("The adaptive Verlet process efficiency: %0.3f"%(round((tstop-tstart)/deltat)/count))
            elif count>=10*round((tstop-tstart)/deltat):
                print("The Verlet process had to be terminated early. We were able to approximate up to t=%0.3f"%restau)
        for i in range(dim):
            sol[i]=np.array(sol[i])
        tau=np.array(tau)
        return sol,tau
    else:
        print("Error in Verlet: check the length of your inputted vectors")
        print("Length of vect0: %i"%len(vect0))
        print("Length of start: %i"%len(start))
        print("Length of stop: %i"%len(stop))
        return [], []

def eulerTableau():
    #Explicit Euler Butcher Tableau
    c=[0.0]
    b=[1.0]
    a=[[0.0]]
    return a,b,c

def rk2Tableau():
    #Explicit RK2 Butcher Tableau
    c=[0.0,0.5]
    b=[0.0,1.0]
    a=[[0.0,0.0],[0.5,0.0]]
    return a,b,c

def rk4Tableau():
    #Explicit RK4 Butcher Tableau
    c = [0.0,0.5,0.5,1.0]
    b = [1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0]
    a = [[0.0,0.0,0.0,0.0],[0.5,0.0,0.0,0.0],[0.0,0.5,0.0,0.0],[0.0,0.0,1.0,0.0]]
    return a,b,c

def rk12Tableau():
    #Explicit RK12(10) Butcher Tableau
    #(source = http://sce.uhcl.edu/rungekutta/rk1210.txt)
    c = np.zeros(25)
    c[1] = 0.2
    c[2] = 0.5555555555555556
    c[3] = 0.8333333333333334
    c[4] = 0.3333333333333333
    c[5] = 1.0
    c[6] = 0.6718357091705138
    c[7] = 0.2887249411106202
    c[8] = 0.5625
    c[9] = 0.8333333333333334
    c[10] = 0.9476954311791993
    c[11] = 0.054811287686380265
    c[12] = 0.08488805186071653
    c[13] = 0.2655756032646429
    c[14] = 0.5
    c[15] = 0.7344243967353571
    c[16] = 0.9151119481392834
    c[17] = 0.9476954311791993
    c[18] = 0.8333333333333334
    c[19] = 0.2887249411106202
    c[20] = 0.6718357091705138
    c[21] = 0.3333333333333333
    c[22] = 0.5555555555555556
    c[23] = 0.2
    c[24] = 1.0
    
    b = np.zeros(25)
    b[0] = 0.023809523809523808
    b[1] = 0.0234375
    b[2] = 0.03125
    b[4] = 0.041666666666666664
    b[6] = 0.05
    b[7] = 0.05
    b[9] = 0.1
    b[10] = 0.07142857142857142
    b[12] = 0.13841302368078298
    b[13] = 0.2158726906049313
    b[14] = 0.2438095238095238
    b[15] = 0.2158726906049313
    b[16] = 0.13841302368078298
    b[17] = -0.07142857142857142
    b[18] = -0.1
    b[19] = -0.05
    b[20] = -0.05
    b[21] = -0.041666666666666664
    b[22] = -0.03125
    b[23] = -0.0234375
    b[24] = 0.023809523809523808
    
    a=np.zeros((25,25))
    a[1][0] = 0.2
    a[2][0] = -0.21604938271604937
    a[2][1] = 0.7716049382716049
    a[3][0] = 0.20833333333333334
    a[3][2] = 0.625
    a[4][0] = 0.19333333333333333
    a[4][2] = 0.22
    a[4][3] = -0.08
    a[5][0] = 0.1
    a[5][3] = 0.4
    a[5][4] = 0.5
    a[6][0] = 0.10336447165001048
    a[6][3] = 0.12405309452894676
    a[6][4] = 0.4831711675610329
    a[6][5] = -0.038753024569476324
    a[7][0] = 0.12403826143183333
    a[7][4] = 0.21705063219795848
    a[7][5] = 0.013745579207596677
    a[7][6] = -0.06610953172676828
    a[8][0] = 0.0914774894856883
    a[8][5] = -0.005443485237174697
    a[8][6] = 0.06807168016884535
    a[8][7] = 0.40839431558264105
    a[9][0] = 0.08900136525025511
    a[9][5] = 0.004995282266455323
    a[9][6] = 0.397918238819829
    a[9][7] = 0.4279302107525766
    a[9][8] = -0.0865117637557827
    a[10][0] = 0.06950876241349076
    a[10][5] = 0.12914694190017645
    a[10][6] = 1.530736381023113
    a[10][7] = 0.57787476112914
    a[10][8] = -0.9512947723210889
    a[10][9] = -0.40827664296563193
    a[11][0] = 0.044486140329513583
    a[11][5] = -0.0038047686705696172
    a[11][6] = 0.01069550640296242
    a[11][7] = 0.020961624449990432
    a[11][8] = -0.023314602325932177
    a[11][9] = 0.0026326598106453697
    a[11][10] = 0.0031547276897702504
    a[12][0] = 0.019458881511975546
    a[12][8] = 6.785129491718125e-05
    a[12][9] = -4.297958590492736e-05
    a[12][10] = 1.7635898226028515e-05
    a[12][11] = 0.0653866627415027
    a[13][0] = 0.2068368356642771
    a[13][8] = 0.016679606710415646
    a[13][9] = -0.008795015632007103
    a[13][10] = 0.003466754553624639
    a[13][11] = -0.8612644601057177
    a[13][12] = 0.9086518820740502
    a[14][0] = 0.0203926084654484
    a[14][8] = 0.0869469392016686
    a[14][9] = -0.019164963041014983
    a[14][10] = 0.006556291594936633
    a[14][11] = 0.09874761281274348
    a[14][12] = 0.005353646955249961
    a[14][13] = 0.3011678640109679
    a[15][0] = 0.2284104339177781
    a[15][8] = -0.4987074007930252
    a[15][9] = 0.1348411683357245
    a[15][10] = -0.03874582440558342
    a[15][11] = -1.2747325747347484
    a[15][12] = 1.4391636446287717
    a[15][13] = -0.21400746796799025
    a[15][14] = 0.9582024177544303
    a[16][0] = 2.002224776559742
    a[16][8] = 2.067018099615249
    a[16][9] = 0.6239781360861395
    a[16][10] = -0.046228368550031144
    a[16][11] = -8.849732883626496
    a[16][12] = 7.7425770785085595
    a[16][13] = -0.5883585192508692
    a[16][14] = -1.1068373336238064
    a[16][15] = -0.929529037579204
    a[17][0] = 3.1378953341207345
    a[17][5] = 0.12914694190017645
    a[17][6] = 1.530736381023113
    a[17][7] = 0.57787476112914
    a[17][8] = 5.420882630551267
    a[17][9] = 0.2315469260348293
    a[17][10] = 0.07592929955789135
    a[17][11] = -12.372997338018651
    a[17][12] = 9.854558834647696
    a[17][13] = 0.08591114313704365
    a[17][14] = -5.652427528626439
    a[17][15] = -1.9430093524281962
    a[17][16] = -0.12835260184940453
    a[18][0] = 1.3836005443219601
    a[18][5] = 0.004995282266455323
    a[18][6] = 0.397918238819829
    a[18][7] = 0.4279302107525766
    a[18][8] = -1.3029910742447577
    a[18][9] = 0.661292278669377
    a[18][10] = -0.14455977430695435
    a[18][11] = -6.965760347317982
    a[18][12] = 6.6580854323599175
    a[18][13] = -1.669973751088415
    a[18][14] = 2.064137023180353
    a[18][15] = -0.6747439626443065
    a[18][16] = -0.001156188347949395
    a[18][17] = -0.005440579086770074
    a[19][0] = 0.9512362970482877
    a[19][4] = 0.21705063219795848
    a[19][5] = 0.013745579207596677
    a[19][6] = -0.06610953172676828
    a[19][8] = 0.15228169673641445
    a[19][9] = -0.33774101835759984
    a[19][10] = -0.019282598163399577
    a[19][11] = -3.682592696968668
    a[19][12] = 3.1619787040698206
    a[19][13] = -0.3704625221068853
    a[19][14] = -0.05149742003654404
    a[19][15] = -0.0008296255321201529
    a[19][16] = 2.798010414192786e-06
    a[19][17] = 0.041860391641236026
    a[19][18] = 0.27908425509087736
    a[20][0] = 0.10336447165001048
    a[20][3] = 0.12405309452894676
    a[20][4] = 0.4831711675610329
    a[20][5] = -0.038753024569476324
    a[20][7] = -0.43831382036112243
    a[20][9] = -0.21863663372167666
    a[20][10] = -0.031233476439471924
    a[20][17] = 0.031233476439471924
    a[20][18] = 0.21863663372167666
    a[20][19] = 0.43831382036112243
    a[21][0] = 0.19333333333333333
    a[21][2] = 0.22
    a[21][3] = -0.08
    a[21][6] = 0.0984256130499316
    a[21][7] = -0.19641088922305466
    a[21][9] = 0.43645793049306875
    a[21][10] = 0.06526137216757211
    a[21][17] = -0.06526137216757211
    a[21][18] = -0.43645793049306875
    a[21][19] = 0.19641088922305466
    a[21][20] = -0.0984256130499316
    a[22][0] = -0.21604938271604937
    a[22][1] = 0.7716049382716049
    a[22][4] = -0.6666666666666666
    a[22][6] = -0.39069646929597845
    a[22][20] = 0.39069646929597845
    a[22][21] = 0.6666666666666666
    a[23][0] = 0.2
    a[23][2] = -0.1646090534979424
    a[23][22] = 0.1646090534979424
    a[24][0] = 1.4717872488111041
    a[24][1] = 0.7875
    a[24][2] = 0.4212962962962963
    a[24][4] = 0.2916666666666667
    a[24][6] = 0.34860071762832956
    a[24][7] = 0.22949954476899484
    a[24][8] = 5.79046485790482
    a[24][9] = 0.4185875118565069
    a[24][10] = 0.307039880222474
    a[24][11] = -4.687009053506033
    a[24][12] = 3.1357166559380225
    a[24][13] = 1.4013482971096571
    a[24][14] = -5.52931101439499
    a[24][15] = -0.8531382355080633
    a[24][16] = 0.10357578037361014
    a[24][17] = -0.14047441695060095
    a[24][18] = -0.4185875118565069
    a[24][19] = -0.22949954476899484
    a[24][20] = -0.34860071762832956
    a[24][21] = -0.2916666666666667
    a[24][22] = -0.4212962962962963
    a[24][23] = -0.7875
    return a,b,c

def rk14Tableau():
    # Explicit RK14(12) Butcher Tableau (source = http://sce.uhcl.edu/rungekutta/rk1412.txt)
    c=np.zeros(35)
    c[1] = 0.1111111111111111
    c[2] = 0.5555555555555556
    c[3] = 0.8333333333333334
    c[4] = 0.3333333333333333
    c[5] = 1.0
    c[6] = 0.669986979272773
    c[7] = 0.29706838421381837
    c[8] = 0.7272727272727273
    c[9] = 0.14015279904218877
    c[10] = 0.7007010397701507
    c[11] = 0.36363636363636365
    c[12] = 0.2631578947368421
    c[13] = 0.039217224665027084
    c[14] = 0.8129175029283767
    c[15] = 0.16666666666666666
    c[16] = 0.9
    c[17] = 0.06412992574519669
    c[18] = 0.20414990928342885
    c[19] = 0.3953503910487606
    c[20] = 0.6046496089512394
    c[21] = 0.7958500907165712
    c[22] = 0.9358700742548033
    c[23] = 0.16666666666666666
    c[24] = 0.8129175029283767
    c[25] = 0.039217224665027084
    c[26] = 0.36363636363636365
    c[27] = 0.7007010397701507
    c[28] = 0.14015279904218877
    c[29] = 0.29706838421381837
    c[30] = 0.669986979272773
    c[31] = 0.3333333333333333
    c[32] = 0.5555555555555556
    c[33] = 0.1111111111111111
    c[34] = 1.0
    
    b=np.zeros(35)
    b[0] = 0.017857142857142856
    b[1] = 0.005859375
    b[2] = 0.01171875
    b[4] = 0.017578125
    b[6] = 0.0234375
    b[7] = 0.029296875
    b[9] = 0.03515625
    b[10] = 0.041015625
    b[11] = 0.046875
    b[13] = 0.052734375
    b[14] = 0.05859375
    b[15] = 0.064453125
    b[17] = 0.10535211357175302
    b[18] = 0.17056134624175218
    b[19] = 0.20622939732935194
    b[20] = 0.20622939732935194
    b[21] = 0.17056134624175218
    b[22] = 0.10535211357175302
    b[23] = -0.064453125
    b[24] = -0.05859375
    b[25] = -0.052734375
    b[26] = -0.046875
    b[27] = -0.041015625
    b[28] = -0.03515625
    b[29] = -0.029296875
    b[30] = -0.0234375
    b[31] = -0.017578125
    b[32] = -0.01171875
    b[33] = -0.005859375
    b[34] = 0.017857142857142856

    a=np.zeros((35,35))
    a[1][0] = 0.1111111111111111
    a[2][0] = -0.8333333333333334
    a[2][1] = 1.3888888888888888
    a[3][0] = 0.20833333333333334
    a[3][2] = 0.625
    a[4][0] = 0.19333333333333333
    a[4][2] = 0.22
    a[4][3] = -0.08
    a[5][0] = 0.1
    a[5][3] = 0.4
    a[5][4] = 0.5
    a[6][0] = 0.10348456163667978
    a[6][3] = 0.12206888730640722
    a[6][4] = 0.4825744903312466
    a[6][5] = -0.0381409600015607
    a[7][0] = 0.12438052665409441
    a[7][4] = 0.2261202821975843
    a[7][5] = 0.013788588761808088
    a[7][6] = -0.06722101339966845
    a[8][0] = 0.09369190656596738
    a[8][5] = -0.00613406843450511
    a[8][6] = 0.21601982562550306
    a[8][7] = 0.4236950635157619
    a[9][0] = 0.08384798124090527
    a[9][5] = -0.01179493671009738
    a[9][6] = -0.24729902056881264
    a[9][7] = 0.0978080858367729
    a[9][8] = 0.21759068924342062
    a[10][0] = 0.061525535976942825
    a[10][5] = 0.005922327803245033
    a[10][6] = 0.47032615996384114
    a[10][7] = 0.299688863848679
    a[10][8] = -0.2476568775939949
    a[10][9] = 0.11089502977143768
    a[11][0] = 0.04197000733627826
    a[11][5] = -0.003179876962662051
    a[11][6] = 0.806397714906192
    a[11][7] = 0.0975983126412389
    a[11][8] = 0.778575578158399
    a[11][9] = 0.20489042383159942
    a[11][10] = -1.5626157962746818
    a[12][0] = 0.04377267822337302
    a[12][8] = 0.006243650275201952
    a[12][9] = 0.20004309710957732
    a[12][10] = -0.008053283678049831
    a[12][11] = 0.021151752806739654
    a[13][0] = 0.028349925036351455
    a[13][8] = 0.002491632048558174
    a[13][9] = 0.023013878785459314
    a[13][10] = -0.003221559566929771
    a[13][11] = 0.009884425494476646
    a[13][12] = -0.021301077132888736
    a[14][0] = 0.343511894290243
    a[14][8] = 0.2104519120236274
    a[14][9] = 1.034274520572304
    a[14][10] = 0.006003036458644225
    a[14][11] = 0.8559381250996195
    a[14][12] = -0.9772350050367669
    a[14][13] = -0.6600269804792946
    a[15][0] = -0.014357400167216807
    a[15][8] = -0.036625327004904
    a[15][9] = 0.03502549756362137
    a[15][10] = 0.03609460163621135
    a[15][11] = -0.02652199675536811
    a[15][12] = 0.044569901130569814
    a[15][13] = 0.12434309333135825
    a[15][14] = 0.004138296932394807
    a[16][0] = 0.3560324044251203
    a[16][8] = -0.4501927589475626
    a[16][9] = 0.4305279070837109
    a[16][10] = 0.5119730290110223
    a[16][11] = 0.9083036388864043
    a[16][12] = -1.2392109337193393
    a[16][13] = -0.6490486616717615
    a[16][14] = 0.25170890458681927
    a[16][15] = 0.7799064703455864
    a[17][0] = 0.013093568740651306
    a[17][12] = -9.32053067985114e-05
    a[17][13] = 0.05053743342622993
    a[17][14] = 8.04470341944488e-07
    a[17][15] = 0.0005917260294941712
    a[17][16] = -4.0161472215455734e-07
    a[18][0] = 0.0207926484466053
    a[18][12] = 0.0005826959188000859
    a[18][13] = -0.00801700732358816
    a[18][14] = 4.0384764384713694e-06
    a[18][15] = 0.08546099980555061
    a[18][16] = -2.0448648093580423e-06
    a[18][17] = 0.10532857882443189
    a[19][0] = 1.4015344979573603
    a[19][12] = -0.23025200098422127
    a[19][13] = -7.211068404669129
    a[19][14] = 0.0037290156069483636
    a[19][15] = -4.7141549572712504
    a[19][16] = -0.0017636765754534924
    a[19][17] = 7.641305480386988
    a[19][18] = 3.5060204365975185
    a[20][0] = 11.951465069412068
    a[20][12] = 7.794809321081759
    a[20][13] = -56.45013938673258
    a[20][14] = 0.0912376306930645
    a[20][15] = -12.73362799254349
    a[20][16] = -0.039689592190471974
    a[20][17] = 54.43921418835709
    a[20][18] = -3.6441163792156925
    a[20][19] = -0.8045032499105099
    a[21][0] = -148.80942650710048
    a[21][12] = -91.72952782912564
    a[21][13] = 707.6561449715983
    a[21][14] = -1.1056361185748245
    a[21][15] = 176.13459188381137
    a[21][16] = 0.49138482421488067
    a[21][17] = -684.278000449815
    a[21][18] = 27.991060499839826
    a[21][19] = 13.193971003028233
    a[21][20] = 1.2512878128398044
    a[22][0] = -9.673079469481968
    a[22][12] = -4.469901508585055
    a[22][13] = 45.51271286909527
    a[22][14] = -0.07130850861838268
    a[22][15] = 11.227361406841274
    a[22][16] = 0.12624437671762273
    a[22][17] = -43.54393395494833
    a[22][18] = 0.787174307543059
    a[22][19] = 0.5322646967446842
    a[22][20] = 0.42242273399632535
    a[22][21] = 0.08591312495030672
    a[23][0] = -10.06640324470547
    a[23][8] = -0.036625327004904
    a[23][9] = 0.03502549756362137
    a[23][10] = 0.03609460163621135
    a[23][11] = -0.02652199675536811
    a[23][12] = -6.270889721814641
    a[23][13] = 48.2079237442563
    a[23][14] = -0.06944716891361656
    a[23][15] = 12.68106902048503
    a[23][16] = 0.011967116896832376
    a[23][17] = -46.72497649924824
    a[23][18] = 1.330296133266267
    a[23][19] = 1.007667875033983
    a[23][20] = 0.02095120519336651
    a[23][21] = 0.02101347063312642
    a[23][22] = 0.009521960144171218
    a[24][0] = -409.4780816777437
    a[24][8] = 0.2104519120236274
    a[24][9] = 1.034274520572304
    a[24][10] = 0.006003036458644225
    a[24][11] = 0.8559381250996195
    a[24][12] = -250.51699854744786
    a[24][13] = 1946.4246665238843
    a[24][14] = -3.0450388210231036
    a[24][15] = 490.6263795282817
    a[24][16] = 1.5664758953127091
    a[24][17] = -1881.9742899401117
    a[24][18] = 75.25922247248472
    a[24][19] = 34.57343569803311
    a[24][20] = 3.21147679440969
    a[24][21] = -0.4604080417384144
    a[24][22] = -0.08707183398418106
    a[24][23] = -7.393518141583031
    a[25][0] = 3.433474758535509
    a[25][8] = 0.002491632048558174
    a[25][9] = 0.023013878785459314
    a[25][10] = -0.003221559566929771
    a[25][11] = 0.009884425494476646
    a[25][12] = 2.162527993779225
    a[25][13] = -16.269986454645743
    a[25][14] = -0.12853450212052456
    a[25][15] = -8.989150426665043
    a[25][16] = -0.0034859536323202534
    a[25][17] = 15.793619411333982
    a[25][18] = -0.574403330914095
    a[25][19] = -0.3456020390213933
    a[25][20] = -0.006622414902065851
    a[25][21] = -0.007777881292422042
    a[25][22] = -0.0035608419240227493
    a[25][23] = 4.792825064499308
    a[25][24] = 0.15372546487306857
    a[26][0] = 32.30385208719854
    a[26][5] = -0.003179876962662051
    a[26][6] = 0.806397714906192
    a[26][7] = 0.0975983126412389
    a[26][8] = 0.778575578158399
    a[26][9] = 0.20489042383159942
    a[26][10] = -1.5626157962746818
    a[26][12] = 16.34298918823106
    a[26][13] = -154.54455529354362
    a[26][14] = 1.5697108870333487
    a[26][15] = 3.2768554508724814
    a[26][16] = -0.05034892451936532
    a[26][17] = 153.32115185804167
    a[26][18] = 7.175681863277205
    a[26][19] = -2.9403674867530047
    a[26][20] = -0.06658459460768032
    a[26][21] = -0.04623460549908437
    a[26][22] = -0.02041987335856794
    a[26][23] = -53.35231064387359
    a[26][24] = -1.3554871471507866
    a[26][25] = -1.5719627580123274
    a[27][0] = -16.64514674863415
    a[27][5] = 0.005922327803245033
    a[27][6] = 0.47032615996384114
    a[27][7] = 0.299688863848679
    a[27][8] = -0.2476568775939949
    a[27][9] = 0.11089502977143768
    a[27][11] = -0.49171904384622916
    a[27][12] = -11.47431544272895
    a[27][13] = 80.25931665762303
    a[27][14] = -0.38413230398004283
    a[27][15] = 7.281476674681076
    a[27][16] = -0.13269938461224837
    a[27][17] = -81.07998325257307
    a[27][18] = -1.2503749283562064
    a[27][19] = 2.592635949695437
    a[27][20] = -0.30144029834640457
    a[27][21] = 0.22138446078983234
    a[27][22] = 0.08275772747718929
    a[27][23] = 18.99606620406115
    a[27][24] = 0.2692319464096397
    a[27][25] = 1.6267482744706654
    a[27][26] = 0.49171904384622916
    a[28][0] = 0.08384798124090527
    a[28][5] = -0.01179493671009738
    a[28][6] = -0.24729902056881264
    a[28][7] = 0.0978080858367729
    a[28][8] = 0.21759068924342062
    a[28][10] = 0.13758560676332524
    a[28][11] = 0.04398702297150467
    a[28][13] = -0.5137008137681933
    a[28][14] = 0.8263556911513155
    a[28][15] = 25.701813971981185
    a[28][23] = -25.701813971981185
    a[28][24] = -0.8263556911513155
    a[28][25] = 0.5137008137681933
    a[28][26] = -0.04398702297150467
    a[28][27] = -0.13758560676332524
    a[29][0] = 0.12438052665409441
    a[29][4] = 0.2261202821975843
    a[29][5] = 0.013788588761808088
    a[29][6] = -0.06722101339966845
    a[29][9] = -0.8562389750854283
    a[29][10] = -1.963375228668589
    a[29][11] = -0.2323328227241194
    a[29][13] = 4.306607190864534
    a[29][14] = -2.927229632494655
    a[29][15] = -82.31316663978589
    a[29][23] = 82.31316663978589
    a[29][24] = 2.927229632494655
    a[29][25] = -4.306607190864534
    a[29][26] = 0.2323328227241194
    a[29][27] = 1.963375228668589
    a[29][28] = 0.8562389750854283
    a[30][0] = 0.10348456163667978
    a[30][3] = 0.12206888730640722
    a[30][4] = 0.4825744903312466
    a[30][5] = -0.0381409600015607
    a[30][7] = -0.5504995253108024
    a[30][9] = -0.7119158115851892
    a[30][10] = -0.5841296056715514
    a[30][13] = 2.1104630812586493
    a[30][14] = -0.08374947367395721
    a[30][15] = 5.1002149907232095
    a[30][23] = -5.1002149907232095
    a[30][24] = 0.08374947367395721
    a[30][25] = -2.1104630812586493
    a[30][27] = 0.5841296056715514
    a[30][28] = 0.7119158115851892
    a[30][29] = 0.5504995253108024
    a[31][0] = 0.19333333333333333
    a[31][2] = 0.22
    a[31][3] = -0.08
    a[31][6] = 0.10999342558072471
    a[31][7] = -0.2542970480762702
    a[31][9] = 0.8655707771166943
    a[31][10] = 3.3241644911409307
    a[31][13] = -12.010222331597793
    a[31][14] = 0.4766014662424932
    a[31][15] = -29.02430112210364
    a[31][23] = 29.02430112210364
    a[31][24] = -0.4766014662424932
    a[31][25] = 12.010222331597793
    a[31][27] = -3.3241644911409307
    a[31][28] = -0.8655707771166943
    a[31][29] = 0.2542970480762702
    a[31][30] = -0.10999342558072471
    a[32][0] = -0.8333333333333334
    a[32][1] = 1.3888888888888888
    a[32][4] = -0.75
    a[32][6] = -0.4925295437180263
    a[32][30] = 0.4925295437180263
    a[32][31] = 0.75
    a[33][0] = 0.1111111111111111
    a[33][2] = -0.2222222222222222
    a[33][32] = 0.2222222222222222
    a[34][0] = 0.28583514038897156
    a[34][1] = 0.2916666666666667
    a[34][2] = 0.21875
    a[34][4] = 0.1640625
    a[34][6] = 0.21819435494555667
    a[34][7] = 0.18039289847869777
    a[34][9] = 0.20571383940484503
    a[34][10] = 0.24271579158177023
    a[34][11] = 0.2464657808136293
    a[34][12] = -3.4499194079089084
    a[34][13] = 0.22887556216003607
    a[34][14] = 0.2832905997021514
    a[34][15] = 3.2108512583776663
    a[34][16] = -0.2235387773648457
    a[34][17] = -0.707121157204419
    a[34][18] = 3.2112334515028707
    a[34][19] = 1.4095434830966977
    a[34][20] = -0.15136205344374262
    a[34][21] = 0.37235057452701426
    a[34][22] = 0.2529787464063613
    a[34][23] = -3.2108512583776663
    a[34][24] = -0.2832905997021514
    a[34][25] = -0.22887556216003607
    a[34][26] = -0.2464657808136293
    a[34][27] = -0.24271579158177023
    a[34][28] = -0.20571383940484503
    a[34][29] = -0.18039289847869777
    a[34][30] = -0.21819435494555667
    a[34][31] = -0.1640625
    a[34][32] = -0.21875
    a[34][33] = -0.2916666666666667
    return a,b,c
