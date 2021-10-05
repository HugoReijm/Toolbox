import math
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import toolbox.generaltoolbox as gtb
import toolbox.matrixtoolbox as mtb

def generate(f,vect0,N,args=[],kwargs={}):
    #This method generates a finite sequence according to an inputted 
    #difference system. Variable f represents the governing difference 
    #system. Variable vect0 is the initial position of the sequence. 
    #Variable N dictates the size of the returning sequence.
    dim=len(vect0)
    sol=[[vect0[i]]+[0.0 for j in range(N-1)] for i in range(dim)]
    resvect=[vect0[i] for i in range(dim)]
    for i in range(1,N):
        try:
            resvect=f(resvect,*args,**kwargs)
            for j in range(dim):
                sol[j][i]=resvect[j]
        except OverflowError:
            print("Discrete System reached max value at iteration %s"%i)
            return sol[:i]
    return sol

def cobweb(f,vect0,N,args=[],kwargs={},color="black",alpha=1.0,savefigName=None):
    sol=generate(f,vect0,N,args=args,kwargs=kwargs)
    graphsize=9
    font = {"family": "serif",
        "color": "black",
        "weight": "bold",
        "size": "20"}
    dim=len(vect0)
    plotaxes=[]
    for i in range(dim):
        minsol=min(sol[i])
        maxsol=max(sol[i])
        plotfig=plt.figure(figsize=(graphsize,graphsize))
        plotaxes.append(plotfig.add_subplot(111))
        if dim==1:
            plotaxes[i].set_title("Cobweb Diagram",fontdict=font)
            plotaxes[i].set_xlabel("$\\mathbf{X_n}$",fontsize=16,rotation=0)
            plotaxes[i].set_ylabel("$\\mathbf{X_{n+1}}$",fontsize=16,rotation=0)
        else:
            plotaxes[i].set_title("Cobweb Diagram of Dimension %i"%i,fontdict=font)
            plotaxes[i].set_xlabel("$\\mathbf{X_n}^(%i)$"%i,fontsize=16,rotation=0)
            plotaxes[i].set_ylabel("$\\mathbf{X_{n+1}^(%i)}$"%i,fontsize=16,rotation=0)
        plotaxes[i].xaxis.set_tick_params(labelsize=16)
        plotaxes[i].yaxis.set_tick_params(labelsize=16)
        plotaxes[i].plot([minsol,maxsol],[minsol,maxsol],color="black",alpha=alpha)
        cobX=[sol[i][0],sol[i][0]]
        cobY=[minsol,sol[i][0]]
        for j in range(1,N):
            cobX+=[sol[i][j],sol[i][j]]
            cobY+=[sol[i][j-1],sol[i][j]]
        plotaxes[i].plot(cobX,cobY,color=color,alpha=alpha)
        if savefigName is not None and isinstance(savefigName,str):
            plt.savefig(savefigName+"Dimension"+str(i)+".png",bbox_inches="tight")
            plt.close()

def lyapunovSpectrum(sol,f,args=[],kwargs={},dist=1e-6,K=1,plotbool=False,plotaxis=None,savefigName=None):
    #This method approximates the first K values of the Lyapunov Spectrum of a 
    #sequence of a difference system. Variable sol is the inputted
    #sequence (organized first by dimension, then by chronology) and variable
    #f is the corresponding difference system. Variable dist is the distance 
    #used between the initial conditions for each trajectory in each iteration.
    #Infrastructure for plotting the convergence of each exponent is available 
    #(for visual verification). Variable savefigName allows for tge resulting 
    #graph to be plotted under the name <savefigName>.png.
    dim=len(sol)
    solcopy=[sol[i][:-1] for i in range(dim)]
    n=len(solcopy[0])
    if any([len(solcopy[i])!=n for i in range(1,dim)]):
        print("Error: Inputted solution is inviable. Terminating process")
        for j in range(dim):
            print("Length of sol[%i]: %i"%(j,len(sol[j])))
        return 0.0
    
    K=max(min(K,dim),1)
    vect0=np.array([float(solcopy[i][0]) for i in range(dim)])
    vects=[np.copy(vect0) for i in range(K)]
    for i in range(K):
        vects[i][i]=vects[i][i]+dist
        
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
            vects[j]=f(vects[j],*args,**kwargs)
            if not isinstance(vects[j],np.ndarray):
                vects[j]=np.array(vects[j])
            vols[j]=math.sqrt(GramDet([vects[l]-vect0 for l in range(j+1)]))

        for j in range(K):
            if vols[j]!=0:
                counts[j]+=1
                Lambda[j]+=math.log(vols[j])-(j+1)*math.log(dist)
                if plotbool:
                    if j==0:
                        plotLambda[j].append(Lambda[j]/i)
                    else:
                        if counts[j-1]==0:
                            plotLambda[j].append(Lambda[j]/i)
                        else:
                            plotLambda[j].append(Lambda[j]/i-Lambda[j-1]/i)
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
        lmbda.append(Lambda[0]/n)
    for i in range(1,K):
        if counts[i]!=0:
            if counts[i-1]==0:
                lmbda.append(Lambda[i]/n)
            else:
                lmbda.append(Lambda[i]/n-Lambda[i-1]/n)
    
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
                plotaxis.set_ylabel("$\mathbf{\lambda_{max}}$",fontsize=16,rotation=0)
            else:
                plotaxis.set_title("Lyapunov Spectrum Convergence",fontdict=font)
                plotaxis.set_xlabel("Iterations",fontsize=16,rotation=0)
                plotaxis.set_ylabel("$\mathbf{\lambda 's}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            showbool=True
        color=gtb.colors(K)
        for i in range(K):
            plotaxis.plot([j for j in range(counts[i])],plotLambda[i],color=color[i])
        if showbool:
            if savefigName is not None and isinstance(savefigName,str):
                plt.savefig(savefigName+".png",bbox_inches="tight")
                plt.close()
            else:
                plt.show()
    return lmbda

def manifold(f,point,start,stop,maxlevel,dist=1e-2,N=100,args=[],kwargs={},plotbool=True,plotaxis=None,colormap=None,color="black",alpha=1):
    dim=len(point)
    if not isinstance(point,np.ndarray):
        point=np.array(point)
    np.seterr(all="raise")
    plotshowbool=False
    if plotbool and plotaxis is None:
        graphsize=9
        font = {"family": "serif",
            "color": "black",
            "weight": "bold",
            "size": "14"}
        plotfig=plt.figure(figsize=(graphsize,graphsize))
        if dim==2:
            plotaxis=plotfig.add_subplot(111)
            plotaxis.set_title("Unstable Manifold of (%0.3f,%0.3f)"%(point[0],point[1]),fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontdict=font)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontdict=font)
            plotshowbool=True
        elif dim==3:
            plotaxis=Axes3D(plotfig)
            plotaxis.set_title("Unstable Manifold of (%0.3f,%0.3f,%0.3f)"%(point[0],point[1],point[2]),fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontdict=font)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontdict=font)
            plotaxis.set_zlabel("$\\mathbf{Z}$",fontdict=font)
            plotshowbool=True
            
    Sol=[]
    if colormap is not None:
        color=[]
        
    Lin=mtb.jacobian(f,point)
    if abs(npla.det(Lin)-1)<1e-6:
        for h in gtb.hammersley(dim*N,dim):
            sol=[point+2*dist*h-dist]
            if colormap is not None:
                c=[colormap(0)]
            pivot=sol[0].copy()
            for j in range(maxlevel-1):
                try:
                    res=np.array(f(pivot))
                    if all([start[k]<=res[k]<=stop[k] for k in range(dim)]):
                        sol.append(res)
                        if colormap is not None:
                            c.append(colormap(j/(maxlevel-1)))
                    pivot=res.copy()
                except:
                    break
            Sol+=sol
            if colormap is not None:
                color+=c

    else:
        eigval,eigvect=npla.eig(Lin)
        eigvect=eigvect.transpose()
        seed=[]
        for i in range(len(eigval)):
            if eigval[i].real>0:
                eigvect[i]=eigvect[i].real
                uniquebool=True
                for j in range(len(seed)):
                    if all([abs(eigvect[i][k]-seed[j][k])<1e-10 for k in range(dim)]):
                        uniquebool=False
                if uniquebool:
                    seed.append(eigvect[i])
        
        for s in seed:
            for i in range(N):
                sol=[point+dist*i*s/(N-1)]
                if colormap is not None:
                    c=[colormap(0)]
                pivot=sol[0].copy()
                for j in range(maxlevel-1):
                    try:
                        res=np.array(f(pivot))
                        if all([start[k]<=res[k]<=stop[k] for k in range(dim)]):
                            sol.append(res)
                            if colormap is not None:
                                c.append(colormap(j/(maxlevel-1)))
                        pivot=res.copy()
                    except:
                        break
                Sol+=sol
                if colormap is not None:
                    color+=c
                
                sol=[point-dist*i*s/(N-1)]
                if colormap is not None:
                    c=[colormap(0)]
                for j in range(maxlevel-1):
                    try:
                        res=np.array(f(pivot))
                        if all([start[k]<=res[k]<=stop[k] for k in range(dim)]):
                            sol.append(res)
                            if colormap is not None:
                                c.append(colormap(j/(maxlevel-1)))
                        pivot=res.copy()
                    except:
                        break
                Sol+=sol
                if colormap is not None:
                    color+=c
                
    if plotbool:
        if dim==2:
            plotaxis.scatter([Sol[i][0] for i in range(len(Sol))],
                        [Sol[i][1] for i in range(len(Sol))],s=1,color=color)
        else:
            plotaxis.scatter([Sol[i][0] for i in range(len(Sol))],
                        [Sol[i][1] for i in range(len(Sol))],
                        [Sol[i][2] for i in range(len(Sol))],s=1,color=color)
    if plotshowbool:
        plt.show()
