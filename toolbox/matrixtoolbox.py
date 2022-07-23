import math
import numpy as np
import numpy.linalg as npla
import scipy.sparse as scarse
import matplotlib.pyplot as plt
from toolbox.generaltoolbox import differentiate
from toolbox.generaltoolbox import double_differentiate

def plotMatrix(M,tol=1e-6,plotaxis=None,colormap=None,color="black",alpha=1.0):
    graphsize=9
    font = {"family": "serif",
        "color": "black",
        "weight": "bold",
        "size": "20"}
    plotfig=plt.figure(figsize=(graphsize,graphsize))
    plotaxis=plotfig.add_subplot(111)
    plt.gca().invert_yaxis()
    plotaxis.set_title("Matrix",fontdict=font)
    plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
    plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
    plotaxis.xaxis.set_tick_params(labelsize=16)
    plotaxis.yaxis.set_tick_params(labelsize=16)
    
    mX=[]
    mY=[]
    
    if ("bsr_matrix" in type(M).__name__ or "coo_matrix" in type(M).__name__ 
        or "csc_matrix" in type(M).__name__ or "csr_matrix" in type(M).__name__
        or "dia_matrix" in type(M).__name__ or "dok_matrix" in type(M).__name__
        or "lil_matrix" in type(M).__name__):
        if not isinstance(M,scarse.coo.coo_matrix):
            M=scarse.coo_matrix(M)
        minVal=M.data[0]
        maxVal=M.data[0]
        for i in range(len(M.data)):
            mX.append(M.row[i])
            mY.append(M.col[i])
            if M.data[i]<minVal:
                minVal=M.data[i]
            if M.data[i]>maxVal:
                maxVal=M.data[i]
        if colormap is not None and (maxVal-minVal)>abs(tol):
            res=plotaxis.scatter(mX,mY,c=M.data,alpha=alpha,s=150/math.sqrt(len(mX)),cmap=colormap)
            plotfig.colorbar(res)
        else:
            plotaxis.scatter(mX,mY,color=color,alpha=alpha,s=150/math.sqrt(len(mX)))   
    else:
        minVal=M[0][0]
        maxVal=M[0][0]
        data=[]
        for i in range(len(M)):
            for j in range(len(M[i])):
                if abs(M[i][j])>=abs(tol):
                    mX.append(i)
                    mY.append(j)
                    data.append(M[i][j])
                if M[i][j]<minVal:
                    minVal=M[i][j]
                if M[i][j]>maxVal:
                    maxVal=M[i][j]
        if colormap is not None and (maxVal-minVal)>abs(tol):
            res=plotaxis.scatter(mX,mY,c=data,alpha=alpha,s=150/math.sqrt(len(mX)),cmap=colormap)
            plotfig.colorbar(res)
        else:
            plotaxis.scatter(mX,mY,color=color,alpha=alpha,s=150/math.sqrt(len(mX)))
    plt.show()    

def jacobian(f,vect,h=1e-6,args=[],kwargs={}):
    #This method approximates the Jacobian of a multidimensional function f with
    #multidimensional input vector vect. Variable h is the step size used to
    #approximate the derivative.
    if ("int" in type(vect).__name__) or ("float" in type(vect).__name__):
        dim=1
    elif ("list" in type(vect).__name__) or ("tuple" in type(vect).__name__) or ("ndarray" in type(vect).__name__):
        dim=len(vect)
    else:
        print("Error: Inputted vector is of incompatible type %s"%type(vect).__name__)
        return [[]]
    
    vect=np.asarray(vect,dtype=np.float)
    
    res=f(vect,*args,**kwargs)
    if ("int" in type(res).__name__) or ("float" in type(res).__name__):
        return np.array([[differentiate(f,vect,h=h,variable_Dim=i,args=args,kwargs=kwargs) for i in range(dim)]])
    elif ("list" in type(res).__name__) or ("tuple" in type(res).__name__) or ("ndarray" in type(res).__name__):
        return np.array([differentiate(f,vect,h=h,variable_Dim=i,args=args,kwargs=kwargs) for i in range(dim)]).T
    else:
        print("Error: Function output is of incompatible type %s"%type(res).__name__)
        return [[]]

def hessian(f,vect,h=1e-3,args=[],kwargs={}):
    #This method approximates the Hessian of a function f with
    #multidimensional input vector vect. Variable h is the step size used to
    #approximate the derivative.
    if ("int" in type(vect).__name__) or ("float" in type(vect).__name__):
        dim=1
    elif ("list" in type(vect).__name__) or ("tuple" in type(vect).__name__) or ("ndarray" in type(vect).__name__):
        dim=len(vect)
    else:
        print("Error: Inputted vector is of incompatible type %s"%type(vect).__name__)
        return [[]]
    
    vect=np.asarray(vect,dtype=np.float)
    
    res=f(vect,*args,**kwargs)
    if ("int" in type(res).__name__) or ("float" in type(res).__name__):
        #H=np.zeros((dim,dim))
        #for i in range(dim):
        #    H[i,i]=double_differentiate(f,vect,h=h,variable_Dim_1=i,variable_Dim_2=i)
        #    for j in range(i+1,dim):
        #        dd=double_differentiate(f,vect,h=h,variable_Dim_1=i,variable_Dim_2=j)
        #        H[i,j]=dd
        #        H[j,i]=dd
        #return H
        return np.array([[double_differentiate(f,vect,h=h,variable_Dim_1=i,variable_Dim_2=j) for j in range(dim)] for i in range(dim)])
    else:
        print("Error: Function output is of incompatible type %s"%type(res).__name__)
        return [[]]

def grad(f,vect,h=1e-6,args=[],kwargs={}):
    vect=np.asarray(vect,dtype=np.float)
    if ("int" in type(vect).__name__) or ("float" in type(vect).__name__):
        dim=1
    elif ("list" in type(vect).__name__) or ("tuple" in type(vect).__name__) or ("ndarray" in type(vect).__name__):
        dim=len(vect)
    else:
        print("Error: Inputted vector is of incompatible type %s"%type(vect).__name__)
        return []
    res=f(vect,*args,**kwargs)
    if ("int" in type(res).__name__) or ("float" in type(res).__name__):
        return np.array([differentiate(f,vect,h=h,variable_Dim=i) for i in range(dim)])
    else:
        print("Error: Function output is of incompatible type %s"%type(res).__name__)
        return []
    
def grammSchmidt(vectors,tol=1e-6,normalize=True,clean=True):
    #This method performs Gramm Schmidt orthogonalization on a set of vectors.
    #Variable tol controls the tolerance for when to a vector orthogonalizable 
    #or not (depends on norm of vector). Variable normalize renormalizes each 
    #other the orthogonalized vectors. Variable clean removes any vectors that 
    #do not meet the orthogonalization tolerance.
    size=len(vectors)
    for i in range(size-1,-1,-1):
        try:
            if not isinstance(vectors[i],np.ndarray):
                vectors[i]=np.array(vectors[i])
        except:
            print("Number %i of the inputted vectors can not be converted to a numpy array"%i)
            return vectors
    
    if any([len(vectors[i])!=len(vectors[i+1]) for i in range(size-1)]):
        print("Not all inputted vectors are the same size")
        return vectors
    dim=len(vectors[0])
    
    orthogonals=[vectors[0]]
    norms=[npla.norm(vectors[0])]
    if norms[0]<tol:
        orthogonals[0]=np.zeros(dim)
        norms[0]=-1
    for i in range(1,size):
        if npla.norm(vectors[i])<tol:
            orthogonals.append(np.zeros(dim))
            norms.append(-1)
        else:
            ortho=vectors[i]-sum([np.inner(orthogonals[j],vectors[i])*orthogonals[j]/norms[j]**2 for j in range(i)])
            norm=npla.norm(ortho)
            if norm>=tol:
                orthogonals.append(ortho)
                norms.append(norm)
            else:
                orthogonals.append(np.zeros(dim))
                norms.append(-1)

    if normalize:
        for i in range(size):
            if norms[i]!=-1:
                orthogonals[i]=orthogonals[i]/norms[i]
            
    if clean:
        for i in range(size-1,-1,-1):
            if norms[i]==-1:
                del orthogonals[i]
            
    return orthogonals

def jacobi(A,b,errtol=1e-6,maxlevel=1000,x0=None,omega=1.0,inform=False):
    #This method implements the (weighted) Jacobi iterative matrix solver for 
    #nonsingular, square matrices. It attempts to solve the equation Ax=b. 
    #Variable errtol controls the tolerance for when approximation is
    #close enough to a solution. Variable maxlevel controls the maximal number
    #of iterations the method can perform. Variable x0 provides an optional 
    #first guess at the solution. Variable omega allows for the optional weighted
    #Jacobi iterative matrix solver, potentially increasing rate of convergence.
    #Variable inform allows the user to receive information about the process
    #(for verification purposes).
    n=A.shape[0]
    if A.shape[0]!=A.shape[1]:
        if inform:
            print("Matrix A is not square, Jacobi can not be applied")
        return np.zeros((A.shape[1],))
    if any([e==0.0 for e in npla.eigvals(A)]):
        if inform:
            print("Matrix A is singular, Jacobi can not be applied")
        return np.zeros((A.shape[1],))
    else:
        if x0!=None:
            x=x0.copy()
        else:
            x=np.array([1.0/n for i in range(n)])
        r=2.0*errtol
        k=0
        while r>errtol and k<maxlevel:
            xtemp=x.copy()
            for i in range(n):
                sigma=0.0
                for j in range(n):
                    if j!=i:
                        sigma+=A[i][j]*xtemp[j]
                x[i]=(1.0-omega)*xtemp[i]+omega*(b[i]-sigma)/A[i][i]
            k+=1
            r=npla.norm(np.dot(A,x)-b)
        if inform:
            print("Jacobi Method Performance Data:")
            if k>=maxlevel:
                print("Number of iterations (Max Number of Iterations Reached): %i" % k)
            else:
                print("Number of iterations: %i" % k)
            print("Error: %.6f" % npla.norm(np.dot(A,x)-b))
        return x

def sor(A,b,errtol=1e-6,maxlevel=1000,x0=None,omega=1.0,inform=False):
    #This method implements the (weighted) Successive Over-Relaxation iterative
    #matrix solver for nonsingular, square matrices. It attempts to solve the 
    #equation Ax=b. Variable errtol controls the tolerance for when 
    #approximation is close enough to a solution. Variable maxlevel controls 
    #the maximal number of iterations the method can perform. Variable x0 provides
    #an optional first guess at the solution. Variable omega allows for the 
    #optional weighted Successive Over-Relaxation iterative matrix solver, 
    #potentially increasing rate of convergence. Variable inform allows the user
    #to receive information about the process (for verification purposes).
    n=A.shape[0]
    if A.shape[0]!=A.shape[1]:
        if inform:
            print("Matrix A is not square, SOR(%0.2f) can not be applied")
        return np.zeros((A.shape[1],))
    if any([e==0.0 for e in npla.eigvals(A)]):
        if inform:
            print("Matrix A is singular, SOR(%0.2f) can not be applied")
        return np.zeros((A.shape[1],))
    else:
        if x0!=None:
            x=x0.copy()
        else:
            x=np.array([1.0/n for i in range(n)])
        r=2.0*errtol
        k=0
        while r>errtol and k<maxlevel:
            for i in range(n):
                sigma=0.0
                for j in range(n):
                    if j!=i:
                        sigma+=A[i][j]*x[j]
                x[i]=(1.0-omega)*x[i]+omega*(b[i]-sigma)/A[i][i]
            k+=1
            r=npla.norm(np.dot(A,x)-b)
        if inform:
            print("SOR(%0.2f) Performance Data:"%omega)
            if k>=maxlevel:
                print("Number of iterations (Max Number of Iterations Reached): %i" % k)
            else:
                print("Number of iterations: %i" % k)
            print("Error: %.6f" % npla.norm(np.dot(A,x)-b))
        return x

def cg(A,b,errtol=1e-6,maxlevel=1000,x0=None,inform=False):
    #This method implements the Conjugate Gradient iterative matrix solver for 
    #Symmetric Positive Definite (SPD) matrices. It attempts to solve the 
    #equation Ax=b. Variable errtol controls the tolerance for when 
    #approximation is close enough to a solution. Variable maxlevel controls 
    #the maximal number of iterations the method can perform. Variable x0 provides
    #an optional first guess at the solution. Variable inform allows the user
    #to receive information about the process (for verification purposes).
    n=A.shape[0]
    spdbool=False
    if (np.array_equal(A,A.T)):
        eigvals=npla.eigvals(A)
        spdbool=all([eigvals[i]>0.0 for i in range(n)])
    if not spdbool:
        if inform:
            print("Matrix A is not SPD, CG can not be applied")
        return np.zeros((A.shape[1],))
    else:
        if x0!=None:
            x=x0.copy()
        else:
            x=np.array([1.0/n for i in range(n)])
        r=b-A.dot(x)
        p=r.copy()
        k=0
        while npla.norm(r)>errtol and k<maxlevel:
            v=A.dot(p)
            alpha=np.inner(r,r)/np.inner(p,v)
            x=x+alpha*p
            r=r-alpha*v
            beta=np.inner(r,r)/np.inner(r+alpha*v,r+alpha+v)
            p=r+beta*p
            k+=1
        if inform:
            print("Conjugate Gradient Method Performance Data:")
            if k >= maxlevel:
                print("Number of iterations (Max Number of Iterations Reached): %i" % k)
            else:
                print("Number of iterations: %i" % k)
            print("Error: %.6f" % npla.norm(np.dot(A, x) - b))
        return x

def linear_system_solve(A,b,zerotol=1e-6,inform=False):
    #This method implements the general Gaussain Elimination solver.
    #It attempts to solve the equation Ax=b. The variable zerotol controls the 
    #tolerance of what is to be considered zero or not. Variable 
    #inform allows the user to receive information about the process 
    #(for verification purposes). Output is given as <(C,(S1,S2,...Sn))>, where
    #the solution for Ax=b can be summed as S1*x1+S2*x2+...Sn*xn+C, where
    #x1,x2,...xn are real numbers.
    A=np.asarray(A,dtype=np.float)
    b=np.asarray(b,dtype=np.float)
    if len(A.shape)==2 and len(b.shape)==1 and A.shape[0]==b.shape[0]:
        Acopy=A.copy()
        bcopy=b.copy()
        n,m=Acopy.shape
        zerotol=abs(zerotol)
        pivots=[]
        for k in range(n):
            pivot=(-1,-1)
            for i in range(k,m):
                if abs(Acopy[k][i])>zerotol:
                    pivot=(k,i)
                    pivots.append(pivot)
                    break
                else:
                    if Acopy[k][i]!=0:
                        Acopy[k][i]=0.0
                    switchedbool=False
                    for j in range(k+1,n):
                        if abs(Acopy[j][i])>zerotol:
                            temp=Acopy[k].copy()
                            Acopy[k]=Acopy[j]
                            Acopy[j]=temp
                            temp=bcopy[k]
                            bcopy[k]=bcopy[j]
                            bcopy[j]=temp
                            pivot=(j,i)
                            pivots.append(pivot)
                            switchedbool=True
                            break
                        elif Acopy[j][i]!=0:
                            Acopy[j][i]=0.0
                    if switchedbool:
                        break
            if pivot[1]!=-1:
                for i in range(k+1,n):
                    if abs(Acopy[i][pivot[1]])>zerotol:
                        res=Acopy[i][pivot[1]]/Acopy[k][pivot[1]]
                        Acopy[i][pivot[1]]=0.0
                        for j in range(pivot[1]+1,m):
                            if abs(Acopy[k][j])>zerotol:
                                Acopy[i][j]=Acopy[i][j]-res*Acopy[k][j]
                                if abs(Acopy[i][j])<=zerotol:
                                    Acopy[i][j]=0.0
                            elif Acopy[k][j]!=0:
                                Acopy[k][j]=0.0
                        bcopy[i]=bcopy[i]-res*bcopy[k]
                        if abs(bcopy[k])<=zerotol:
                            bcopy[k]=0.0
                    elif Acopy[i][pivot[1]]!=0:
                        Acopy[i][pivot[1]]=0.0

        for k in range(pivots[-1][0]+1,n):
            empty=True
            for j in range(k,m):
                if abs(Acopy[k][j])>zerotol:
                    empty=False
                    break
                elif Acopy[k][j]!=0.0:
                    Acopy[k][j]=0.0
            if empty:
                if abs(bcopy[k])>zerotol:
                    if inform:
                        print("Matrix equation does not have a solution")
                    return []
        
        for pivot in pivots[-1::-1]:
            for j in range(pivot[1]+1,m):
                Acopy[pivot[0]][j]=Acopy[pivot[0]][j]/Acopy[pivot[0]][pivot[1]]
                if abs(Acopy[pivot[0]][j])<=zerotol:
                    Acopy[pivot[0]][j]=0.0
            bcopy[pivot[0]]=bcopy[pivot[0]]/Acopy[pivot[0]][pivot[1]]
            if abs(bcopy[pivot[0]])<=zerotol:
                bcopy[pivot[0]]=0.0
            Acopy[pivot[0]][pivot[1]]=1.0
            for i in range(pivot[0]):
                res=Acopy[i][pivot[1]]
                for j in range(pivot[1],m):
                    Acopy[i][j]=Acopy[i][j]-res*Acopy[pivot[0]][j]
                    if abs(Acopy[i][j])<=zerotol:
                        Acopy[i][j]=0.0
                bcopy[i]=bcopy[i]-res*bcopy[pivot[0]]
                if abs(bcopy[i])<=zerotol:
                    bcopy[i]=0.0
            
        if inform:
            print("Matrix Reduced Echelon Form:")
            print(Acopy)
            print()
            print("Vector Reduced Echelon Form:")
            print(bcopy)
            print()
        
        c_vector=np.zeros(m,dtype=np.float)
        for pivot in pivots:
            c_vector[pivot[1]]=bcopy[pivot[0]]
        free_vectors=[]
        for p in range(len(pivots)-1):
            for i in range(pivots[p][1]+1,pivots[p+1][1]):
                res=np.zeros(m,dtype=np.float)
                for pivot in pivots:
                    res[pivot[1]]=-Acopy[pivot[0]][i]
                res[i]=1.0
                free_vectors.append(res)
        for i in range(pivots[-1][1]+1,min(n,m)):
            res=np.zeros(m,dtype=np.float)
            for pivot in pivots:
                res[pivot[1]]=-Acopy[pivot[0]][i]
            res[i]=1.0
            free_vectors.append(res)
        return (c_vector,tuple(free_vectors))
    else:
        if inform:
            print("Matrix A or vector b not of the right shape; can not perform LU decomposition")
        return None
    
def lu_symbolic(A,b,inform=False):
    #This method implements the symbolic LU Decomposition matrix solver.
    #It attempts to solve the equation Ax=b. Variable inform allows the
    #user to receive information about the process (for verification purposes).
    A=np.asarray(A,dtype=np.float)
    b=np.asarray(b,dtype=np.float)
    if len(A.shape)==2 and len(b.shape)==1 and A.shape[0]==b.shape[0]:
        if A.shape[0]>A.shape[1] and inform:
            print("More equations than variables; reducing number of equations")
        Acopy=A[:min(A.shape[0],A.shape[1])].copy()
        bcopy=b[:min(A.shape[0],A.shape[1])].copy()
        n,m=Acopy.shape
        pivots=[]
        
        from sympy import eye
        from sympy import zeros
        from sympy import simplify
        #from sympy import Matrix
        L=eye(n)
        for k in range(n-1):
            pivot=-1
            for i in range(k,m):
                if Acopy[k][i]!=0.0:
                    pivot=i
                    break
                else:
                    switchedbool=False
                    for j in range(k+1,n):
                        if Acopy[j][i]!=0.0:
                            temp=Acopy[k].copy()
                            Acopy[k]=Acopy[j]
                            Acopy[j]=temp
                            temp=bcopy[k]
                            bcopy[k]=bcopy[j]
                            bcopy[j]=temp
                            temp=L[k,:]
                            L[k,:]=L[j,:]
                            L[j,:]=temp
                            temp=L[:,k]
                            L[:,k]=L[:,j]
                            L[:,j]=temp
                            pivot=i
                            switchedbool=True
                            break
                    if switchedbool:
                        break
            if pivot!=-1:
                pivots.append(pivot)
                for i in range(k+1,n):
                    if Acopy[i][pivot]!=0.0:
                        L[i,pivot]=simplify(Acopy[i][pivot]/Acopy[k][pivot])
                        Acopy[i][pivot]=0.0
                        for j in range(pivot+1,m):
                            if Acopy[k][j]!=0.0:
                                Acopy[i][j]=simplify(Acopy[i][j]-L[i,k]*Acopy[k][j])
        for i in range(n-1,m):
            if Acopy[n-1][i]!=0.0:
                pivots.append(n-1)
                break
        if inform:
            free=[i for i in range(m) if i not in pivots]
            print("Lower Triangular Matrix P*L:")
            for i in range(n):
                if i==0:
                    print("[["+"  ".join([str(L[i,j]) for j in range(m)])+"]")
                elif i==n-1:
                    print(" ["+"  ".join([str(L[i,j]) for j in range(m)])+"]]")
                else:
                    print(" ["+"  ".join([str(L[i,j]) for j in range(m)])+"]")
            print()
            print("Upper Triangular Matrix U:")
            print(Acopy)
            print()
            print("Vector P*b:")
            print(bcopy)
            print()
            if len(free)>0:
                print("System has potentially infinite solutions; choosing one solution")
            
        y=zeros(1,n)
        for i in range(n):
            y[i]=simplify((bcopy[i]-sum([L[i,j]*y[j] for j in range(i)]))/L[i,i])
           
        for k in range(n-1,-1,-1):
            pivot=-1
            empty=True
            for j in range(k,m):
                if Acopy[k][j]!=0.0:
                    pivot=j
                    empty=False
                    break
            if empty:
                if y[i]!=0.0:
                    if inform:
                        print("Matrix equation does not have a solution")
                    return []
            if pivot!=-1:
                for j in range(pivot+1,m):
                    Acopy[k][j]=Acopy[k][j]/Acopy[k][pivot]
                y[k]=y[k]/Acopy[k][pivot]
                Acopy[k][pivot]=1.0
                for i in range(k):
                    temp=Acopy[i][pivot]
                    for j in range(pivot,m):
                        Acopy[i][j]=Acopy[i][j]-temp*Acopy[k][j]
                    y[i]=y[i]-temp*y[k]
        if len(pivots)<m:
            for i in range(len(pivots),m):
                Acopy[i][i]=-1.0
            return (y,tuple([-Acopy[:,j] for j in range(len(pivots),m)]))
        else:
            return y
        
        """
        x=zeros(1,m)
        for i in range(n,m):
            x[i]=1.0
        for i in range(n-1,-1,-1):
            pivot=-1
            empty=True
            for j in range(i,m):
                if Acopy[i][j]!=0.0:
                    if j!=i:
                        x[i]=1.0
                    pivot=j
                    empty=False
                    break
                elif Acopy[i][j]!=0.0:
                    Acopy[i][j]=0.0
            if empty:
                if y[i]==0.0:
                    continue
                else:
                    if inform:
                        print("Matrix equation does not have a solution")
                    return []
            if pivot!=-1:
                x[pivot]=simplify((y[i]-sum([Acopy[i][j]*x[j] for j in range(pivot+1,m)]))/Acopy[i][pivot])
        error=simplify(Matrix([sum([A[i][j]*x[j] for j in range(len(A[i]))])-b[i] for i in range(len(A))]).norm())
        if error!=0.0:
            if inform:
                print("Matrix equation may not have a solution; please check yourself")
                print("L Error: %s" % simplify(Matrix([sum([L[i,j]*y[j] for j in range(L.shape[0])])-bcopy[i] for i in range(L.shape[1])]).norm()))
                print("U Error: %s" % simplify(Matrix([sum([Acopy[i,j]*x[j] for j in range(Acopy.shape[1])])-y[i] for i in range(Acopy.shape[0])]).norm()))
                print("Total Error: %s" % error)
            return x
        if inform:
            print("L Error: %s" % simplify(Matrix([sum([L[i,j]*y[j] for j in range(L.shape[0])])-bcopy[i] for i in range(L.shape[1])]).norm()))
            print("U Error: %s" % simplify(Matrix([sum([Acopy[i,j]*x[j] for j in range(Acopy.shape[1])])-y[i] for i in range(Acopy.shape[0])]).norm()))
            print("Total Error: %s" % error)
        return x
        """
    else:
        if inform:
            print("Matrix A or vector b not of the right shape; can not perform LU decomposition")
        return []
    
def lu(A,b,symbol=False,sparse=False):
    if symbol:
        if sparse:
            from sympy.matrices import SparseMatrix
            if not "SparseMatrix" in type(A).__name__:
                A=SparseMatrix(A)
            if not "SparseMatrix" in type(b).__name__:
                b=SparseMatrix(b)
            return A.solve(b)
        else:
            from sympy import Matrix
            if not "DenseMatrix" in type(A).__name__:
                A=Matrix(A)
            if not "DenseMatrix" in type(b).__name__:
                b=Matrix(b)
            return A.LUsolve(b)
    else:
        if sparse:
            from scarse.linalg import spsolve
            if not "csc_matrix" in type(A).__name__:
                A=scarse.csc_matrix(A)
            if not "csc_matrix" in type(b).__name__:
                b=scarse.csc_matrix(b)
            return spsolve(A,b)
        else:
            from scipy.linalg import lu_factor, lu_solve
            lu,piv=lu_factor(A)
            return lu_solve((lu,piv),b,check_finite=True)
        
def nullspace(A,symbol=False,sparse=False):
    if symbol:
        if sparse:
            from sympy.matrices import SparseMatrix
            if not "SparseMatrix" in type(A).__name__:
                A=SparseMatrix(A)
            return A.solve(SparseMatrix([0 for row in A.rows]))
        else:
            from sympy import Matrix
            if not "DenseMatrix" in type(A).__name__:
                A=Matrix(A)
            return A.nullspace()
    else:
        if sparse:
            from scarse.linalg import spsolve
            if not "csc_matrix" in type(A).__name__:
                A=scarse.csc_matrix(A)
            return spsolve(A,scarse.csc_matrix([0 for row in A.shape[0]]))
        else:
            from scipy.linalg import null_space
            return null_space(A)

def powerMethod(A,tol=1e-6,max_iter=100,x0=None,report=False):
    n=A.shape[0]
    if x0!=None:
        x=x0.copy()
    else:
        x=np.random.rand(A.shape[0])
    lmbda=0
    r=2*tol
    k=0
    while r>tol and k<max_iter:
        y=A.dot(x)
        x=y/npla.norm(y)
        lmbda=np.inner(np.conjugate(x),y)
        r=npla.norm((A-lmbda*np.identity(n)).dot(x))
        k+=1
    if report:
        print("Power Method Performance Data:")
        if k >= max_iter:
            print("Number of iterations (Max Number of Iterations Reached): %i" % k)
        else:
            print("Number of iterations: %i" % k)
        print("Error: %.6f" % r)
    return lmbda,x
