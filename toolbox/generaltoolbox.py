import math
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.colors as pltcolors
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("error")

graphsize=9
font = {"family": "serif",
    "color": "black",
    "weight": "bold",
    "size": "20"}
        
def decimal_to_fraction(x,n=2):
    #This method converts a decimal x rounded off to the nth decimal place
    #into a list containing it's numerator and denominator (in that order). 
    if not isinstance(n,int):
        n=int(n)
    if x==0.0:
        num=0
        den=1
    else:
        x=round(x,n)
        num=round(x*10**n)
        den=10**n
        
        c2=0
        while round(num)%2==0 and c2<n:
            num/=2
            den/=2
            c2+=1
        c5=0
        while round(num)%5==0 and c5<n:
            num/=5
            den/=5
            c5+=1
    return [int(num),int(den)]

def colors(N,plotly=False):
    #This method producesa list of N evenly-spaced color strings used for plotting.
    #The plotly option converts those colors to rgb strings compatible with 
    #plotly plotting methods.
    N=int(round(N))
    if N<=0:
        return []
    elif N==1:
        if plotly:
            return ["rgb(0,0,0)"]
        else:
            return ["black"]
    elif N==2:
        if plotly:
            return ["rgb(255,0,0)","rgb(0,0,255)"]
        else:
            return ["red","blue"]
    elif N==3:
        if plotly:
            return ["rgb(255,0,0)","rgb(0,255,0)","rgb(0,0,255)"]
        else:
            return ["red","green","blue"]
    elif N==4:
        if plotly:
            return ["rgb(255,0,0)","rgb(255,255,0)","rgb(0,255,255)","rgb(0,255)"]
        else:
            return ["red","yellow","cyan","blue"]
    else:
        res=[]
        for i in range(N):
            col=i/(N-1)
            if 0<=col<0.2:
                #red-yellow
                if plotly:
                    res.append("rgb(255,"+str(round(255*5*col))+",0)")
                else:
                    res.append([1,5*col,0])
            elif 0.2<=col<0.4:
                #yellow-green
                if plotly:
                    res.append("rgb("+str(round(255*(1-5*(col-0.2))))+",255,0)")
                else:
                    res.append([1-5*(col-0.2),1,0])
            elif 0.4<=col<0.6:
                #green-cyan
                if plotly:
                    res.append("rgb(0,255,"+str(round(255*5*(col-0.4)))+")")
                else:
                    res.append([0,1,5*(col-0.4)])
            elif 0.6<=col<0.8:
                #cyan-blue
                if plotly:
                    res.append("rgb(0,"+str(round(255*(1-5*(col-0.6))))+",255)")
                else:
                    res.append([0,1-5*(col-0.6),1])
            elif 0.8<=col<=1:
                #blue-magenta
                if plotly:
                    res.append("rgb("+str(round(255*5*(col-0.8)))+",0,255)")
                else:
                    res.append([5*(col-0.8),0,1])
        return res

def hammersley(N,dim,points=True):
    #This method produces N psuedo-random Hammersley points of dimension dim.
    #The points option selects how the points are returned: either as a list
    #of points, or as a list of coordinate lists.
    primes=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137]
    B=primes[:dim]
    dim=len(B)
    def Phi(k,b):
        bprime=b
        kprime=k
        phi=0
        while kprime>0:
            a=float(kprime%b)
            phi+=a/bprime
            kprime=int(kprime/b)
            bprime=bprime*b
        return phi
    if points:
        return [np.array([n/N]+[Phi(n,B[i-1]) for i in range(1,dim)]) for n in range(N+1)]
    else:
        return [np.array([n/N for n in range(N+1)])]+[np.array([Phi(n,B[i-1]) for n in range(N+1)]) for i in range(1,dim)]

def differentiate(f,x,args=[],kwargs={},h=1e-3,variableDim=0):
    #This method approximates the (partial) derivative of function f:R^n->R
    #at point x with respect the the (variableDim)th variable,
    #using a 5-point-stencil whenever possible.
    
    if ("int" in type(x).__name__) or ("float" in type(x).__name__):
        x2hminus=x-2*h
        xhminus=x-h
        xhplus=x+h
        x2hplus=x+2*h
    elif ("list" in type(x).__name__):
        variableDim=max(min(variableDim,len(x)),0)
        x2hminus=x[:variableDim]+[x[variableDim]-2*h]+x[variableDim+1:]
        xhminus=x[:variableDim]+[x[variableDim]-h]+x[variableDim+1:]
        xhplus=x[:variableDim]+[x[variableDim]+h]+x[variableDim+1:]
        x2hplus=x[:variableDim]+[x[variableDim]+2*h]+x[variableDim+1:]
    elif ("tuple" in type(x).__name__):
        variableDim=max(min(variableDim,len(x)),0)
        x2hminus=x[:variableDim]+tuple([x[variableDim]-2*h])+x[variableDim+1:]
        xhminus=x[:variableDim]+tuple([x[variableDim]-h])+x[variableDim+1:]
        xhplus=x[:variableDim]+tuple([x[variableDim]+h])+x[variableDim+1:]
        x2hplus=x[:variableDim]+tuple([x[variableDim]+2*h])+x[variableDim+1:]
    elif ("ndarray" in type(x).__name__):
        variableDim=max(min(variableDim,len(x)),0)
        x2hminus=np.concatenate([x[:variableDim],[x[variableDim]-2*h],x[variableDim+1:]])
        xhminus=np.concatenate([x[:variableDim],[x[variableDim]-h],x[variableDim+1:]])
        xhplus=np.concatenate([x[:variableDim],[x[variableDim]+h],x[variableDim+1:]])
        x2hplus=np.concatenate([x[:variableDim],[x[variableDim]+2*h],x[variableDim+1:]])
    else:
        print("Error: point x is of an incompatible type %s"%type(x).__name__)
        return None
    
    try:
        fx=f(x,*args,**kwargs)
    except:
        print(x)
        print("Error: Function is not defined on point x")
        return None
    
    try:
        fx2hminus=f(x2hminus,*args,**kwargs)
        fx2hplus=f(x2hplus,*args,**kwargs)
        try:
            fxhminus=f(xhminus,*args,**kwargs)
            fxhplus=f(xhplus,*args,**kwargs)
            return (-fx2hplus+8*fxhplus-8*fxhminus+fx2hminus)/(12*h)
        except:
            return (fx2hplus-fx2hminus)/(4*h)
    except:
        try:
            fxhminus=f(xhminus,*args,**kwargs)
            try:
                fxhplus=f(xhplus,*args,**kwargs)
                return (fxhplus-fxhminus)/(2*h)
            except:
                return (fx-fxhminus)/h
        except:
            try:
                fxhplus=f(xhplus,*args,**kwargs)
                return (fxhplus-fx)/h
            except:
                print("Error: Function can not be differentiated in point ["+", ".join([str(elem) for elem in x])+"]")
                return None

def integrate(f,start,stop,args=[],kwargs={},mode="gauss",maxlevel=5,errtol=1e-3,adapt=True):
    #This method approximates the integral of a function f:R->R between the lower bound
    #start and upper bound stop. The integral can be computed using either 
    #the adaptive (G30,K61) Gauss-Kronrod quadrature or the adaptive trapezium method.
    #The maxlevel variable determines how many times the adaptive method can
    #iterate, while the errtol variable provides an additional stopping
    #criterion.
    maxlevel=int(round(maxlevel))
    if (("int" not in type(start).__name__) and ("float" not in type(start).__name__)) or (("int" not in type(stop).__name__) and ("float" not in type(stop).__name__)):
        print("Error: inputted upper bound b or lower bound a are of incompatible type")
        print("Type of lower bound a: %s"%type(start).__name__)
        print("Type of upper bound b: %s"%type(stop).__name__)
        return None
    
    start=min(start,stop)
    stop=max(start,stop)
    
    if start==stop:
        return 0.0
    elif "gauss" in mode.lower() or "kronrod" in mode.lower():
        F=0.0
        xg=[-0.9968934840746495,-0.9836681232797472,-0.9600218649683075,
            -0.9262000474292743,-0.8825605357920527,-0.8295657623827684,
            -0.7677774321048262,-0.6978504947933158,-0.6205261829892429,
            -0.5366241481420199,-0.44703376953808915,-0.3527047255308781,
            -0.25463692616788985,-0.15386991360858354,-0.0514718425553177,
            0.0514718425553177,0.15386991360858354,0.25463692616788985,
            0.3527047255308781,0.44703376953808915,0.5366241481420199,
            0.6205261829892429,0.6978504947933158,0.7677774321048262,
            0.8295657623827684,0.8825605357920527,0.9262000474292743,
            0.9600218649683075,0.9836681232797472,0.9968934840746495]
        wg=[0.007968192496166605,0.01846646831109096,0.02878470788332337,
            0.03879919256962705,0.04840267283059405,0.057493156217619065,
            0.06597422988218049,0.0737559747377052,0.08075589522942021,
            0.08689978720108298,0.09212252223778612,0.09636873717464425,
            0.09959342058679527,0.1017623897484055,0.10285265289355884,
            0.10285265289355884,0.1017623897484055,0.09959342058679527,
            0.09636873717464425,0.09212252223778612,0.08689978720108298,
            0.08075589522942021,0.0737559747377052,0.06597422988218049,
            0.057493156217619065,0.04840267283059405,0.03879919256962705,
            0.02878470788332337,0.01846646831109096,0.007968192496166605]
        xk=[-0.9994844100504906,-0.9968934840746495,-0.9916309968704046,
            -0.9836681232797472,-0.9731163225011262,-0.9600218649683075,
            -0.94437444474856,-0.9262000474292743,-0.9055733076999078,
            -0.8825605357920527,-0.8572052335460612,-0.8295657623827684,
            -0.799727835821839,-0.7677774321048262,-0.7337900624532268,
            -0.6978504947933158,-0.6600610641266269,-0.6205261829892429,
            -0.5793452358263617,-0.5366241481420199,-0.49248046786177857,
            -0.44703376953808915,-0.4004012548303944,-0.3527047255308781,
            -0.30407320227362505,-0.25463692616788985,-0.20452511668230988,
            -0.15386991360858354,-0.10280693796673702,-0.0514718425553177,
            0.0,0.0514718425553177,0.10280693796673702,0.15386991360858354,
            0.20452511668230988,0.25463692616788985,0.30407320227362505,
            0.3527047255308781,0.4004012548303944,0.44703376953808915,
            0.49248046786177857,0.5366241481420199,0.5793452358263617,
            0.6205261829892429,0.6600610641266269,0.6978504947933158,
            0.7337900624532268,0.7677774321048262,0.799727835821839,
            0.8295657623827684,0.8572052335460612,0.8825605357920527,
            0.9055733076999078,0.9262000474292743,0.94437444474856,
            0.9600218649683075,0.9731163225011262,0.9836681232797472,
            0.9916309968704046,0.9968934840746495,0.9994844100504906]
        wk=[0.0013890136986770077,0.003890461127099884,0.0066307039159312926,
            0.009273279659517764,0.011823015253496341,0.014369729507045804,
            0.01692088918905327,0.019414141193942382,0.021828035821609193,
            0.0241911620780806,0.0265099548823331,0.02875404876504129,
            0.030907257562387762,0.03298144705748372,0.034979338028060025,
            0.03688236465182123,0.038678945624727595,0.040374538951535956,
            0.041969810215164244,0.04345253970135607,0.04481480013316266,
            0.04605923827100699,0.04718554656929915,0.04818586175708713,
            0.04905543455502978,0.04979568342707421,0.05040592140278235,
            0.05088179589874961,0.051221547849258774,0.05142612853745902,
            0.05149472942945157,0.05142612853745902,0.051221547849258774,
            0.05088179589874961,0.05040592140278235,0.04979568342707421,
            0.04905543455502978,0.04818586175708713,0.04718554656929915,
            0.04605923827100699,0.04481480013316266,0.04345253970135607,
            0.041969810215164244,0.040374538951535956,0.038678945624727595,
            0.03688236465182123,0.034979338028060025,0.03298144705748372,
            0.030907257562387762,0.02875404876504129,0.0265099548823331,
            0.0241911620780806,0.021828035821609193,0.019414141193942382,
            0.01692088918905327,0.014369729507045804,0.011823015253496341,
            0.009273279659517764,0.0066307039159312926,0.003890461127099884,
            0.0013890136986770077]
        
        gauss=lambda start,stop:0.5*(stop-start)*sum([wg[i]*f(0.5*((stop-start)*xg[i]+(stop+start)),*args,**kwargs) for i in range(len(wg))])
        kronrod=lambda start,stop:0.5*(stop-start)*sum([wk[i]*f(0.5*((stop-start)*xk[i]+(stop+start)),*args,**kwargs) for i in range(len(wk))])
        if adapt:
            intervals=[[start,stop]]
            intervalErr=[abs(kronrod(intervals[0][0],intervals[0][1])
                            -gauss(intervals[0][0],intervals[0][1]))]
            totalErr=intervalErr[0]
            intlength=abs(intervals[0][1]-intervals[0][0])
            iteratebool=True
            count=0
            while iteratebool and count<=round(2**maxlevel):
                if totalErr>errtol:
                    splitintervalindex=[]
                    for i in range(len(intervals)):
                        if math.log(intlength/abs(intervals[i][1]-intervals[i][0]),2)<maxlevel:
                            splitintervalindex.append(i)
                    if len(splitintervalindex)>0:
                        i=splitintervalindex[0]
                        for j in range(1,len(splitintervalindex)):
                            if intervalErr[splitintervalindex[j]]>intervalErr[i]:
                                i=splitintervalindex[j]
                        aInterval=intervals[i][0]
                        bInterval=(intervals[i][1]+intervals[i][0])/2
                        cInterval=intervals[i][1]
                        del intervals[i]
                        del intervalErr[i]
                        intervals.append([aInterval,bInterval])
                        intervals.append([bInterval,cInterval])
                        intervalErr.append(abs(kronrod(aInterval,bInterval)-gauss(aInterval,bInterval)))
                        intervalErr.append(abs(kronrod(bInterval,cInterval)-gauss(bInterval,cInterval)))
                        totalErr=sum(intervalErr)
                    else:
                        iteratebool=False
                else:
                    iteratebool=False
                count+=1
            F=sum([kronrod(interval[0],interval[1]) for interval in intervals])
        else:
            F=kronrod(start,stop)
        return F
    else:
        F=0.0
        trapezoid=lambda x1,x2:0.5*(f(x2,*args,**kwargs)+f(x1,*args,**kwargs))*(x2-x1)
        if adapt:
            intervals=[[start,stop]]
            intervalErr=[abs(trapezoid(0.5*(intervals[0][1]+intervals[0][0]),intervals[0][1])
                            +trapezoid(intervals[0][0],0.5*(intervals[0][1]+intervals[0][0]))
                            -trapezoid(intervals[0][0],intervals[0][1]))]
            totalErr=intervalErr[0]
            intlength=abs(intervals[0][1]-intervals[0][0])
            iteratebool=True
            count=0
            while iteratebool and count<=round(2**maxlevel):
                if totalErr>errtol:
                    splitintervalindex=[]
                    for i in range(len(intervals)):
                        if math.log(intlength/abs(intervals[i][1]-intervals[i][0]),2)<maxlevel:
                            splitintervalindex.append(i)
                    if len(splitintervalindex)>0:
                        i=splitintervalindex[0]
                        for j in range(1,len(splitintervalindex)):
                            if intervalErr[splitintervalindex[j]]>intervalErr[i]:
                                i=splitintervalindex[j]
                        aInterval=intervals[i][0]
                        bInterval=0.5*(intervals[i][1]+intervals[i][0])
                        cInterval=intervals[i][1]
                        del intervals[i]
                        del intervalErr[i]
                        intervals.append([aInterval,bInterval])
                        intervals.append([bInterval,cInterval])
                        intervalErr.append(abs(trapezoid(aInterval,0.5*(bInterval+aInterval))+trapezoid(0.5*(bInterval+aInterval),bInterval)-trapezoid(aInterval,bInterval)))
                        intervalErr.append(abs(trapezoid(bInterval,0.5*(cInterval+bInterval))+trapezoid(0.5*(cInterval+bInterval),cInterval)-trapezoid(bInterval,cInterval)))
                        totalErr=sum(intervalErr)
                    else:
                        iteratebool=False
                else:
                    iteratebool=False
                count+=1
            F=sum([trapezoid(interval[0],interval[1]) for interval in intervals])
        else:
            for i in range(100):
                F+=trapezoid(start+i*(stop-start)/100,start+(i+1)*(stop-start)/100)
        return F

def multiIntegrate(f,start,stop,args=[],kwargs={},N=1e3):
    #This method approximates the integral of a function f:R^n->R between the lower bound
    #vector start and upper bound B. The integral is computed using the quasi-random
    #Monte Carlo integration, with the MonteCarloPoints variable accounting for
    #the precision of the approximation.
    if (("list" in type(start).__name__) or ("ndarray" in type(start).__name__)) and (("list" in type(stop).__name__) or ("ndarray" in type(stop).__name__)):
        if len(start)==len(stop):
            dim=len(start)
            if dim>0 and all([("int" in type(a).__name__) or ("float" in type(a).__name__) for a in start]) and all([("int" in type(b).__name__) or ("float" in type(b).__name__) for b in stop]):
                for i in range(dim):
                    start[i]=min(start[i],stop[i])
                    stop[i]=max(start[i],stop[i])
            else:
                print("Error: inputted upper bound vector stop or lower bound start contains incompatible element types")
                print("Types in lower bound start: %s"%", ".join([type(a).__name__ for a in start]))
                print("Types in upper bound stop: %s"%", ".join([type(b).__name__ for b in stop]))
                return None
        else:
            print("Error: Lower bound vector start and upper bound vector stop are of different sizes")
            return None
    else:
        print("Error: inputted upper bound vector stop or lower bound start are of incompatible type")
        print("Type of lower bound start: %s"%type(start).__name__)
        print("Type of upper bound stop: %s"%type(stop).__name__)
        return None
    
    if ("int" not in type(N).__name__) and ("float" not in type(N).__name__):
        print("Error: N is of incompatible type")
        print("Type of N: %s"%type(N).__name__)
        return None
    
    N=int(round(N))
    npA=np.array(start)
    npB=np.array(stop)
    Points=hammersley(N,dim)
    return np.prod(npB-npA)*sum([f(npA+Points[i]*(npB-npA),*args,**kwargs) for i in range(N)])/N

def interpolate(sol,t,N):
    #This method interpolates a set of data (sol,t) into N temporally
    #evenly-spaced data points.
    dim=len(sol)
    n=len(t)
    if any(len(sol[i])!=n for i in range(dim)):
        print("Error: check the length of your inputted vectors")
        for i in range(dim):
            print("Length of sol[%i]: %i"%(i,len(sol[i])))
        print("Length of t: %i"%len(t))
        return [], []
    
    tprime=np.linspace(t[0],t[n-1],N)
    solprime=[[sol[i][0]]+[0 for j in range(N-1)] for i in range(dim)]
    index=0
    for i in range(1,N):
        for j in range(index,n):
            if abs(t[j]-tprime[i])<=1e-6:
                index=j
                for k in range(dim):
                    solprime[k][i]=sol[k][index]
                break
            if abs(t[j]-tprime[i-1])>=abs(tprime[i]-tprime[i-1]):
                index=j
                for k in range(dim):
                    if n>2:
                        if j==n-1:
                            solprime[k][i]=(sol[k][j-2]*(tprime[i]-t[j-1])*(tprime[i]-t[j])/((t[j-2]-t[j-1])*(t[j-2]-t[j]))
                                            +sol[k][j-1]*(tprime[i]-t[j-2])*(tprime[i]-t[j])/((t[j-1]-t[j-2])*(t[j-1]-t[j]))
                                            +sol[k][j]*(tprime[i]-t[j-2])*(tprime[i]-t[j-1])/((t[j]-t[j-2])*(t[j]-t[j-1])))
                        else:
                            solprime[k][i]=(sol[k][j-1]*(tprime[i]-t[j])*(tprime[i]-t[j+1])/((t[j-1]-t[j])*(t[j-1]-t[j+1]))
                                            +sol[k][j]*(tprime[i]-t[j-1])*(tprime[i]-t[j+1])/((t[j]-t[j-1])*(t[j]-t[j+1]))
                                            +sol[k][j+1]*(tprime[i]-t[j-1])*(tprime[i]-t[j])/((t[j+1]-t[j-1])*(t[j+1]-t[j])))
                    else:
                        solprime[k][i]=sol[k][j-1]*(t[j]-tprime[i])/(t[j]-t[j-1])+sol[k][j]*(tprime[i]-t[j-1])/(t[j]-t[j-1])
                break
    return solprime,tprime

def spline(sol,t_array):
    #This method constructs a spline function that smoothly interpolates
    #between a number of inputted points. The variable t_array is the 
    #collection of temporal independent variables, while the variable sol
    #is the collection of corresponding dependent vectors.
    if len(t_array)==len(sol):
        N=len(t_array)
        a=np.asarray(sol)
        h=np.array([t_array[i+1]-t_array[i] for i in range(N-1)])
        alpha=np.array([3*(a[i+2]-a[i+1])/h[i+1]-3*(a[i+1]-a[i])/h[i] for i in range(N-2)])
        l,mu,z=np.ones(N),np.zeros(N),np.zeros(N)
        for i in range(1,N-1):
            l[i]=2*h[i]+(2-mu[i-1])*h[i-1]
            mu[i]=h[i]/l[i]
            z[i]=(alpha[i-1]-h[i-1]*z[i-1])/l[i]
        b=np.zeros(N-1)
        c=np.zeros(N)
        d=np.zeros(N)
        for i in range(N-2,-1,-1):
            c[i]=z[i]-mu[i]*c[i+1]
            b[i]=(a[i+1]-a[i])/h[i]-h[i]*(c[i+1]+2*c[i])/3
            d[i]=(c[i+1]-c[i])/(3*h[i])
        
        def spline_func(t):
            for i in range(N):
                if t_array[i]<=t<=t_array[i+1]:
                    return a[i]+b[i]*(t-t_array[i])+c[i]*(t-t_array[i])**2+d[i]*(t-t_array[i])**3
        return spline_func
    else:
        print("Error: inputted vectors do not have matching lengths")
        print("Length of sol: %i"%len(sol))
        print("Length of sol: %i"%len(t_array))
        return None

def line_search(f,start,stop,args=[],kwargs={},errtol=1e-6,maxlevel=100,inform=False):
    #This method performs a line search of a function f:R->R to find the
    #minimum of f between variables start and stop using the Brent minimization
    #method, which itself is a hybrid of Golden Section Search and Jarrat
    #iteration. It is guaranteed to converge, and usually does super-linearly.
    #Variable errtol sets the minimum distance between start and stop before an
    #approximate of the minimum is made. Variable maxlevel determines the
    #maximum number of iterations the method can perform before it is forced to
    #end. If variable inform is set to true, the method relays information on 
    #it's converge behavior to the user. 
    if stop<start:
        temp=start
        start=stop
        stop=temp
    phi=(1+math.sqrt(5))/2
    x=stop-(phi-1)*(stop-start)
    fx=f(x,*args,**kwargs)
    v=x
    w=x
    fv=fx
    fw=fx
    d=0
    e=0
    m=(start+stop)/2
    count=1
    if inform:
        F=[f(m,*args,**kwargs)]
    while stop-start>errtol and count<maxlevel:
        r=(x-w)*(fx-fv)
        tq=(x-v)*(fx-fw)
        tp=(x-v)*tq-(x-w)*r
        tq2=2*(tq-r)
        if tq2>0:
            p=-tp
            q=tq2
        else:
            p=tp
            q=-tq2
        safe=(q!=0.0)
        if safe:
            try:
                deltax=p/q
            except Exception:
                deltax=0.0
        else:
            deltax=0.0
        parabolic=(safe and (start<x+deltax<stop) and (abs(deltax)<abs(e)/2))
        if parabolic:
            e=d
            d=deltax
        else:
            if x<m:
                e=stop-x
            else:
                e=start-x
            d=(2-phi)*e
        u=x+d
        fu=f(u,*args,**kwargs)
        if fu<=fx:
            if u<x:
                stop=x
            else:
                start=x
            v=w
            w=x
            x=u
            fv=fw
            fw=fx
            fx=fu
        else:
            if u<x:
                start=u
            else:
                stop=u
            if fu<=fw or w==x:
                v=w
                w=u
                fv=fw
                fw=fu
            elif fu<=fv or v==x or v==w:
                v=u
                fv=fu
        m=(start+stop)/2
        if inform:
            F.append(f(m,*args,**kwargs))
        count+=1
    if stop-start>errtol:
        if inform:
            print("Unable to find minimum after %i iterations"%count)
        return None
    else:    
        if inform:
            print("Minimum found after %i iterations"%count)
            print("f(x) = %0.6f"%F[-1])
            ax=plt.figure(figsize=(graphsize,graphsize)).add_subplot(111)
            ax.plot([i for i in range(len(F))],F)
            ax.set_title("Function Value per Iteration",fontdict=font)
            ax.set_xlabel("Iteration",fontsize=16,rotation=0)
            ax.set_ylabel("Function Value",fontsize=16)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
        return m

def newton_Raphson(f,x0,args=[],kwargs={},errtol=1e-6,maxlevel=100,mode="bad_broyden",adapt=True,inform=False):
    #This method performs a Newton-Raphson approximation of the root of the 
    #function f:R->R, using scalar x0 as it's initial guess.
    #Variable errtol sets the minimum norm that the function can take on before
    #an approximate of the root is made. Variable maxlevel determines the
    #maximum number of iterations the method can perform before it is forced to
    #end. If variable mode is set to newton_raphson, the inverse Jacobian is
    #computed every iteration. If variable mode is set to broyden (default),
    #the inverse jacobian is approximated using the Broyden method. If variable
    #adapt is set to True, the method uses the weak Wolfe conditions to damp
    #(or accelerate) the Newton-Raphson iteration to coerce global converge to
    #a minimum or root. If variable inform is set to true, the method relays
    #information on it's converge behavior to the user. 
    
    f0=f(x0,*args,**kwargs)
    nf0=abs(f0)
    if ("int" in type(x0).__name__) or ("float" in type(x0).__name__):
        if ("int" not in type(f0).__name__) and ("float" not in type(f0).__name__):
            if inform:
                print("Error: unable to apply Newton-Raphson iteration to this function")
            return None
    else:
        if inform:
            print("Error: unable to apply Newton-Raphson iteration to this function")
        return None
        
    c1=1e-4
    c2=0.9
    try:
        dfinv=1/differentiate(f,x0,args=args,kwargs=kwargs)
    except Exception:
        dfinv=0
    x1=x0-dfinv*f0
    f1=f(x1,*args,**kwargs)
    nf1=abs(f1)
    alpha=1.0
    
    mode=mode.lower()
    if "broyden" in mode:
        mode="broyden"
    else:
        mode="newton_raphson"
    
    newt_count=1
    if inform:
        F=[nf0,nf1]
    
    while nf1>errtol and abs(x1-x0)>errtol and newt_count<maxlevel:
        if "broyden" in mode:
            try:
                dfinv=(x1-x0)/(f1-f0)
            except Exception:
                try:
                    dfinv=1/differentiate(f,x1,args=args,kwargs=kwargs)
                except Exception:
                    dfinv=0
        else:
            try:
                dfinv=1/differentiate(f,x1,args=args,kwargs=kwargs)
            except Exception:
                try:
                    dfinv=(x1-x0)/(f1-f0)
                except Exception:
                    dfinv=0
        delta_x=-dfinv*f1
        x0=x1
        f0=f1
        nf0=nf1
        
        x1=x1+alpha*delta_x
        f1=f(x1,*args,**kwargs)
        nf1=abs(f1)
        if adapt and abs(delta_x)>0:
            nf=lambda x:abs(f(x,*args,**kwargs))
            grad_nf0=differentiate(nf,x0)
            a=0
            b=np.inf
            bisect_count=0
            wolfe_flag=False
            double_flag=True
            while not wolfe_flag and bisect_count<100:
                if nf(x0+alpha*delta_x)>nf0+c1*alpha*delta_x*grad_nf0:
                    b=alpha
                    alpha=0.5*(a+b)
                    double_flag=False
                elif delta_x*differentiate(nf,x0+alpha*delta_x)<c2*delta_x*grad_nf0:
                    a=alpha
                    if b<np.inf:
                        alpha=0.5*(a+b)
                    else:
                        alpha=2*a
                    double_flag=False
                else:
                    if double_flag and alpha<=1:
                        alpha*=2
                    else:
                        wolfe_flag=True
                bisect_count+=1
            x1=x0+alpha*delta_x
            f1=f(x1,*args,**kwargs)
            nf1=abs(f1)
        if inform:
            F.append(nf1)
        newt_count+=1

    if nf1>errtol and abs(x1-x0)>errtol:
        if inform:
            print("Unable to find extremum or minimum after %i iterations"%newt_count)
            ax=plt.figure(figsize=(graphsize,graphsize)).add_subplot(111)
            ax.plot([i for i in range(len(F))],F)
            ax.set_title("Function Norm per Iteration",fontdict=font)
            ax.set_xlabel("Iteration",fontsize=16,rotation=0)
            ax.set_ylabel("Function Norm",fontsize=16)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
        return None
    else:
        if inform:
            if nf1<=errtol:
                print("Root found after %i iterations"%newt_count)
                print("|f(x)| = %0.6f"%nf1)
            else:
                print("Minumum found after %i iterations"%newt_count)
                print("|f(x)| = %0.6f"%nf1)
            ax=plt.figure(figsize=(graphsize,graphsize)).add_subplot(111)
            ax.plot([i for i in range(len(F))],F)
            ax.set_title("Function Norm per Iteration",fontdict=font)
            ax.set_xlabel("Iteration",fontsize=16,rotation=0)
            ax.set_ylabel("Function Norm",fontsize=16)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
        return x1

def multi_Newton_Raphson(f,x0,args=[],kwargs={},errtol=1e-6,maxlevel=100,mode="bad_broyden",adapt=True,inform=False):
    #This method performs a Newton-Raphson approximation of the root of the 
    #function f:R^n->R^n, using vector x0 as it's initial guess.
    #Variable errtol sets the minimum norm that the function can take on before
    #an approximate of the root is made. Variable maxlevel determines the
    #maximum number of iterations the method can perform before it is forced to
    #end. If variable mode is set to newton_raphson, the inverse Jacobian is
    #computed every iteration. If variable mode is set to good_broyden, the
    #inverse jacobian is approximated using the good Broyden method. If
    #variable mode is set to bad_broyden (default), the inverse jacobian is 
    #approximated using the bad Broyden method. If variable adapt is set to
    #True, the method uses the weak Wolfe conditions to damp (or accelerate)
    #the Newton-Raphson iteration to coerce global converge to a minimum or root.
    #If variable inform is set to true, the method relays information on 
    #it's converge behavior to the user. 
    
    x0=np.asarray(x0)
    f0=f(x0,*args,**kwargs)
    nf0=npla.norm(f0)
    
    if ("list" in type(x0).__name__) or ("tuple" in type(x0).__name__) or ("ndarray" in type(x0).__name__):
        if ("list" in type(x0).__name__) or ("tuple" in type(x0).__name__) or ("ndarray" in type(x0).__name__):
            if len(x0)==len(f0):
                dim=len(x0)
                from toolbox.matrixtoolbox import jacobian,grad
            else:
                if inform:
                    print("Error: unable to apply Newton-Raphson iteration to f:R^%i->R^%i"%(len(x0),len(f0)))
                return None
        else:
            if inform:
                print("Error: unable to apply Newton-Raphson iteration to this function")
            return None
    else:
        if inform:
            print("Error: unable to apply Newton-Raphson iteration to this function")
        return None
    
    c1=1e-4
    c2=0.9
    try:
        dfinv=npla.inv(jacobian(f,x0,args=args,kwargs=kwargs))
    except Exception:
        dfinv=np.zeros(dim,dim)
    x1=x0-np.dot(dfinv,f0)
    f1=f(x1,*args,**kwargs)
    nf1=npla.norm(f1)
    alpha=1.0
    
    mode=mode.lower()
    if "broyden" in mode:
        if "g" in mode:
            mode="good_broyden"
        else:
            mode="bad_broyden"
    else:
        mode="newton_raphson"
    
    newt_count=1
    if inform:
        F=[nf0,nf1]
    
    while nf1>errtol and npla.norm(x1-x0)>errtol and newt_count<maxlevel:
        if mode=="bad_broyden":
            try:
                dfinv=dfinv+np.outer(((x1-x0)-np.dot(dfinv,f1-f0))/sum((f1-f0)**2),f1-f0)
            except Exception:
                try:
                    dfinv=dfinv+np.outer(((x1-x0)-np.dot(dfinv,f1-f0))/np.dot(x1-x0,np.dot(dfinv,f1-f0)),np.dot(x1-x0,dfinv))
                except Exception:
                    try:
                        dfinv=npla.inv(jacobian(f,x0,args=args,kwargs=kwargs))
                    except Exception:
                        dfinv=np.zeros((dim,dim))
        elif mode=="good_broyden":
            try:
                dfinv=dfinv+np.outer(((x1-x0)-np.dot(dfinv,f1-f0))/np.dot(x1-x0,np.dot(dfinv,f1-f0)),np.dot(x1-x0,dfinv))
            except Exception:
                try:
                    dfinv=dfinv+np.outer(((x1-x0)-np.dot(dfinv,f1-f0))/sum((f1-f0)**2),f1-f0)
                except Exception:
                    try:
                        dfinv=npla.inv(jacobian(f,x0,args=args,kwargs=kwargs))
                    except Exception:
                        dfinv=np.zeros((dim,dim))
        else:
            try:
                dfinv=npla.inv(jacobian(f,x0,args=args,kwargs=kwargs))
            except Exception:
                try:
                    dfinv=dfinv+np.outer(((x1-x0)-np.dot(dfinv,f1-f0))/sum((f1-f0)**2),f1-f0)
                except Exception:
                    try:
                        dfinv=dfinv+np.outer(((x1-x0)-np.dot(dfinv,f1-f0))/np.dot(x1-x0,np.dot(dfinv,f1-f0)),np.dot(x1-x0,dfinv))
                    except Exception:
                        dfinv=np.zeros((dim,dim))

        delta_x=-np.dot(dfinv,f1)
        x0=x1.copy()
        f0=f1.copy()
        nf0=nf1.copy()
        x1=x1+alpha*delta_x
        f1=f(x1,*args,**kwargs)
        nf1=npla.norm(f1)
        if adapt and npla.norm(delta_x)>0:
            nf=lambda x:npla.norm(f(x,*args,**kwargs))
            grad_nf0=grad(nf,x0)
            a=0
            b=np.inf
            bisect_count=0
            wolfe_flag=False
            double_flag=True
            while not wolfe_flag and bisect_count<100:
                if nf(x0+alpha*delta_x)>nf0+c1*alpha*np.dot(delta_x,grad_nf0):
                    b=alpha
                    alpha=0.5*(a+b)
                    double_flag=False
                elif np.dot(delta_x,grad(nf,x0+alpha*delta_x))<c2*np.dot(delta_x,grad_nf0):
                    a=alpha
                    if b<np.inf:
                        alpha=0.5*(a+b)
                    else:
                        alpha=2*a
                    double_flag=False
                else:
                    if double_flag and alpha<=1:
                        alpha*=2
                    else:
                        wolfe_flag=True
                bisect_count+=1
            x1=x0+alpha*delta_x
            f1=f(x1,*args,**kwargs)
            nf1=npla.norm(f1)
        if inform:
            F.append(nf1)
        newt_count+=1
    if nf1>errtol and npla.norm(x1-x0)>errtol:
        if inform:
            print("Unable to find extremum or minimum after %i iterations"%newt_count)
            ax=plt.figure(figsize=(graphsize,graphsize)).add_subplot(111)
            ax.plot([i for i in range(len(F))],F)
            ax.set_title("Function Norm per Iteration",fontdict=font)
            ax.set_xlabel("Iteration",fontsize=16,rotation=0)
            ax.set_ylabel("Function Norm",fontsize=16)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
        return None
    else:
        if inform:
            if nf1<=errtol:
                print("Root found after %i iterations"%newt_count)
                print("||f(x)||_2 = %0.6f"%nf1)
            else:
                print("Minumum found after %i iterations"%newt_count)
                print("||f(x)||_2 = %0.6f"%nf1)
            ax=plt.figure(figsize=(graphsize,graphsize)).add_subplot(111)
            ax.plot([i for i in range(len(F))],F)
            ax.set_title("Function Norm per Iteration",fontdict=font)
            ax.set_xlabel("Iteration",fontsize=16,rotation=0)
            ax.set_ylabel("Function Norm",fontsize=16)
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
        return x1

def polyRegression(X,Y,dim=1,errtol=1e-6,plotbool=True,plotaxis=None,color="black",alpha=1.0):
    if dim>len(X):
        dim=len(X)-1
    plotshowbool=False
    if plotbool and plotaxis is None:
        plotfig=plt.figure(figsize=(graphsize,graphsize))
        plotaxis=plotfig.add_subplot(111)
        plotaxis.set_title("Data Best Fit Line",fontdict=font)
        plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
        plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
        plotaxis.xaxis.set_tick_params(labelsize=16)
        plotaxis.yaxis.set_tick_params(labelsize=16)
        plotshowbool=True
        
    duplicates=[]
    for i in range(len(X)-1):
        for j in range(i+1,len(X)):
            if abs(X[i]-X[j])<=abs(errtol):
                if j not in duplicates:
                    duplicates.append(j)
    for ind in duplicates[::-1]:
        del X[ind]
        
    M=np.array([[x**j for j in range(dim+1)] for x in X])
    coeff=npla.solve(np.dot(M.T,M),np.dot(M.T,Y))
    
    if plotbool:
        plotaxis.scatter(X,Y,color="black",alpha=1.0)
        minX=min(X)
        maxX=max(X)
        Xplot=np.linspace(minX,maxX,100)
        plotaxis.plot(Xplot,[sum([coeff[j]*x**j for j in range(dim+1)]) for x in Xplot],color=color,alpha=alpha)
        if plotshowbool:
            plt.show()
    return coeff

def pointplot(f,start,stop,delta,args=[],kwargs={},adapt=True,maxlevel=10,mode="linear",plotbool=True,plotaxis=None,colormap=None,color="black",alpha=1.0):
    #This method approximates the 0D solutions of f(x)=0, using a 
    #"marching-line" algorithm, which is a 1D form of the marching cube algorithm.
    #The zeros are found between scalar lower bound start and scalar upper bound
    #stop, and divided into subsections of length delta. The method can subdivide
    #even further through variable adapt. The method then bisects each subsection
    #and only focusses on subsections that, as best as the method can tell, do 
    #not contain any solutions. The subsections then form a structure similar to
    #a sparse bitree. The maximum level of subdivisions is determined by variable
    #maxlevel. After subdivision, the zeros are approximated using either linear,
    #quadratic, or cubic interpolation, determined by variable mode. A colormap option
    #is available to when plotting (colormap values are based on the x-coordinate).
    plotshowbool=False
    if plotbool:
        if plotaxis is None:
            plotfig=plt.figure(figsize=(graphsize,graphsize))
            plotaxis=plotfig.add_subplot(111)
            plotaxis.set_title("Function Point",fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotshowbool=True
        xlim = plotaxis.get_xlim()
        ylim = plotaxis.get_ylim()
        if xlim != (0.0, 1.0):
            plotaxis.set_xlim([min(xlim[0],start),max(xlim[1],stop)])
        else:
            plotaxis.set_xlim([start,stop])
        if ylim != (0.0, 1.0):
            plotaxis.set_ylim([min(ylim[0],-1),max(ylim[1],1)])
        else:
            plotaxis.set_ylim([-1, 1])
    
    mode=mode.lower()
    
    if color is not None:
        try:
            color = pltcolors.to_rgb(color)
        except ValueError:
            color = (0.0, 0.0, 0.0)
    if colormap is not None:
        try:
            pltcolors.to_rgb(colormap(0.0))
            pltcolors.to_rgb(colormap(0.5))
            pltcolors.to_rgb(colormap(1.0))
        except Exception:
            colormap = None
    
    paths={0:[[]],
           1:[[0,1]],
           2:[[0,1]],
           3:[[]]}
    
    class line:
        def __init__(self,Sv1,delta,f):
            self.delta=delta
            self.V=[Sv1,Sv1+delta]
            self.c=0.5*(self.V[0]+self.V[1])
            self.err=1e-3*(self.V[1]-self.V[0])
            self.zero=0.0#1e-6*self.delta
            self.f=f
            self.F=[0 for i in range(2)]
            self.fc=0
            self.full=False
            self.done=False
            self.Px=[]
        
        def adjust(self, a, b=None, terminate=True):
            shift = 100
            i = 0
            if b is None:
                b = self.c
            nanbool = True
            while nanbool and i <= shift:
                try:
                    tempa = a + i * (b - a) / shift
                    tempfa = f(tempa, *args, **kwargs)
                    nanbool = math.isinf(tempfa) or math.isnan(tempfa)
                except Exception:
                    nanbool = True
                i += 1
            if nanbool:
                tempa = a
                tempfa = 0
                self.done = terminate
            return tempa, tempfa
        
        def intersect(self,a,b,fa,fb,P,mode):
            if mode.lower() in ["quad","quadratic","2","two","second"]:
                c,fc=self.adjust(self.c,b=self.V[0])
                A=((b-c)*fa+(c-a)*fb+(a-b)*fc)/((a-b)*(a-c)*(b-c))
                B=((c**2-b**2)*fa+(a**2-c**2)*fb+(b**2-a**2)*fc)/((a-b)*(a-c)*(b-c))
                C=((b-c)*b*c*fa+(c-a)*c*a*fb+(a-b)*a*b*fc)/((a-b)*(a-c)*(b-c))
                R=np.roots([A,B,C])
                res=0
                rcount=0
                for r in R:
                    if abs(r.imag)<=self.zero:
                        if abs(a-r.real)<=self.err:
                            res+=a
                            rcount+=1
                        elif abs(b-r.real)<=self.err:
                            res+=b
                            rcount+=1
                        else:
                            if a<r.real<b:
                                res+=r.real
                                rcount+=1
                            elif a>r.real>b:
                                res+=r.real
                                rcount+=1
                if rcount>0:
                    P.append(res/rcount)
                    
            elif mode.lower() in ["cube","cubic","3","three","third"]:
                h=1e-3*self.delta
                aplus=a+h
                aminus=a-h
                bplus=b+h
                bminus=b-h
                
                aplus,faplus=self.adjust(aplus)
                aminus,faminus=self.adjust(aminus)
                bplus,fbplus=self.adjust(bplus)
                bminus,fbminus=self.adjust(bminus)
                
                faprime=(faplus-faminus)/(2*h)
                fbprime=(fbplus-fbminus)/(2*h)
                    
                A=(a*faprime+a*fbprime-b*faprime-b*fbprime-2*fa+2*fb)/((a-b)**3)
                B=(-a**2*faprime-2*a**2*fbprime-a*b*faprime+a*b*fbprime+3*a*fa-3*a*fb+2*b**2*faprime+b**2*fbprime+3*b*fa-3*b*fb)/((a-b)**3)
                C=(a**3*fbprime+2*a**2*b*faprime+a**2*b*fbprime-a*b**2*faprime-2*a*b**2*fbprime-6*a*b*fa+6*a*b*fb-b**3*faprime)/((a-b)**3)
                D=(-a**3*b*fbprime+a**3*fb-a**2*b**2*faprime+a**2*b**2*fbprime-3*a**2*b*fb+a*b**3*faprime+3*a*b**2*fa-b**3*fa)/((a-b)**3)
                R=np.roots([A,B,C,D])
                res=0
                rcount=0
                for r in R:
                    if abs(r.imag)<=self.zero:
                        if abs(a-r.real)<=self.err:
                            res+=a
                            rcount+=1
                        elif abs(b-r.real)<=self.err:
                            res+=b
                            rcount+=1
                        else:
                            if a<r.real<b:
                                res+=r.real
                                rcount+=1
                            elif a>r.real>b:
                                res+=r.real
                                rcount+=1
                if rcount>0:
                    P.append(res/rcount)
            else:
                temp=(a*fb-b*fa)/(fb-fa)
                P.append(temp)
        
        def calc(self):
            for i in range(2):
                v,fv=self.adjust(self.V[i])
                self.F[i]=fv
            self.c,self.fc=self.adjust(self.c,b=self.V[0])
            self.full=True
            for fv in self.F:
                if abs(fv)>self.zero:
                    self.full=False
                    break
            
            if not self.full:
                if all([fv<0 for fv in self.F]) or all([fv>0 for fv in self.F]):
                    self.done=True
                    for i in range(2):
                        if self.F[i]*self.fc<0:
                            self.done=False
                            break

        def is_straight(self):
            diff=np.zeros(len(self.V))
            for i in range(len(self.V)):
                plus,fplus=self.adjust(self.V[i]+1e-3*self.delta,
                                       b=self.V[i]+self.delta,terminate=False)
                minus,fminus=self.adjust(self.V[i]-1e-3*self.delta,
                                         b=self.V[i]-self.delta,terminate=False)
                diff[i]=(fplus-fminus)/(plus-minus)
            if (diff[1]-diff[0])**2<=(0.25)**2:
                return True
            elif math.isinf(diff[1]-diff[0]) or math.isnan(diff[1]-diff[0]):
                return True
            return False
        
        def split(self,S):
            if not self.done:
                if not self.full and not self.is_straight():
                    splitS=[]
                    for s in [line(0.5*(self.V[0]+self.V[i]),self.delta/2,f) for i in range(2)]:
                        s.calc()
                        if not s.done:
                            splitS.append(s)
                    S+=splitS
                else:
                    S+=[self]
                
        def findpoints(self):
            if not self.done:
                if not self.full:
                    index=0
                    for i in range(2):
                        if self.F[i]<=self.zero:
                            index+=2**(1-i)
                    path=paths[index]
                    for p in path:
                        if len(p)>1:
                            self.intersect(self.V[0],self.V[1],self.F[0],self.F[1],self.Px,mode)

    if adapt:
        delta=max((stop-start)/10,delta)
    else:
        delta=max((stop-start)/128,delta)        
    
    Lv1=[]
    for i in range(int(round((stop-start)/delta))):
        Lv1.append(start+delta*i)
    
    L=[line(l,delta,f) for l in Lv1]
    for l in L:
        l.calc()

    if adapt:
        for i in range(maxlevel-1):
            newL=[]
            for l in L:
                l.split(newL)
            L=newL.copy()
            
    for l in L:
        l.findpoints()
            
    Px=[]
    for l in L:
        for i in range(len(l.Px)):
            matchbool=False
            for j in range(len(Px)):
                if abs(l.Px[i]-Px[j])<l.zero:
                    matchbool=True
                    break
            if not matchbool:
                Px.append(l.Px[i])    
    
    if plotbool:
        segments=[]
        colors=[]
        for l in L:
            if not l.done and l.full:
                segments.append(np.column_stack([line.vertices,[0,0]]))
                if colormap is not None:
                    colors.append(colormap((0.5*(line.vertices[1]+line.vertices[0])-start)/(stop-start)))
        if len(segments) > 0:
            if colormap is not None:
                plotaxis.add_collection(LineCollection(segments,colors=colors))
            else:
                plotaxis.add_collection(LineCollection(segments,colors=color))
        if len(Px) > 0:
            if colormap is not None:
                plotaxis.scatter(Px,[0 for elem in Px],color=[colormap((elem-start)/(stop-start)) for elem in Px],alpha=alpha)
            else:
                plotaxis.scatter(Px,[0 for elem in Px],color=color,alpha=alpha)
        if plotshowbool:
            plt.show()
    else:
        return Px

def plot2D(f,start,stop,delta,args=[],kwargs={},limit=[None,None],dependentvar="y",mode="cartesian",plotaxis=None,color="black",alpha=1.0):
    #This method provides a consistent way to plot any f:R->R function 
    #(cartesian or polar) between scalar lower bound start and scalar upper bound
    #stop, using delta as a step variable. The variable limit provides a 
    #lower and upper limit on the value of f. The variable dependentvar determines
    #which variable is a function of the other. For example, if the dependentvar
    #is y, then y(x)=f(x).
    plotshowbool=False
    if plotaxis is None:
        plotfig=plt.figure(figsize=(graphsize,graphsize))
        if "pol" in mode.lower():
            plotaxis=plotfig.add_subplot(111,projection="polar")
        else:
            plotaxis=plotfig.add_subplot(111)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
        plotaxis.set_title("Function Graph",fontdict=font)
        plotaxis.xaxis.set_tick_params(labelsize=16)
        plotaxis.yaxis.set_tick_params(labelsize=16)
        plotshowbool=True
        
    Xgrid=np.linspace(start,stop,round((stop-start)/delta)+1)
    Ygrid=np.zeros(Xgrid.shape)
    
    for i in range(len(Xgrid)):
        try:
            res=f(Xgrid[i],*args,**kwargs)
        except ZeroDivisionError:
            neighbors=[]
            if i>0:
                try:
                    neighbors.append(f(Xgrid[i-1],*args,**kwargs))
                except:
                    pass
            if i<len(Xgrid)-1:
                try:
                    neighbors.append(f(Xgrid[i+1],*args,**kwargs))
                except:
                    pass
            if len(neighbors)>0:
                res=sum(neighbors)/len(neighbors)
            else:
                res=0
        Ygrid[i]=res
    
    if "x" in dependentvar.lower():
        plotaxis.plot(Ygrid,Xgrid,color=color,alpha=alpha)
    else:
        plotaxis.plot(Xgrid,Ygrid,color=color,alpha=alpha)
    if plotshowbool:
        plt.show()
                
def lineplot(f,start,stop,delta,args=[],kwargs={},adapt=True,maxlevel=5,mode="linear",plotbool=True,plotaxis=None,scatterplot=False,colormap=None,color="black",alpha=1.0):
    #This method approximates the 1D solutions of f(x,y)=0, using the marching 
    #square algorithm. The zeros are found between lower bound vector start and 
    #upper bound vector stop, and divided into subsections of dimensions delta. 
    #The method can subdivide even further through variable adapt. The method 
    #then evenly bisects each subsection through each dimension and only focusses
    #on subsections that, as best as the method can tell, do not contain any 
    #solutions. The subsections then form a structure similar to a sparse 
    #quadtree. The maximum level of subdivisions is determined by variable
    #maxlevel. After subdivision, the zeros are approximated using either linear,
    #quadratic, or cubic interpolation, determined by variable mode. If plotted,
    #solutions are plotted either using a 2D wireframe (mostly just consisting
    #of straight lines) or by just plotting the point-cloud. A colormap option
    #is available to when plotting (colormap values are based on the y-coordinate).
    plotshowbool=False
    if plotbool:
        if plotaxis is None:
            plotfig=plt.figure(figsize=(graphsize,graphsize))
            plotaxis=plotfig.add_subplot(111)
            plotaxis.set_title("Function Line",fontdict=font)
            plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
            plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
            plotaxis.xaxis.set_tick_params(labelsize=16)
            plotaxis.yaxis.set_tick_params(labelsize=16)
            plotshowbool=True
        xlim = plotaxis.get_xlim()
        ylim = plotaxis.get_ylim()
        if xlim != (0.0, 1.0):
            plotaxis.set_xlim([min(xlim[0], start[0]), max(xlim[1], stop[0])])
        else:
            plotaxis.set_xlim([start[0], stop[0]])
        if ylim != (0.0, 1.0):
            plotaxis.set_ylim([min(ylim[0], start[1]), max(ylim[1], stop[1])])
        else:
            plotaxis.set_ylim([start[1], stop[1]])
            
    mode = mode.lower()

    if color is not None:
        try:
            color = pltcolors.to_rgb(color)
        except ValueError:
            color = (0.0, 0.0, 0.0)
    if colormap is not None:
        try:
            pltcolors.to_rgb(colormap(0.0))
            pltcolors.to_rgb(colormap(0.5))
            pltcolors.to_rgb(colormap(1.0))
        except Exception:
            colormap = None
        
    paths={0:[[]],
           1:[[2, 3]],
           2:[[1, 2]],
           3:[[1, 3]],
           4:[[0, 1]],
           5:[[0, 1], [2, 3]],
           6:[[0, 2]],
           7:[[0, 3]],
           8:[[0, 3]],
           9:[[0, 2]],
           10:[[0, 3], [1, 2]],
           11:[[0, 1]],
           12:[[1, 3]],
           13:[[1, 2]],
           14:[[2, 3]],
           15:[[]]}
    
    class square:
        def __init__(self,Sv1,deltax,deltay,f):
            if not isinstance(Sv1,np.ndarray):
                Sv1=np.array(Sv1)
            self.deltax=deltax
            self.deltay=deltay
            self.V=[Sv1,Sv1+np.array([deltax,0]),Sv1+np.array([deltax,deltay]),Sv1+np.array([0,deltay])]
            self.c=0.5*(self.V[0]+self.V[2])
            self.errx=1e-3*(self.V[2][0]-self.V[0][0])
            self.erry=1e-3*(self.V[2][1]-self.V[0][1])
            self.zero=1e-12#1e-6*self.deltax*self.deltay
            self.f=f
            self.F=[0 for i in range(4)]
            self.fc=0
            self.Px=[]
            self.Py=[]
            self.full=False
            self.done=False
            
        def adjust(self, a, b=None, terminate=True):
            shift = 100
            i = 0
            if b is None:
                b = self.c
            nanbool = True
            while nanbool and i <= shift:
                try:
                    tempa = a + i * (b - a) / shift
                    tempfa = self.f(tempa, *args, **kwargs)
                    nanbool = math.isinf(tempfa) or math.isnan(tempfa)
                except Exception:
                    nanbool = True
                i += 1
            if nanbool:
                tempa = a.copy()
                tempfa = 0
                self.done = terminate
            return tempa, tempfa
        
        def intersect(self,a,b,fa,fb,Px,Py,mode="linear"):
            if mode in ["quad","quadratic","2","two","second"]:
                if abs(a[0]-b[0])<=self.errx:
                    c=np.array([a[0],0.5*(a[1]+b[1])])
                    pivot=1
                else:
                    c=np.array([0.5*(a[0]+b[0]),a[1]])
                    pivot=0
                c,fc=self.adjust(c,b=self.V[0])
                A=((b[pivot]-c[pivot])*fa+(c[pivot]-a[pivot])*fb+(a[pivot]-b[pivot])*fc)/((a[pivot]-b[pivot])*(a[pivot]-c[pivot])*(b[pivot]-c[pivot]))
                B=((c[pivot]**2-b[pivot]**2)*fa+(a[pivot]**2-c[pivot]**2)*fb+(b[pivot]**2-a[pivot]**2)*fc)/((a[pivot]-b[pivot])*(a[pivot]-c[pivot])*(b[pivot]-c[pivot]))
                C=((b[pivot]-c[pivot])*b[pivot]*c[pivot]*fa+(c[pivot]-a[pivot])*c[pivot]*a[pivot]*fb+(a[pivot]-b[pivot])*a[pivot]*b[pivot]*fc)/((a[pivot]-b[pivot])*(a[pivot]-c[pivot])*(b[pivot]-c[pivot]))
                R=np.roots([A,B,C])
                res=0
                rcount=0
                for r in R:
                    if abs(r.imag)<=self.zero:
                        if pivot==0:
                            err=self.errx
                        else:
                            err=self.erry
                        if abs(a[pivot]-r.real)<=err:
                            res+=a[pivot]
                            rcount+=1
                        elif abs(b[pivot]-r.real)<=err:
                            res+=b[pivot]
                            rcount+=1
                        else:
                            if a[pivot]<=b[pivot]:
                                if a[pivot]<r.real<b[pivot]:
                                    res+=r.real
                                    rcount+=1
                            else:
                                if a[pivot]>r.real>b[pivot]:
                                    res+=r.real
                                    rcount+=1
                if rcount>0:
                    if pivot==0:
                        Px.append(res/rcount)
                        Py.append(a[1])
                    else:
                        Px.append(a[0])
                        Py.append(res/rcount)
            elif mode in ["cube","cubic","3","three","third"]:
                if abs(a[0]-b[0])<=self.errx:
                    h=1e-3*self.deltay
                    aplus=a+np.array([0,h])
                    aminus=a-np.array([0,h])
                    bplus=b+np.array([0,h])
                    bminus=b-np.array([0,h])
                    pivot=1
                else:
                    h=1e-3*self.deltax
                    aplus=a+np.array([h,0])
                    aminus=a-np.array([h,0])
                    bplus=b+np.array([h,0])
                    bminus=b-np.array([h,0])
                    pivot=0
                
                aplus,faplus=self.adjust(aplus)
                aminus,faminus=self.adjust(aminus)
                bplus,fbplus=self.adjust(bplus)
                bminus,fbminus=self.adjust(bminus)
                
                faprime=(faplus-faminus)/(2*h)
                fbprime=(fbplus-fbminus)/(2*h)
                    
                A=(a[pivot]*faprime+a[pivot]*fbprime-b[pivot]*faprime-b[pivot]*fbprime-2*fa+2*fb)/((a[pivot]-b[pivot])**3)
                B=(-a[pivot]**2*faprime-2*a[pivot]**2*fbprime-a[pivot]*b[pivot]*faprime+a[pivot]*b[pivot]*fbprime+3*a[pivot]*fa-3*a[pivot]*fb+2*b[pivot]**2*faprime+b[pivot]**2*fbprime+3*b[pivot]*fa-3*b[pivot]*fb)/((a[pivot]-b[pivot])**3)
                C=(a[pivot]**3*fbprime+2*a[pivot]**2*b[pivot]*faprime+a[pivot]**2*b[pivot]*fbprime-a[pivot]*b[pivot]**2*faprime-2*a[pivot]*b[pivot]**2*fbprime-6*a[pivot]*b[pivot]*fa+6*a[pivot]*b[pivot]*fb-b[pivot]**3*faprime)/((a[pivot]-b[pivot])**3)
                D=(-a[pivot]**3*b[pivot]*fbprime+a[pivot]**3*fb-a[pivot]**2*b[pivot]**2*faprime+a[pivot]**2*b[pivot]**2*fbprime-3*a[pivot]**2*b[pivot]*fb+a[pivot]*b[pivot]**3*faprime+3*a[pivot]*b[pivot]**2*fa-b[pivot]**3*fa)/((a[pivot]-b[pivot])**3)
                R=np.roots([A,B,C,D])
                res=0
                rcount=0
                for r in R:
                    if abs(r.imag)<=self.zero:
                        if pivot==0:
                            err=self.errx
                        else:
                            err=self.erry
                        if abs(a[pivot]-r.real)<=err:
                            res+=a[pivot]
                            rcount+=1
                        elif abs(b[pivot]-r.real)<=err:
                            res+=b[pivot]
                            rcount+=1
                        else:
                            if a[pivot]<=b[pivot]:
                                if a[pivot]<r.real<b[pivot]:
                                    res+=r.real
                                    rcount+=1
                            else:
                                if a[pivot]>r.real>b[pivot]:
                                    res+=r.real
                                    rcount+=1
                if rcount>0:
                    if pivot==0:
                        Px.append(res/rcount)
                        Py.append(a[1])
                    elif pivot==1:
                        Px.append(a[0])
                        Py.append(res/rcount)
            else:
                temp=(a*fb-b*fa)/(fb-fa)
                Px.append(temp[0])
                Py.append(temp[1])
        
        def calc(self):
            for i in range(4):
                v,fv=self.adjust(self.V[i])
                self.F[i]=fv
            self.c,self.fc=self.adjust(self.c,b=self.V[0])
            self.full=True
            for fv in self.F:
                if abs(fv)>self.zero:
                    self.full=False
                    break
            
            if not self.full:
                if all([fv<0 for fv in self.F]) or all([fv>0 for fv in self.F]):
                    self.done=True
                    for i in range(4):
                        if i<3:
                            mid=0.5*(self.V[i]+self.V[i+1])
                        else:
                            mid=0.5*(self.V[0]+self.V[3])
                        mid,fm=self.adjust(mid,terminate=False)
                        if fm*self.fc<0:
                            self.done=False
                            break        
                
        def is_straight(self):
            diffx=np.zeros(len(self.V))
            diffy=np.zeros(len(self.V))
            for i in range(len(self.V)):
                plus,fplus=self.adjust(self.V[i]+np.array([1e-3*self.deltax,0]),
                                       b=self.V[i]+np.array([self.deltax,0]),terminate=False)
                minus,fminus=self.adjust(self.V[i]-np.array([1e-3*self.deltax,0]),
                                         b=self.V[i]-np.array([self.deltax,0]),terminate=False)
                diffx[i]=(fplus-fminus)/(plus[0]-minus[0])
                
                plus,fplus=self.adjust(self.V[i]+np.array([0,1e-3*self.deltay]),
                                       b=self.V[i]+np.array([0,self.deltay]),terminate=False)
                minus,fminus=self.adjust(self.V[i]-np.array([0,1e-3*self.deltay]),
                                         b=self.V[i]-np.array([0,self.deltay]),terminate=False)
                diffy[i]=(fplus-fminus)/(plus[1]-minus[1])
                
            if all([(diffx[i]-diffx[0])**2+(diffy[i]-diffy[0])**2<=2*(0.25)**2 for i in range(1,len(self.V))]):
                return True
            elif any([math.isinf(diffx[i] - diffx[0]) or math.isinf(diffy[i] - diffy[0]) or 
                      math.isnan(diffx[i] - diffx[0]) or math.isnan(diffy[i] - diffy[0])
                      for i in range(1,len(self.V))]):
                return True
            return False
                
        def split(self,S):
            if not self.done:
                if not self.full and not self.is_straight():
                    splitS=[]
                    for s in [square(0.5*(self.V[0]+self.V[i]),self.deltax/2,self.deltay/2,f) for i in range(4)]:
                        s.calc()
                        if not s.done:
                            splitS.append(s)
                    S+=splitS
                else:
                    S+=[self]
        
        def findpoints(self):
            if not self.done and not self.full:
                index=0
                for i in range(4):
                    if self.F[i]<=self.zero:
                        index+=2**(3-i)
                if index==5:
                    if self.fc<=self.zero:
                        index=10
                elif index==10:
                    if self.fc<=self.zero:
                        index=5
                path=paths[index]
                for p in path:
                    if len(p)>1:
                        for e in p:
                            if 0<=e<=2:
                                self.intersect(self.V[e],self.V[e+1],self.F[e],self.F[e+1],self.Px,self.Py,mode=mode)
                            else:
                                self.intersect(self.V[3],self.V[0],self.F[3],self.F[0],self.Px,self.Py,mode=mode)
        
        def draw(self,segments,colors):
            if not self.done:
                if self.full:
                    segments.append(np.column_stack([[self.V[i][0] for i in range(4)]+[self.V[0][0]],
                                                     [self.V[i][1] for i in range(4)]+[self.V[0][1]]]))
                    if colormap is not None:
                        colors.append(colormap((0.5*(self.vertices[0][1]+self.vertices[2][1])-start[1])/(stop[1]-start[1])))
                else:
                    if len(self.Px)>0:
                        segments.append(np.column_stack([self.Px, self.Py]))
                        if colormap is not None:
                            colors.append(colormap((sum(self.Py)/len(self.Py)-start[1])/(stop[1]-start[1])))
                        
    if adapt:
        delta=[max((stop[i]-start[i])/10,delta[i]) for i in range(2)]
    else:
        delta=[max((stop[i]-start[i])/128,delta[i]) for i in range(2)]        
    
    Sv1=[]
    for i in range(round((stop[0]-start[0])/delta[0])):
        for j in range(round((stop[1]-start[1])/delta[1])):
            Sv1.append(np.array([start[0]+delta[0]*i,start[1]+delta[1]*j]))
    
    S=[square(s,delta[0],delta[1],f) for s in Sv1]
    for s in S:
        s.calc()

    if adapt:
        for i in range(maxlevel-1):
            newS=[]
            for s in S:
                s.split(newS)
            S=[s for s in newS]
    
    for s in S:
        s.findpoints()
        
    if plotbool:
        if not scatterplot:
            segments = []
            colors = []
            for s in S:
                s.draw(segments, colors)
            if colormap is not None:
                plotaxis.add_collection(art3d.LineCollection(segments,colors=colors))
            else:
                plotaxis.add_collection(art3d.LineCollection(segments,colors=color))
        else:
            Px,Py=[],[]
            for s in S:
                for i in range(len(s.Px)):
                    matchbool=False
                    for j in range(len(Px)):
                        if abs(s.Px[i]-Px[j])<=s.zero and abs(s.Py[i]-Py[j])<=s.zero:
                            matchbool=True
                            break
                    if not matchbool:
                        Px.append(s.Px[i])
                        Py.append(s.Py[i])
            if colormap is not None:
                try:
                    plotaxis.scatter(Px,Py,color=[colormap((py-start[1])/(stop[1]-start[1])) for py in Py],alpha=alpha)
                except:
                    plotaxis.scatter(Px,Py,color=color,alpha=alpha)
            else:
                plotaxis.scatter(Px,Py,color=color,alpha=alpha)
        if plotshowbool:
            plt.show()
    else:
        Px,Py=[],[]
        for s in S:
            for i in range(len(s.Px)):
                matchbool=False
                for j in range(len(Px)):
                    if abs(s.Px[i]-Px[j])<=s.zero and abs(s.Py[i]-Py[j])<=s.zero:
                        matchbool=True
                        break
                if not matchbool:
                    Px.append(s.Px[i])
                    Py.append(s.Py[i])
        return Px,Py
        
def plot3D(f,start,stop,delta,args=[],kwargs={},limit=[None,None],dependentvar="z",plotaxis=None,wireframe=False,colormap=None,color="black",alpha=1.0):
    #This method provides a consistent way to plot any f:R^2->R function 
    #between lower bound vector start and upper bound vector stop, using delta 
    #as a step vector. The variable limit provides a lower and upper limit on 
    #the value of f. The method either returns the entire surface of the graph 
    #when the variable wireframe is False, or returns just the wireframe when 
    #the variable wireframe is True. The variable dependentvar determines which
    #variable is a function of the other. For example, if the dependentvar is y,
    #then y(x,z)=f(x,z). A colormap option is available to when plotting 
    #(colormap values are based on the dependentvar-coordinate), but only when 
    #the varaiable wireframe is False.
    plotshowbool=False
    if plotaxis is None:
        plotaxis=Axes3D(plt.figure(figsize=(graphsize,graphsize)))
        plotaxis.set_title("Function Surface",fontdict=font)
        plotaxis.xaxis.set_rotate_label(False)
        plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
        plotaxis.yaxis.set_rotate_label(False)
        plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
        plotaxis.zaxis.set_rotate_label(False)
        plotaxis.set_zlabel("$\\mathbf{Z}$",fontsize=16,rotation=0)
        plotaxis.xaxis.set_tick_params(labelsize=16)
        plotaxis.yaxis.set_tick_params(labelsize=16)
        plotaxis.zaxis.set_tick_params(labelsize=16)
        plotshowbool=True
    
    x=np.linspace(start[0],stop[0],round((stop[0]-start[0])/delta[0])+1)
    y=np.linspace(start[1],stop[1],round((stop[1]-start[1])/delta[1])+1)
    Xgrid,Ygrid=np.meshgrid(x,y)
    Zgrid=np.zeros(Xgrid.shape)
    
    if isinstance(limit[0],int) or isinstance(limit[0],float):
        lowerlimitbool=True
    else:
        lowerlimitbool=False
    if isinstance(limit[1],int) or isinstance(limit[1],float):
        upperlimitbool=True
    else:
        upperlimitbool=False
    
    for i in range(len(y)):
        for j in range(len(x)):
            try:
                res=f(x[j],y[i],*args,**kwargs)
            except ZeroDivisionError:
                neighbors=[]
                if i>0:
                    try:
                        neighbors.append(f(x[j],y[i-1],*args,**kwargs))
                    except:
                        pass
                if i<len(y)-1:
                    try:
                        neighbors.append(f(x[j],y[i+1],*args,**kwargs))
                    except:
                        pass
                if j>0:
                    try:
                        neighbors.append(f(x[j-1],y[i],*args,**kwargs))
                    except:
                        pass
                if j<len(x)-1:
                    try:
                        neighbors.append(f(x[j+1],y[i],*args,**kwargs))
                    except:
                        pass
                if len(neighbors)>0:
                    res=sum(neighbors)/len(neighbors)
                else:
                    res=0
            if lowerlimitbool and res<limit[0]:
                res=limit[0]
            if upperlimitbool and res>limit[1]:
                res=limit[1]
            Zgrid[i][j]=res
            
    if wireframe:
        if "x" in dependentvar.lower():
            plotaxis.plot_wireframe(Zgrid,Xgrid,Ygrid,color=color,alpha=alpha)
        elif "y" in dependentvar.lower():
            plotaxis.plot_wireframe(Xgrid,Zgrid,Ygrid,color=color,alpha=alpha)
        else:
            plotaxis.plot_wireframe(Xgrid,Ygrid,Zgrid,color=color,alpha=alpha)
    else:
        if colormap is not None:
            if "x" in dependentvar.lower():
                plotaxis.plot_surface(Zgrid,Xgrid,Ygrid,cmap=colormap,alpha=alpha)
            elif "y" in dependentvar.lower():
                plotaxis.plot_surface(Xgrid,Zgrid,Ygrid,cmap=colormap,alpha=alpha)
            else:
                plotaxis.plot_surface(Xgrid,Ygrid,Zgrid,cmap=colormap,alpha=alpha)
        else:
            if "x" in dependentvar.lower():
                plotaxis.plot_surface(Zgrid,Xgrid,Ygrid,color=color,alpha=alpha)
            elif "y" in dependentvar.lower():
                plotaxis.plot_surface(Xgrid,Zgrid,Ygrid,color=color,alpha=alpha)
            else:
                plotaxis.plot_surface(Xgrid,Ygrid,Zgrid,color=color,alpha=alpha)
    if plotshowbool:
        plt.show()

def surfaceplot(f,start,stop,delta,args=[],kwargs={},adapt=True,maxlevel=3,mode="linear",plotbool=True,plotaxis=None,scatterplot=False,colormap=None,color="black",alpha=1.0):
    #This method approximates the 2D solutions of f(x,y,z)=0, using the marching 
    #cube algorithm. The zeros are found between lower bound vector start and 
    #upper bound vector stop, and divided into subsections of dimensions delta. 
    #The method can subdivide even further through variable adapt. The method 
    #then evenly bisects each subsection through each dimension and only focusses
    #on subsections that, as best as the method can tell, do not contain any 
    #solutions. The subsections then form a structure similar to a sparse 
    #octree. The maximum level of subdivisions is determined by variable
    #maxlevel. After subdivision, the zeros are approximated using either linear,
    #quadratic, or cubic interpolation, determined by variable mode. If plotted,
    #solutions are plotted either using a wireframe (when variable wireframe is
    #True), a polygonal surface reconstruction (when the variable surface is True),
    #or by just plotting the point-cloud (when both wireframe and surface are 
    #False). A colormap option is available to when plotting (colormap values 
    #are based on the z-coordinate).
    plotshowbool=False
    if plotbool and plotaxis is None:
        plotfig=plt.figure(figsize=(graphsize,graphsize))
        plotaxis=Axes3D(plotfig)
        plotaxis.set_title("Function Surface",fontdict=font)
        plotaxis.xaxis.set_rotate_label(False)
        plotaxis.set_xlabel("$\\mathbf{X}$",fontsize=16,rotation=0)
        plotaxis.yaxis.set_rotate_label(False)
        plotaxis.set_ylabel("$\\mathbf{Y}$",fontsize=16,rotation=0)
        plotaxis.zaxis.set_rotate_label(False)
        plotaxis.set_zlabel("$\\mathbf{Z}$",fontsize=16,rotation=0)
        plotaxis.xaxis.set_tick_params(labelsize=16)
        plotaxis.yaxis.set_tick_params(labelsize=16)
        plotaxis.zaxis.set_tick_params(labelsize=16)
        plotshowbool=True
    
    xlim=plotaxis.get_xlim()
    ylim=plotaxis.get_ylim()
    zlim=plotaxis.get_zlim()
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
        
    phi=plotaxis.azim*math.pi/180
    theta=(90-plotaxis.elev)*math.pi/180
    viewWeights=[math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)]
    
    mode = mode.lower()
    
    if color is not None:
        try:
            color = pltcolors.to_rgb(color)
        except ValueError:
            color = (0.0, 0.0, 0.0)
    if colormap is not None:
        try:
            pltcolors.to_rgb(colormap(0.0))
            pltcolors.to_rgb(colormap(0.5))
            pltcolors.to_rgb(colormap(1.0))
        except Exception:
            colormap = None
    
    paths={0:[[]],
        1:[[6, 7, 11, 6]],
        2:[[5, 6, 10, 5]],
        3:[[5, 7, 11, 10, 5]],
        4:[[4, 5, 9, 4]],
        5:[[4, 5, 9, 4], [6, 7, 11, 6]],
        6:[[4, 6, 10, 9, 4]],
        7:[[4, 7, 11, 10, 9, 4]],
        8:[[4, 7, 8, 4]],
        9:[[4, 6, 11, 8, 4]],
        10:[[4, 7, 8, 4], [5, 6, 10, 5]],
        11:[[4, 5, 10, 11, 8, 4]],
        12:[[5, 7, 8, 9, 5]],
        13:[[5, 6, 11, 8, 9, 5]],
        14:[[6, 7, 8, 9, 10, 6]],
        15:[[8, 9, 10, 11, 8]],
        16:[[2, 3, 11, 2]],
        17:[[2, 3, 7, 6, 2]],
        18:[[2, 3, 11, 2], [5, 6, 10, 5]],
        19:[[2, 3, 7, 5, 10, 2]],
        20:[[2, 3, 11, 2], [4, 5, 9, 4]],
        21:[[2, 3, 7, 6, 2], [4, 5, 9, 4]],
        22:[[2, 3, 11, 2], [4, 6, 10, 9, 4]],
        23:[[2, 3, 7, 4, 9, 10, 2]],
        24:[[2, 3, 11, 2], [4, 7, 8, 4]],
        25:[[2, 3, 8, 4, 6, 2]],
        26:[[2, 3, 11, 2], [4, 7, 8, 4], [5, 6, 10, 5]],
        27:[[2, 3, 8, 4, 5, 10, 2]],
        28:[[2, 3, 11, 2], [5, 7, 8, 9, 5]],
        29:[[2, 3, 8, 9, 5, 6, 2]],
        30:[[2, 3, 11, 2], [6, 7, 8, 9, 10, 6]],
        31:[[2, 3, 8, 9, 10, 2]],
        32:[[1, 2, 10, 1]],
        33:[[1, 2, 10, 1], [6, 7, 11, 6]],
        34:[[1, 2, 6, 5, 1]],
        35:[[1, 2, 11, 7, 5, 1]],
        36:[[1, 2, 10, 1], [4, 5, 9, 4]],
        37:[[1, 2, 10, 1], [4, 5, 9, 4], [6, 7, 11, 6]],
        38:[[1, 2, 6, 4, 9, 1]],
        39:[[1, 2, 11, 7, 4, 9, 1]],
        40:[[1, 2, 10, 1], [4, 7, 8, 4]],
        41:[[1, 2, 10, 1], [4, 6, 11, 8, 4]],
        42:[[1, 2, 6, 5, 1], [4, 7, 8, 4]],
        43:[[1, 2, 11, 8, 4, 5, 1]],
        44:[[1, 2, 10, 1], [5, 7, 8, 9, 5]],
        45:[[1, 2, 10, 1], [5, 6, 11, 8, 9, 5]],
        46:[[1, 2, 6, 7, 8, 9, 1]],
        47:[[1, 2, 11, 8, 9, 1]],
        48:[[1, 3, 11, 10, 1]],
        49:[[1, 3, 7, 6, 10, 1]],
        50:[[1, 3, 11, 6, 5, 1]],
        51:[[1, 3, 7, 5, 1]],
        52:[[1, 3, 11, 10, 1], [4, 5, 9, 4]],
        53:[[1, 3, 7, 6, 10, 1], [4, 5, 9, 4]],
        54:[[1, 3, 11, 6, 4, 9, 1]],
        55:[[1, 3, 7, 4, 9, 1]],
        56:[[1, 3, 11, 10, 1], [4, 7, 8, 4]],
        57:[[1, 3, 8, 4, 6, 10, 1]],
        58:[[1, 3, 11, 6, 5, 1], [4, 7, 8, 4]],
        59:[[1, 3, 8, 4, 5, 1]],
        60:[[1, 3, 11, 10, 1], [5, 7, 8, 9, 5]],
        61:[[1, 3, 8, 9, 1], [5, 6, 10, 5]],
        62:[[1, 3, 8, 9, 1], [6, 7, 11, 6]],
        63:[[1, 3, 8, 9, 1]],
        64:[[0, 1, 9, 0]],
        65:[[0, 1, 9, 0], [6, 7, 11, 6]],
        66:[[0, 1, 9, 0], [5, 6, 10, 5]],
        67:[[0, 1, 9, 0], [5, 7, 11, 10, 5]],
        68:[[0, 1, 5, 4, 0]],
        69:[[0, 1, 5, 4, 0], [6, 7, 11, 6]],
        70:[[0, 1, 10, 6, 4, 0]],
        71:[[0, 1, 10, 11, 7, 4, 0]],
        72:[[0, 1, 9, 0], [4, 7, 8, 4]],
        73:[[0, 1, 9, 0], [4, 6, 11, 8, 4]],
        74:[[0, 1, 9, 0], [4, 7, 8, 4], [5, 6, 10, 5]],
        75:[[0, 1, 9, 0], [4, 5, 10, 11, 8, 4]],
        76:[[0, 1, 5, 7, 8, 0]],
        77:[[0, 1, 5, 6, 11, 8, 0]],
        78:[[0, 1, 10, 6, 7, 8, 0]],
        79:[[0, 1, 10, 11, 8, 0]],
        80:[[0, 1, 9, 0], [2, 3, 11, 2]],
        81:[[0, 1, 9, 0], [2, 3, 7, 6, 2]],
        82:[[0, 1, 9, 0], [2, 3, 11, 2], [5, 6, 10, 5]],
        83:[[0, 1, 9, 0], [2, 3, 7, 5, 10, 2]],
        84:[[0, 1, 5, 4, 0], [2, 3, 11, 2]],
        85:[[0, 1, 5, 4, 0], [2, 3, 7, 6, 2]],
        86:[[0, 1, 10, 6, 4, 0], [2, 3, 11, 2]],
        87:[[0, 3, 7, 4, 0], [1, 2, 10, 1]],
        88:[[0, 1, 9, 0], [2, 3, 11, 2], [4, 7, 8, 4]],
        89:[[0, 1, 9, 0], [2, 3, 8, 4, 6, 2]],
        90:[[0, 1, 9, 0], [2, 3, 11, 2], [4, 7, 8, 4], [5, 6, 10, 5]],
        91:[[0, 1, 9, 0], [2, 3, 8, 4, 5, 10, 2]],
        92:[[0, 1, 5, 7, 8, 0], [2, 3, 11, 2]],
        93:[[0, 3, 8, 0], [1, 2, 6, 5, 1]],
        94:[[0, 3, 8, 0], [1, 2, 10, 1], [6, 7, 11, 6]],
        95:[[0, 3, 8, 0], [1, 2, 10, 1]],
        96:[[0, 2, 10, 9, 0]],
        97:[[0, 2, 10, 9, 0], [6, 7, 11, 6]],
        98:[[0, 2, 6, 5, 9, 0]],
        99:[[0, 2, 11, 7, 5, 9, 0]],
        100:[[0, 2, 10, 5, 4, 0]],
        101:[[0, 2, 11, 7, 4, 0], [5, 6, 10, 5]],
        102:[[0, 2, 6, 4, 0]],
        103:[[0, 2, 11, 7, 4, 0]],
        104:[[0, 2, 10, 9, 0], [4, 7, 8, 4]],
        105:[[0, 2, 10, 9, 0], [4, 6, 11, 8, 4]],
        106:[[0, 2, 6, 5, 9, 0], [4, 7, 8, 4]],
        107:[[0, 2, 11, 8, 0], [4, 5, 9, 4]],
        108:[[0, 2, 10, 5, 7, 8, 0]],
        109:[[0, 2, 11, 8, 0], [5, 6, 10, 5]],
        110:[[0, 2, 6, 7, 8, 0]],
        111:[[0, 2, 11, 8, 0]],
        112:[[0, 3, 11, 10, 9, 0]],
        113:[[0, 3, 7, 6, 10, 9, 0]],
        114:[[0, 3, 11, 6, 5, 9, 0]],
        115:[[0, 3, 7, 5, 9, 0]],
        116:[[0, 3, 11, 10, 5, 4, 0]],
        117:[[0, 3, 7, 4, 0], [5, 6, 10, 5]],
        118:[[0, 3, 11, 6, 4, 0]],
        119:[[0, 3, 7, 4, 0]],
        120:[[0, 3, 11, 10, 9, 0], [4, 7, 8, 4]],
        121:[[0, 3, 8, 0], [4, 6, 10, 9, 4]],
        122:[[0, 3, 8, 0], [4, 5, 9, 4], [6, 7, 11, 6]],
        123:[[0, 3, 8, 0], [4, 5, 9, 4]],
        124:[[0, 3, 8, 0], [5, 7, 11, 10, 5]],
        125:[[0, 3, 8, 0], [5, 6, 10, 5]],
        126:[[0, 3, 8, 0], [6, 7, 11, 6]],
        127:[[0, 3, 8, 0]],
        128:[[0, 3, 8, 0]],
        129:[[0, 3, 8, 0], [6, 7, 11, 6]],
        130:[[0, 3, 8, 0], [5, 6, 10, 5]],
        131:[[0, 3, 8, 0], [5, 7, 11, 10, 5]],
        132:[[0, 3, 8, 0], [4, 5, 9, 4]],
        133:[[0, 3, 8, 0], [4, 5, 9, 4], [6, 7, 11, 6]],
        134:[[0, 3, 8, 0], [4, 6, 10, 9, 4]],
        135:[[0, 3, 8, 0], [4, 7, 11, 10, 9, 4]],
        136:[[0, 3, 7, 4, 0]],
        137:[[0, 3, 11, 6, 4, 0]],
        138:[[0, 3, 7, 4, 0], [5, 6, 10, 5]],
        139:[[0, 3, 11, 10, 5, 4, 0]],
        140:[[0, 3, 7, 5, 9, 0]],
        141:[[0, 3, 11, 6, 5, 9, 0]],
        142:[[0, 3, 7, 6, 10, 9, 0]],
        143:[[0, 3, 11, 10, 9, 0]],
        144:[[0, 2, 11, 8, 0]],
        145:[[0, 2, 6, 7, 8, 0]],
        146:[[0, 2, 11, 8, 0], [5, 6, 10, 5]],
        147:[[0, 2, 10, 5, 7, 8, 0]],
        148:[[0, 2, 11, 8, 0], [4, 5, 9, 4]],
        149:[[0, 2, 6, 7, 8, 0], [4, 5, 9, 4]],
        150:[[0, 2, 11, 8, 0], [4, 6, 10, 9, 4]],
        151:[[0, 2, 10, 9, 0], [4, 7, 8, 4]],
        152:[[0, 2, 11, 7, 4, 0]],
        153:[[0, 2, 6, 4, 0]],
        154:[[0, 2, 11, 7, 4, 0], [5, 6, 10, 5]],
        155:[[0, 2, 10, 5, 4, 0]],
        156:[[0, 2, 11, 7, 5, 9, 0]],
        157:[[0, 2, 6, 5, 9, 0]],
        158:[[0, 2, 10, 9, 0], [6, 7, 11, 6]],
        159:[[0, 2, 10, 9, 0]],
        160:[[0, 3, 8, 0], [1, 2, 10, 1]],
        161:[[0, 3, 8, 0], [1, 2, 10, 1], [6, 7, 11, 6]],
        162:[[0, 3, 8, 0], [1, 2, 6, 5, 1]],
        163:[[0, 3, 8, 0], [1, 2, 11, 7, 5, 1]],
        164:[[0, 3, 8, 0], [1, 2, 10, 1], [4, 5, 9, 4]],
        165:[[0, 3, 8, 0], [1, 2, 10, 1], [4, 5, 9, 4], [6, 7, 11, 6]],
        166:[[0, 3, 8, 0], [1, 2, 6, 4, 9, 1]],
        167:[[0, 3, 8, 0], [1, 2, 11, 7, 4, 9, 1]],
        168:[[0, 3, 7, 4, 0], [1, 2, 10, 1]],
        169:[[0, 3, 11, 6, 4, 0], [1, 2, 10, 1]],
        170:[[0, 3, 7, 4, 0], [1, 2, 6, 5, 1]],
        171:[[0, 1, 5, 4, 0], [2, 3, 11, 2]],
        172:[[0, 1, 9, 0], [2, 3, 7, 5, 10, 2]],
        173:[[0, 1, 9, 0], [2, 3, 11, 2], [5, 6, 10, 5]],
        174:[[0, 1, 9, 0], [2, 3, 7, 6, 2]],
        175:[[0, 1, 9, 0], [2, 3, 11, 2]],
        176:[[0, 1, 10, 11, 8, 0]],
        177:[[0, 1, 10, 6, 7, 8, 0]],
        178:[[0, 1, 5, 6, 11, 8, 0]],
        179:[[0, 1, 5, 7, 8, 0]],
        180:[[0, 1, 9, 0], [4, 5, 10, 11, 8, 4]],
        181:[[0, 1, 9, 0], [4, 7, 8, 4], [5, 6, 10, 5]],
        182:[[0, 1, 9, 0], [4, 6, 11, 8, 4]],
        183:[[0, 1, 9, 0], [4, 7, 8, 4]],
        184:[[0, 1, 10, 11, 7, 4, 0]],
        185:[[0, 1, 10, 6, 4, 0]],
        186:[[0, 1, 5, 4, 0], [6, 7, 11, 6]],
        187:[[0, 1, 5, 4, 0]],
        188:[[0, 1, 9, 0], [5, 7, 11, 10, 5]],
        189:[[0, 1, 9, 0], [5, 6, 10, 5]],
        190:[[0, 1, 9, 0], [6, 7, 11, 6]],
        191:[[0, 1, 9, 0]],
        192:[[1, 3, 8, 9, 1]],
        193:[[1, 3, 8, 9, 1], [6, 7, 11, 6]],
        194:[[1, 3, 8, 9, 1], [5, 6, 10, 5]],
        195:[[1, 3, 8, 9, 1], [5, 7, 11, 10, 5]],
        196:[[1, 3, 8, 4, 5, 1]],
        197:[[1, 3, 8, 4, 5, 1], [6, 7, 11, 6]],
        198:[[1, 3, 8, 4, 6, 10, 1]],
        199:[[1, 3, 11, 10, 1], [4, 7, 8, 4]],
        200:[[1, 3, 7, 4, 9, 1]],
        201:[[1, 3, 11, 6, 4, 9, 1]],
        202:[[1, 3, 7, 6, 10, 1], [4, 5, 9, 4]],
        203:[[1, 3, 11, 10, 1], [4, 5, 9, 4]],
        204:[[1, 3, 7, 5, 1]],
        205:[[1, 3, 11, 6, 5, 1]],
        206:[[1, 3, 7, 6, 10, 1]],
        207:[[1, 3, 11, 10, 1]],
        208:[[1, 2, 11, 8, 9, 1]],
        209:[[1, 2, 6, 7, 8, 9, 1]],
        210:[[1, 2, 11, 8, 9, 1], [5, 6, 10, 5]],
        211:[[1, 2, 10, 1], [5, 7, 8, 9, 5]],
        212:[[1, 2, 11, 8, 4, 5, 1]],
        213:[[1, 2, 6, 5, 1], [4, 7, 8, 4]],
        214:[[1, 2, 10, 1], [4, 6, 11, 8, 4]],
        215:[[1, 2, 10, 1], [4, 7, 8, 4]],
        216:[[1, 2, 11, 7, 4, 9, 1]],
        217:[[1, 2, 6, 4, 9, 1]],
        218:[[1, 2, 11, 7, 4, 9, 1], [5, 6, 10, 5]],
        219:[[1, 2, 10, 1], [4, 5, 9, 4]],
        220:[[1, 2, 11, 7, 5, 1]],
        221:[[1, 2, 6, 5, 1]],
        222:[[1, 2, 10, 1], [6, 7, 11, 6]],
        223:[[1, 2, 10, 1]],
        224:[[2, 3, 8, 9, 10, 2]],
        225:[[2, 3, 11, 2], [6, 7, 8, 9, 10, 6]],
        226:[[2, 3, 8, 9, 5, 6, 2]],
        227:[[2, 3, 11, 2], [5, 7, 8, 9, 5]],
        228:[[2, 3, 8, 4, 5, 10, 2]],
        229:[[2, 3, 11, 2], [4, 7, 8, 4], [5, 6, 10, 5]],
        230:[[2, 3, 8, 4, 6, 2]],
        231:[[2, 3, 11, 2], [4, 7, 8, 4]],
        232:[[2, 3, 7, 4, 9, 10, 2]],
        233:[[2, 3, 11, 2], [4, 6, 10, 9, 4]],
        234:[[2, 3, 7, 6, 2], [4, 5, 9, 4]],
        235:[[2, 3, 11, 2], [4, 5, 9, 4]],
        236:[[2, 3, 7, 5, 10, 2]],
        237:[[2, 3, 11, 2], [5, 6, 10, 5]],
        238:[[2, 3, 7, 6, 2]],
        239:[[2, 3, 11, 2]],
        240:[[8, 9, 10, 11, 8]],
        241:[[6, 7, 8, 9, 10, 6]],
        242:[[5, 6, 11, 8, 9, 5]],
        243:[[5, 7, 8, 9, 5]],
        244:[[4, 5, 10, 11, 8, 4]],
        245:[[4, 7, 8, 4], [5, 6, 10, 5]],
        246:[[4, 6, 11, 8, 4]],
        247:[[4, 7, 8, 4]],
        248:[[4, 7, 11, 10, 9, 4]],
        249:[[4, 6, 10, 9, 4]],
        250:[[4, 5, 9, 4], [6, 7, 11, 6]],
        251:[[4, 5, 9, 4]],
        252:[[5, 7, 11, 10, 5]],
        253:[[5, 6, 10, 5]],
        254:[[6, 7, 11, 6]],
        255:[[]]}
    
    class cube:
        def __init__(self,Sv1,deltax,deltay,deltaz,f):
            if not isinstance(Sv1,np.ndarray):
                Sv1=np.array(Sv1)
            self.deltax=deltax
            self.deltay=deltay
            self.deltaz=deltaz
            self.V=[Sv1,Sv1+np.array([deltax,0,0]),Sv1+np.array([deltax,deltay,0]),Sv1+np.array([0,deltay,0]),Sv1+np.array([0,0,deltaz]),Sv1+np.array([deltax,0,deltaz]),Sv1+np.array([deltax,deltay,deltaz]),Sv1+np.array([0,deltay,deltaz])]
            self.viewVal=sum([(self.V[0][i]-start[i]+self.V[6][i]-stop[i])/2*viewWeights[i] for i in range(3)])
            self.c=0.5*(self.V[0]+self.V[6])
            self.errx=1e-3*(self.V[6][0]-self.V[0][0])
            self.erry=1e-3*(self.V[6][1]-self.V[0][1])
            self.errz=1e-3*(self.V[6][2]-self.V[0][2])
            self.zero=0.0#1e-6*self.deltax*self.deltay*self.deltaz
            self.f=f
            self.F=[0 for i in range(8)]
            self.fc=0
            self.full=False
            self.done=False
            self.Px=[]
            self.Py=[]
            self.Pz=[]
            
        def adjust(self, a, b=None, terminate=True):
            shift = 100
            i = 0
            if b is None:
                b = self.c
            nanbool = True
            while nanbool and i <= shift:
                try:
                    tempa = a + i * (b - a) / shift
                    tempfa = f(tempa, *args, **kwargs)
                    nanbool = math.isinf(tempfa) or math.isnan(tempfa)
                except Exception:
                    nanbool = True
                i += 1
            if nanbool:
                tempa = a.copy()
                tempfa = 0
                self.done = terminate
            return tempa, tempfa
        
        def intersect(self,a,b,fa,fb,Px,Py,Pz,mode="linear"):
            if mode in ["quad","quadratic","2","two","second"]:
                if abs(a[0]-b[0])<=self.errx and abs(a[1]-b[1])<=self.erry:
                    c=np.array([a[0],a[1],0.5*(a[2]+b[2])])
                    pivot=2
                elif abs(a[0]-b[0])<=self.erry and abs(a[2]-b[2])<=self.errz:
                    c=np.array([a[0],0.5*(a[1]+b[1]),a[2]])
                    pivot=1
                else:
                    c=np.array([0.5*(a[0]+b[0]),a[1],a[2]])
                    pivot=0
                c,fc=self.adjust(c,b=self.V[0])
                A=((b[pivot]-c[pivot])*fa+(c[pivot]-a[pivot])*fb+(a[pivot]-b[pivot])*fc)/((a[pivot]-b[pivot])*(a[pivot]-c[pivot])*(b[pivot]-c[pivot]))
                B=((c[pivot]**2-b[pivot]**2)*fa+(a[pivot]**2-c[pivot]**2)*fb+(b[pivot]**2-a[pivot]**2)*fc)/((a[pivot]-b[pivot])*(a[pivot]-c[pivot])*(b[pivot]-c[pivot]))
                C=((b[pivot]-c[pivot])*b[pivot]*c[pivot]*fa+(c[pivot]-a[pivot])*c[pivot]*a[pivot]*fb+(a[pivot]-b[pivot])*a[pivot]*b[pivot]*fc)/((a[pivot]-b[pivot])*(a[pivot]-c[pivot])*(b[pivot]-c[pivot]))
                R=np.roots([A,B,C])
                res=0
                rcount=0
                for r in R:
                    if abs(r.imag)<=self.zero:
                        if pivot==0:
                            err=self.errx
                        elif pivot==1:
                            err=self.erry
                        else:
                            err=self.errz
                        if abs(a[pivot]-r.real)<=err:
                            res+=a[pivot]
                            rcount+=1
                        elif abs(b[pivot]-r.real)<=err:
                            res+=b[pivot]
                            rcount+=1
                        else:
                            if a[pivot]<=b[pivot]:
                                if a[pivot]<r.real<b[pivot]:
                                    res+=r.real
                                    rcount+=1
                            else:
                                if a[pivot]>r.real>b[pivot]:
                                    res+=r.real
                                    rcount+=1
                if rcount>0:
                    if pivot==0:
                        Px.append(res/rcount)
                        Py.append(a[1])
                        Pz.append(a[2])
                    elif pivot==1:
                        Px.append(a[0])
                        Py.append(res/rcount)
                        Pz.append(a[2])
                    else:
                        Px.append(a[0])
                        Py.append(a[1])
                        Pz.append(res/rcount)
            elif mode in ["cube","cubic","3","three","third"]:
                if abs(a[0]-b[0])<=self.errx and abs(a[1]-b[1])<=self.erry:
                    h=1e-3*self.deltaz
                    aplus=a+np.array([0,0,h])
                    aminus=a-np.array([0,0,h])
                    bplus=b+np.array([0,0,h])
                    bminus=b-np.array([0,0,h])
                    pivot=2
                elif abs(a[0]-b[0])<=self.errx and abs(a[2]-b[2])<=self.errz:
                    h=1e-3*self.deltay
                    aplus=a+np.array([0,h,0])
                    aminus=a-np.array([0,h,0])
                    bplus=b+np.array([0,h,0])
                    bminus=b-np.array([0,h,0])
                    pivot=1
                else:
                    h=1e-3*self.deltax
                    aplus=a+np.array([h,0,0])
                    aminus=a-np.array([h,0,0])
                    bplus=b+np.array([h,0,0])
                    bminus=b-np.array([h,0,0])
                    pivot=0
                
                aplus,faplus=self.adjust(aplus)
                aminus,faminus=self.adjust(aminus)
                bplus,fbplus=self.adjust(bplus)
                bminus,fbminus=self.adjust(bminus)
                
                faprime=(faplus-faminus)/(2*h)
                fbprime=(fbplus-fbminus)/(2*h)
                    
                A=(a[pivot]*faprime+a[pivot]*fbprime-b[pivot]*faprime-b[pivot]*fbprime-2*fa+2*fb)/((a[pivot]-b[pivot])**3)
                B=(-a[pivot]**2*faprime-2*a[pivot]**2*fbprime-a[pivot]*b[pivot]*faprime+a[pivot]*b[pivot]*fbprime+3*a[pivot]*fa-3*a[pivot]*fb+2*b[pivot]**2*faprime+b[pivot]**2*fbprime+3*b[pivot]*fa-3*b[pivot]*fb)/((a[pivot]-b[pivot])**3)
                C=(a[pivot]**3*fbprime+2*a[pivot]**2*b[pivot]*faprime+a[pivot]**2*b[pivot]*fbprime-a[pivot]*b[pivot]**2*faprime-2*a[pivot]*b[pivot]**2*fbprime-6*a[pivot]*b[pivot]*fa+6*a[pivot]*b[pivot]*fb-b[pivot]**3*faprime)/((a[pivot]-b[pivot])**3)
                D=(-a[pivot]**3*b[pivot]*fbprime+a[pivot]**3*fb-a[pivot]**2*b[pivot]**2*faprime+a[pivot]**2*b[pivot]**2*fbprime-3*a[pivot]**2*b[pivot]*fb+a[pivot]*b[pivot]**3*faprime+3*a[pivot]*b[pivot]**2*fa-b[pivot]**3*fa)/((a[pivot]-b[pivot])**3)
                R=np.roots([A,B,C,D])
                res=0
                rcount=0
                for r in R:
                    if abs(r.imag)<=self.zero:
                        if pivot==0:
                            err=self.errx
                        elif pivot==1:
                            err=self.erry
                        else:
                            err=self.errz
                        if abs(a[pivot]-r.real)<=err:
                            res+=a[pivot]
                            rcount+=1
                        elif abs(b[pivot]-r.real)<=err:
                            res+=b[pivot]
                            rcount+=1
                        else:
                            if a[pivot]<=b[pivot]:
                                if a[pivot]<r.real<b[pivot]:
                                    res+=r.real
                                    rcount+=1
                            else:
                                if a[pivot]>r.real>b[pivot]:
                                    res+=r.real
                                    rcount+=1
                if rcount>0:
                    if pivot==0:
                        Px.append(res/rcount)
                        Py.append(a[1])
                        Pz.append(a[2])
                    elif pivot==1:
                        Px.append(a[0])
                        Py.append(res/rcount)
                        Pz.append(a[2])
                    else:
                        Px.append(a[0])
                        Py.append(a[1])
                        Pz.append(res/rcount)
            else:
                temp=(a*fb-b*fa)/(fb-fa)
                Px.append(temp[0])
                Py.append(temp[1])
                Pz.append(temp[2])
        
        def calc(self):
            for i in range(8):
                v,fv=self.adjust(self.V[i])
                self.F[i]=fv
            self.c,self.fc=self.adjust(self.c,b=self.V[0])
            self.full=True
            for fv in self.F:
                if abs(fv)>self.zero:
                    self.full=False
                    break
            
            if not self.full:
                if all([fv<0 for fv in self.F]) or all([fv>0 for fv in self.F]):
                    self.done=True
                    for i in range(6):
                        if i in [0,1,2]:
                            mid=0.5*(self.V[i]+self.V[i+5])
                        elif i==3:
                            mid=0.5*(self.V[3]+self.V[4])
                        elif i==4:
                            mid=0.5*(self.V[0]+self.V[2])
                        else:
                            mid=0.5*(self.V[4]+self.V[6])
                        mid,fm=self.adjust(mid,terminate=False)
                        if fm*self.fc<0:
                            self.done=False
                            break
                    if self.done:
                        for i in range(12):
                            if i in [0,1,2]:
                                mid=0.5*(self.V[i]+self.V[i+1])
                            elif i==3:
                                mid=0.5*(self.V[3]+self.V[0])
                            elif i in [4,7]:
                                mid=0.5*(self.V[i]+self.V[i-4])
                            elif i in [8,9,10]:
                                mid=0.5*(self.V[i-4]+self.V[i-3])
                            else:
                                mid=0.5*(self.V[7]+self.V[4])
                            mid,fm=self.adjust(mid,terminate=False)
                            if fm*self.fc<0:
                                self.done=False
                                break
        
        def is_straight(self):
            diffx=np.zeros(len(self.V))
            diffy=np.zeros(len(self.V))
            diffz=np.zeros(len(self.V))
            for i in range(len(self.V)):
                plus,fplus=self.adjust(self.V[i]+np.array([1e-3*self.deltax,0,0]),
                                       b=self.V[i]+np.array([self.deltax,0,0]),terminate=False)
                minus,fminus=self.adjust(self.V[i]-np.array([1e-3*self.deltax,0,0]),
                                         b=self.V[i]-np.array([self.deltax,0,0]),terminate=False)
                diffx[i]=(fplus-fminus)/(plus[0]-minus[0])
                
                plus,fplus=self.adjust(self.V[i]+np.array([0,1e-3*self.deltay,0]),
                                       b=self.V[i]+np.array([0,self.deltay,0]),terminate=False)
                minus,fminus=self.adjust(self.V[i]-np.array([0,1e-3*self.deltay,0]),
                                         b=self.V[i]-np.array([0,self.deltay,0]),terminate=False)
                diffy[i]=(fplus-fminus)/(plus[1]-minus[1])
                
                plus,fplus=self.adjust(self.V[i]+np.array([0,0,1e-3*self.deltaz]),
                                       b=self.V[i]+np.array([0,0,self.deltaz]),terminate=False)
                minus,fminus=self.adjust(self.V[i]-np.array([0,0,1e-3*self.deltaz]),
                                         b=self.V[i]-np.array([0,0,self.deltaz]),terminate=False)
                diffz[i]=(fplus-fminus)/(plus[2]-minus[2])
                
            if all([(diffx[i] - diffx[0]) ** 2 + (diffy[i] - diffy[0]) ** 2 +
                    (diffz[i] - diffz[0]) ** 2 <= 3 * (0.5) ** 2
                    for i in range(1, len(self.V))]):
                return True
            elif any([math.isinf(diffx[i] - diffx[0]) or math.isinf(diffy[i] - diffy[0]) or
                      math.isinf(diffz[i] - diffz[0]) or math.isnan(diffx[i] - diffx[0]) or
                      math.isnan(diffy[i] - diffy[0]) or math.isnan(diffz[i] - diffz[0])
                      for i in range(1, len(self.V))]):
                return True
            return False
        
        def split(self,C):
            if not self.done:
                if not self.full and not self.is_straight():
                    splitC=[]
                    for c in [cube(0.5*(self.V[0]+self.V[i]),self.deltax/2,self.deltay/2,self.deltaz/2,self.f) for i in range(8)]:
                        c.calc()
                        if not c.done:
                            splitC.append(c)
                    C+=splitC
                else:
                    C+=[self]
        
        def findpoints(self):
            if not self.done and not self.full:
                index=0
                for i in range(8):
                    if self.F[i]<=self.zero:
                        index+=2**(7-i)
                path=paths[index]
                for p in path:
                    if len(p)>1:
                        for e in p[:-1]:
                            if 0<=e<=2 or 4<=e<=6:
                                self.intersect(self.V[e],self.V[e+1],self.F[e],self.F[e+1],self.Px,self.Py,self.Pz,mode=mode)
                            elif e==3 or e==7:
                                self.intersect(self.V[e],self.V[e-3],self.F[e],self.F[e-3],self.Px,self.Py,self.Pz,mode=mode)
                            else:
                                self.intersect(self.V[e-8],self.V[e-4],self.F[e-8],self.F[e-4],self.Px,self.Py,self.Pz,mode=mode)
                
        def draw(self,segments,colors,zorders):
            if not self.done:
                if self.full:
                    self.drawbox(segments,colors,zorders)
                else:
                    if len(self.Px)>0:
                        segments.append(np.column_stack((self.Px,self.Py,self.Pz)))
                        if colormap is not None:
                            colors.append(colormap((sum(self.Pz)/len(self.Pz)-start[2])/(stop[2]-start[2])))
                        zorders.append(self.viewVal)
            
        def drawbox(self,segments,colors,zorders):
            segments.extend([[(self.V[i][0],self.V[i][1],self.V[i][2]) for i in (0, 1, 2, 3)],
                             [(self.V[i][0],self.V[i][1],self.V[i][2]) for i in (0, 1, 5, 4)],
                             [(self.V[i][0],self.V[i][1],self.V[i][2]) for i in (1, 2, 6, 5)],
                             [(self.V[i][0],self.V[i][1],self.V[i][2]) for i in (2, 3, 7, 6)],
                             [(self.V[i][0],self.V[i][1],self.V[i][2]) for i in (0, 3, 7, 4)],
                             [(self.V[i][0],self.V[i][1],self.V[i][2]) for i in (4, 5, 6, 7)]])
            if colormap is not None:
                colors.extend([colormap((0.5 * (self.vertices[0][2] +
                                         self.vertices[6][2]) -
                                         start[2]) /
                                        (stop[2] - start[2]))
                               for i in range(6)])
            zorders.extend([self.viewVal for i in range(6)])
            self.done = True

    if adapt:
        delta=[max((stop[i]-start[i])/5,delta[i]) for i in range(3)]
    else:
        delta=[max((stop[i]-start[i])/30,delta[i]) for i in range(3)]
    Sv1=[]
    for i in range(round((stop[0]-start[0])/delta[0])):
        for j in range(round((stop[1]-start[1])/delta[1])):
            for k in range(round((stop[2]-start[2])/delta[2])):
                Sv1.append(np.array([start[0]+delta[0]*i,start[1]+delta[1]*j,start[2]+delta[2]*k]))
    
    C=[cube(s,delta[0],delta[1],delta[2],f) for s in Sv1]
    for c in C:
        c.calc()

    if adapt:
        for i in range(maxlevel-1):
            newC=[]
            for c in C:
                c.split(newC)
            C=[c for c in newC]    
    
    for c in C:
        c.findpoints()
    
    if plotbool:
        if not scatterplot:
            segments=[]
            colors=[]
            zorders=[]
            for c in C:
                c.draw(segments, colors, zorders)
            if color is not None:
                if colormap is not None:
                    plotaxis.add_collection3d(art3d.Poly3DCollection(segments,facecolor=colors,edgecolor="black",
                                                               zorder=zorders,alpha=alpha))
                else:
                    if sum(color) <= 0.5:
                        plotaxis.add_collection3d(art3d.Poly3DCollection(segments,facecolor=color,edgecolor="white",
                                                                   zorder=zorders,alpha=alpha))
                    else:
                        plotaxis.add_collection3d(art3d.Poly3DCollection(segments,facecolor=color,edgecolor="black",
                                                                   zorder=zorders,alpha=alpha))
            else:
                if colormap is not None:
                    plotaxis.add_collection3d(art3d.Poly3DCollection(segments,facecolor=(0,0,0,0),edgecolor=colors,zorder=zorders))
                else:
                    plotaxis.add_collection3d(art3d.Poly3DCollection(segments,facecolor=(0,0,0,0),edgecolor="black",zorder=zorders))
        else:
            Px,Py,Pz=[],[],[]
            for c in C:
                for i in range(len(c.Px)):
                    matchbool=False
                    for j in range(len(Px)):
                        if abs(c.Px[i]-Px[j])<=c.zero and abs(c.Py[i]-Py[j])<=c.zero and abs(c.Pz[i]-Pz[j])<=c.zero:
                            matchbool=True
                            break
                    if not matchbool:
                        Px.append(c.Px[i])
                        Py.append(c.Py[i])
                        Pz.append(c.Pz[i])
            if colormap is not None:
                try:
                    plotaxis.scatter(Px,Py,Pz,color=[colormap((pz-start[2])/(stop[2]-start[2])) for pz in Pz],alpha=alpha)
                except:
                    plotaxis.scatter(Px,Py,Pz,color=color,alpha=alpha)
            else:
                plotaxis.scatter(Px,Py,Pz,color=color,alpha=alpha)
        if plotshowbool:
            plt.show()
    else:
        Px,Py,Pz=[],[],[]
        for c in C:
            for i in range(len(c.Px)):
                matchbool=False
                for j in range(len(Px)):
                    if abs(c.Px[i]-Px[j])<=c.zero and abs(c.Py[i]-Py[j])<=c.zero and abs(c.Pz[i]-Pz[j])<=c.zero:
                        matchbool=True
                        break
                if not matchbool:
                    Px.append(c.Px[i])
                    Py.append(c.Py[i])
                    Pz.append(c.Pz[i])
        return Px,Py,Pz
