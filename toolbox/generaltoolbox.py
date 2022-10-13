import math
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
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
        return [np.array([n/(N-1)]+[Phi(n,B[i-1]) for i in range(1,dim)]) for n in range(N)]
    else:
        return [np.array([n/(N-1) for n in range(N)])]+[np.array([Phi(n,B[i-1]) for n in range(N)]) for i in range(1,dim)]

def differentiate(f,x,args=[],kwargs={},h=1e-6,variable_Dim=0):
    #This method approximates the (partial) derivative of function f:R^n->R
    #at point x with respect the the (variableDim)th variable,
    #using a complex step whenever possible.
    
    if ("int" in type(x).__name__) or ("float" in type(x).__name__):
        x=float(x)
        try:
            xhplus=x+h*1j
            return np.imag(f(xhplus,*args,**kwargs))/h
        except Exception:
            try:
                xhminus=x-h*1j
                return np.imag(f(xhminus,*args,**kwargs))/-h
            except Exception:
                h=max(1e-6,h)
                try:
                    x2hminus=x-2*h
                    x2hplus=x+2*h
                    fx2hminus=f(x2hminus,*args,**kwargs)
                    fx2hplus=f(x2hplus,*args,**kwargs)
                    try:
                        xhminus=x-h
                        xhplus=x+h
                        fxhminus=f(xhminus,*args,**kwargs)
                        fxhplus=f(xhplus,*args,**kwargs)
                        return (-fx2hplus+8*fxhplus-8*fxhminus+fx2hminus)/(12*h)
                    except Exception:
                        return (fx2hplus-fx2hminus)/(4*h)
                except Exception:
                    try:
                        xhminus=x-h
                        fxhminus=f(xhminus,*args,**kwargs)
                        try:
                            xhplus=x+h
                            fxhplus=f(xhplus,*args,**kwargs)
                            return (fxhplus-fxhminus)/(2*h)
                        except Exception:
                            try:
                                fx=f(x,*args,**kwargs)
                                return (fx-fxhminus)/h
                            except Exception:
                                print("Error: Function is not defined in point ["+", ".join([str(elem) for elem in x])+"]")
                                return None
                    except Exception:
                        try:
                            xhplus=x+h
                            fxhplus=f(xhplus,*args,**kwargs)
                            try:
                                fx=f(x,*args,**kwargs)
                                return (fxhplus-fx)/h
                            except Exception:
                                print("Error: Function is not defined in point ["+", ".join([str(elem) for elem in x])+"]")
                                return None
                        except Exception:
                            print("Error: Function can not be differentiated in point ["+", ".join([str(elem) for elem in x])+"]")
                            return None
                        
    elif ("list" in type(x).__name__) or ("tuple" in type(x).__name__) or ("ndarray" in type(x).__name__):
        x=np.asarray(x,dtype=np.float)
        variable_Dim=max(min(variable_Dim,len(x)),0)
        try:
            xhplus=np.asarray(x,dtype=complex)
            xhplus[variable_Dim]+=h*1j
            return np.imag(f(xhplus,*args,**kwargs))/h
        except Exception:
            try:
                xhminus=np.asarray(x,dtype=complex)
                xhminus[variable_Dim]-=h*1j
                return np.imag(f(xhminus,*args,**kwargs))/-h
            except Exception:
                h=max(1e-6,h)
                try:
                    x2hminus=x.copy()
                    x2hminus[variable_Dim]-=2*h
                    x2hplus=x.copy()
                    x2hplus[variable_Dim]+=2*h
                    fx2hminus=f(x2hminus,*args,**kwargs)
                    fx2hplus=f(x2hplus,*args,**kwargs)
                    try:
                        xhminus=x.copy()
                        xhminus[variable_Dim]-=h
                        xhplus=x.copy()
                        xhplus[variable_Dim]+=h
                        fxhminus=f(xhminus,*args,**kwargs)
                        fxhplus=f(xhplus,*args,**kwargs)
                        return (-fx2hplus+8*fxhplus-8*fxhminus+fx2hminus)/(12*h)
                    except Exception:
                        return (fx2hplus-fx2hminus)/(4*h)
                except Exception:
                    try:
                        xhminus=x.copy()
                        xhminus[variable_Dim]-=h
                        fxhminus=f(xhminus,*args,**kwargs)
                        try:
                            xhplus=x.copy()
                            xhplus[variable_Dim]+=h
                            fxhplus=f(xhplus,*args,**kwargs)
                            return (fxhplus-fxhminus)/(2*h)
                        except Exception:
                            try:
                                fx=f(x,*args,**kwargs)
                                return (fx-fxhminus)/h
                            except Exception:
                                print("Error: Function is not defined in point ["+", ".join([str(elem) for elem in x])+"]")
                                return None
                    except Exception:
                        try:
                            xhplus=x.copy()
                            xhplus[variable_Dim]+=h
                            fxhplus=f(xhplus,*args,**kwargs)
                            try:
                                fx=f(x,*args,**kwargs)
                                return (fxhplus-fx)/h
                            except Exception:
                                print("Error: Function is not defined in point ["+", ".join([str(elem) for elem in x])+"]")
                                return None
                        except Exception:
                            print("Error: Function can not be differentiated in point ["+", ".join([str(elem) for elem in x])+"]")
                            return None
    else:
        print("Error: point x is of an incompatible type %s"%type(x).__name__)
        return None

def double_differentiate(f,x,args=[],kwargs={},h=1e-3,variable_Dim_1=0,variable_Dim_2=0):
    if ("int" in type(x).__name__) or ("float" in type(x).__name__):
        x=float(x)
        try:
            fx=f(x,*args,**kwargs)
        except Exception:
            print("Error: function not defined on inputted point x")
            return None
        try:
            xhplus=x+h
            fxhplus=f(xhplus,*args,**kwargs)
            try:
                xhminus=x-h
                fxhminus=f(xhminus,*args,**kwargs)
                return (fxhplus-2*fx+fxhminus)/(h**2)
            except Exception:
                try:
                    x2hplus=x+2*h
                    fx2hplus=f(x2hplus,*args,**kwargs)
                    try:
                        x3hplus=x+3*h
                        fx3hplus=f(x3hplus,*args,**kwargs)
                        return (2*fx-5*fxhplus+4*fx2hplus-fx3hplus)/(h**2)
                    except Exception:
                        return (fx-2*fxhplus+fx2hplus)/(h**2)
                except Exception:
                    print("Error: unable to find second derivative")
                    return None
        except Exception:
            try:
                xhminus=x-h
                fxhminus=f(xhminus,*args,**kwargs)
                x2hminus=x-2*h
                fx2hminus=f(x2hminus,*args,**kwargs)
                try:
                    x3hminus=x-3*h
                    fx3hminus=f(x3hminus,*args,**kwargs)
                    return (2*fx-5*fxhminus+4*fx2hminus-fx3hminus)/(h**2)
                except Exception:
                    return (fx-2*fxhminus+fx2hminus)/(h**2)
            except Exception:
                print("Error: unable to find second derivative")
                return None
            
    elif ("list" in type(x).__name__) or ("tuple" in type(x).__name__) or ("ndarray" in type(x).__name__):
        x=np.asarray(x,dtype=np.float)
        try:
            fx=f(x,*args,**kwargs)
        except Exception:
            print("Error: function not defined on inputted point x")
            return None
        variable_Dim_1=max(min(variable_Dim_1,len(x)),0)
        variable_Dim_2=max(min(variable_Dim_2,len(x)),0)
        if variable_Dim_1==variable_Dim_2:
            try:
                xhplus=x.copy()
                xhplus[variable_Dim_1]+=h
                fxhplus=f(xhplus,*args,**kwargs)
                try:
                    xhminus=x.copy()
                    xhminus[variable_Dim_1]-=h
                    fxhminus=f(xhminus,*args,**kwargs)
                    return (fxhplus-2*fx+fxhminus)/(h**2)
                except Exception:
                    try:
                        x2hplus=x.copy()
                        x2hplus[variable_Dim_1]+=2*h
                        fx2hplus=f(x2hplus,*args,**kwargs)
                        try:
                            x3hplus=x.copy()
                            x3hplus[variable_Dim_1]+=3*h
                            fx3hplus=f(x3hplus,*args,**kwargs)
                            return (2*fx-5*fxhplus+4*fx2hplus-fx3hplus)/(h**2)
                        except Exception:
                            return (fx-2*fxhplus+fx2hplus)/(h**2)
                    except Exception:
                        print("Error: unable to find second derivative")
                        return None
            except Exception:
                try:
                    xhminus=x.copy()
                    xhminus[variable_Dim_1]-=h
                    fxhminus=f(xhminus,*args,**kwargs)
                    x2hminus=x.copy()
                    x2hminus[variable_Dim_1]-=2*h
                    fx2hminus=f(x2hminus,*args,**kwargs)
                    try:
                        x3hminus=x.copy()
                        x3hminus[variable_Dim_1]+=3*h
                        fx3hminus=f(x3hminus,*args,**kwargs)
                        return (2*fx-5*fxhminus+4*fx2hminus-fx3hminus)/(h**2)
                    except Exception:
                        return (fx-2*fxhminus+fx2hminus)/(h**2)
                except Exception:
                    print("Error: unable to find second derivative")
                    return None
        else:
            try:
                xhplusplus=x.copy()
                xhplusplus[variable_Dim_1]+=h
                xhplusplus[variable_Dim_2]+=h
                fxhplusplus=np.asarray(f(xhplusplus,*args,**kwargs))
                xhplusminus=x.copy()
                xhplusminus[variable_Dim_1]+=h
                xhplusminus[variable_Dim_2]-=h
                fxhplusminus=np.asarray(f(xhplusminus,*args,**kwargs))
                xhminusplus=x.copy()
                xhminusplus[variable_Dim_1]-=h
                xhminusplus[variable_Dim_2]+=h
                fxhminusplus=np.asarray(f(xhminusplus,*args,**kwargs))
                xhminusminus=x.copy()
                xhminusminus[variable_Dim_1]-=h
                xhminusminus[variable_Dim_2]-=h
                fxhminusminus=np.asarray(f(xhminusminus,*args,**kwargs))
                res=(fxhplusplus-fxhplusminus-fxhminusplus+fxhminusminus)/(4*h**2)
                try:
                    return res.item()
                except Exception:
                    return res
            except Exception:
                try:
                    xhplusminus=x.copy()
                    xhplusminus[variable_Dim_1]+=h
                    xhplusminus[variable_Dim_2]-=h
                    fxhplusminus=np.asarray(f(xhplusminus,*args,**kwargs))
                    xhminusplus=x.copy()
                    xhminusplus[variable_Dim_1]-=h
                    xhminusplus[variable_Dim_2]+=h
                    fxhminusplus=np.asarray(f(xhminusplus,*args,**kwargs))
                    xhminusminus=x.copy()
                    xhminusminus[variable_Dim_1]-=h
                    xhminusminus[variable_Dim_2]-=h
                    fxhminusminus=np.asarray(f(xhminusminus,*args,**kwargs))
                    xhminusminus2=x.copy()
                    xhminusminus2[variable_Dim_1]-=2*h
                    xhminusminus2[variable_Dim_2]-=2*h
                    fxhminusminus2=np.asarray(f(xhminusminus2,*args,**kwargs))
                    res=(3*fx-fxhplusminus-fxhminusplus-2*fxhminusminus+fxhminusminus2)/(4*h**2)
                    try:
                        return res.item()
                    except Exception:
                        return res
                except Exception:
                    try:
                        xhplusplus=x.copy()
                        xhplusplus[variable_Dim_1]+=h
                        xhplusplus[variable_Dim_2]+=h
                        fxhplusplus=np.asarray(f(xhplusplus,*args,**kwargs))
                        xhplusplus2=x.copy()
                        xhplusplus2[variable_Dim_1]+=2*h
                        xhplusplus2[variable_Dim_2]+=2*h
                        fxhplusplus2=np.asarray(f(xhplusplus2,*args,**kwargs))
                        xhplusminus=x.copy()
                        xhplusminus[variable_Dim_1]+=h
                        xhplusminus[variable_Dim_2]-=h
                        fxhplusminus=np.asarray(f(xhplusminus,*args,**kwargs))
                        xhminusplus=x.copy()
                        xhminusplus[variable_Dim_1]-=h
                        xhminusplus[variable_Dim_2]+=h
                        fxhminusplus=np.asarray(f(xhminusplus,*args,**kwargs))
                        res=(3*fx-fxhplusminus-fxhminusplus-2*fxhplusplus+fxhplusplus2)/(4*h**2)
                        try:
                            return res.item()
                        except Exception:
                            return res
                    except Exception:
                        try:
                            xhplusplus=x.copy()
                            xhplusplus[variable_Dim_1]+=h
                            xhplusplus[variable_Dim_2]+=h
                            fxhplusplus=np.asarray(f(xhplusplus,*args,**kwargs))
                            xhplusminus=x.copy()
                            xhplusminus[variable_Dim_1]+=h
                            xhplusminus[variable_Dim_2]-=h
                            fxhplusminus=np.asarray(f(xhplusminus,*args,**kwargs))
                            xhplusminus2=x.copy()
                            xhplusminus2[variable_Dim_1]+=2*h
                            xhplusminus2[variable_Dim_2]-=2*h
                            fxhplusminus2=np.asarray(f(xhplusminus2,*args,**kwargs))
                            xhminusminus=x.copy()
                            xhminusminus[variable_Dim_1]-=h
                            xhminusminus[variable_Dim_2]-=h
                            fxhminusminus=np.asarray(f(xhminusminus,*args,**kwargs))
                            res=(3*fx-fxhplusplus-fxhminusminus-2*fxhplusminus+fxhplusminus2)/(4*h**2)
                            try:
                                return res.item()
                            except Exception:
                                return res
                        except Exception:
                            try:
                                xhplusplus=x.copy()
                                xhplusplus[variable_Dim_1]+=h
                                xhplusplus[variable_Dim_2]+=h
                                fxhplusplus=np.asarray(f(xhplusplus,*args,**kwargs))
                                xhminusplus=x.copy()
                                xhminusplus[variable_Dim_1]-=h
                                xhminusplus[variable_Dim_2]+=h
                                fxhminusplus=np.asarray(f(xhminusplus,*args,**kwargs))
                                xhminusplus2=x.copy()
                                xhminusplus2[variable_Dim_1]-=2*h
                                xhminusplus2[variable_Dim_2]+=2*h
                                fxhminusplus2=np.asarray(f(xhminusplus2,*args,**kwargs))
                                xhminusminus=x.copy()
                                xhminusminus[variable_Dim_1]-=h
                                xhminusminus[variable_Dim_2]-=h
                                fxhminusminus=np.asarray(f(xhminusminus,*args,**kwargs))
                                res=(3*fx-fxhplusplus-fxhminusminus-2*fxhminusplus+fxhminusplus2)/(4*h**2)
                                try:
                                    return res.item()
                                except Exception:
                                    return res
                            except Exception:
                                try:
                                    xhplusplus=x.copy()
                                    xhplusplus[variable_Dim_1]+=h
                                    xhplusplus[variable_Dim_2]+=h
                                    fxhplusplus=np.asarray(f(xhplusplus,*args,**kwargs))
                                    xhplus0=x.copy()
                                    xhplus0[variable_Dim_1]+=h
                                    fxhplus0=np.asarray(f(xhplus0,*args,**kwargs))
                                    xh0plus=x.copy()
                                    xh0plus[variable_Dim_2]+=h
                                    fxh0plus=np.asarray(f(xh0plus,*args,**kwargs))
                                    res=(fx-fxh0plus-fxhplus0+fxhplusplus)/(h**2)
                                    try:
                                        return res.item()
                                    except Exception:
                                        return res
                                except Exception:
                                    try:
                                        xhplusminus=x.copy()
                                        xhplusminus[variable_Dim_1]+=h
                                        xhplusminus[variable_Dim_2]-=h
                                        fxhplusminus=np.asarray(f(xhplusminus,*args,**kwargs))
                                        xhplus0=x.copy()
                                        xhplus0[variable_Dim_1]+=h
                                        fxhplus0=np.asarray(f(xhplus0,*args,**kwargs))
                                        xh0minus=x.copy()
                                        xh0minus[variable_Dim_2]-=h
                                        fxh0minus=np.asarray(f(xh0minus,*args,**kwargs))
                                        res=(fx-fxh0minus-fxhplus0+fxhplusminus)/(h**2)
                                        try:
                                            return res.item()
                                        except Exception:
                                            return res
                                    except Exception:
                                        try:
                                            xhminusplus=x.copy()
                                            xhminusplus[variable_Dim_1]-=h
                                            xhminusplus[variable_Dim_2]+=h
                                            fxhminusplus=np.asarray(f(xhplusminus,*args,**kwargs))
                                            xhminus0=x.copy()
                                            xhminus0[variable_Dim_1]-=h
                                            fxhminus0=np.asarray(f(xhminus0,*args,**kwargs))
                                            xh0plus=x.copy()
                                            xh0plus[variable_Dim_2]+=h
                                            fxh0plus=np.asarray(f(xh0plus,*args,**kwargs))
                                            res=(fx-fxh0plus-fxhminus0+fxhminusplus)/(h**2)
                                            try:
                                                return res.item()
                                            except Exception:
                                                return res
                                        except Exception:
                                            try:
                                                xhminusminus=x.copy()
                                                xhminusminus[variable_Dim_1]-=h
                                                xhminusminus[variable_Dim_2]-=h
                                                fxhminusminus=np.asarray(f(xhminusminus,*args,**kwargs))
                                                xhminus0=x.copy()
                                                xhminus0[variable_Dim_1]-=h
                                                fxhminus0=np.asarray(f(xhminus0,*args,**kwargs))
                                                xh0minus=x.copy()
                                                xh0minus[variable_Dim_2]-=h
                                                fxh0minus=np.asarray(f(xh0minus,*args,**kwargs))
                                                res=(fx-fxh0minus-fxhminus0+fxhminusminus)/(h**2)
                                                try:
                                                    return res.item()
                                                except Exception:
                                                    return res
                                            except Exception:
                                                print("Error: unable to find second derivative")
                                                return None
    else:
        print("Error: point x is of an incompatible type %s"%type(x).__name__)
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

def FFT(sol,inverse=False,min_factor=32):
    #This method performs a 1D Fast Fourier Transform (of FFT) of the inputted 
    #signal of size N. The FFT is the fast version for calculating the DFT,
    #which is generally calculated through matrix-vector multiplication, an
    #O(N^2) process. Instead, the FFT solves this in O(NlogN) time by splitting
    #the problem into N1 smaller problems, each sub-problem containing 1/N1=N2
    #of the total data. These sub-problems are then solved and recombined into
    #one final solution.
    #More specifically, this algorithm is divided into three steps. The first
    #step reorders the inputted variable <sol> into subsections based on each
    #element's modulo N2; this is done by function R.
    #Next, function G applies the Fourier Transform to each of the subsections
    #of <sol>. Notice that function G allows for recursion;
    #variable <min_factor> determines how long each subsection must be before
    #recursion halts and the applicable matrix-vector multiplications are
    #calculated directly.
    #Finally, function B reconnects all the subsections together to produce one
    #final solution.
    #Notice that this method also calculates the inverse FFT by setting
    #variable <inverse> to True (default False).
    N=len(sol)
    min_factor=max(1,min(N,int(round(min_factor))))
    def factors(N):
        #This code tries to find radices as small as possible so that
        #function B can be as fast as possible in its current set-up.
        for i in range(2,int(math.sqrt(N))+1):
            if N%i==0:
                return i,int(N/i)
        return 1,N
        #This code tries to find a radix as close to sqrt(N) as possible.
        #This can be beneficial for highly parrallelizable methods, but
        #this particular implementation is not designed for that.
        #for i in range(int(math.sqrt(N)),0,-1):
        #    if N%i==0:
        #        return i,int(N/i)
        
    
    def R(sol,N,N1,N2):
        sol_r=np.empty(N,dtype=np.complex)
        for i in range(N1):
            sol_r[i*N2:(i+1)*N2]=sol[i::N1]
        return sol_r
    
    def F(N):
        F_mat=np.empty((N,N),dtype=np.complex)
        F_mat[0,:]=np.ones(N,dtype=np.complex)
        if inverse:
            for i in range(1,int(N/2)+1):
                F_mat[i,i:N-i+1]=np.exp(2j*math.pi*i/N*np.arange(i,N-i+1))
                F_mat[N-i:i-1:-1,N-i]=F_mat[i,i:N-i+1]
        else:
            for i in range(1,int(N/2)+1):
                F_mat[i,i:N-i+1]=np.exp(-2j*math.pi*i/N*np.arange(i,N-i+1))
                F_mat[N-i:i-1:-1,N-i]=F_mat[i,i:N-i+1]
        for i in range(1,N):
            F_mat[i,:i]=F_mat[:i,i]
        return F_mat
        
    def G(sol,N,N1,N2):
        if N2<=min_factor:
            F_mat=F(N2)
            for i in range(N1):
                sol[i*N2:(i+1)*N2]=np.dot(F_mat,sol[i*N2:(i+1)*N2])
            return sol
        else:
            N3,N4=factors(N2)
            if N3>1:
                for i in range(N1):            
                    sol[i*N2:(i+1)*N2]=B(G(R(sol[i*N2:(i+1)*N2],N2,N3,N4),N2,N3,N4),N2,N3,N4)
                return sol
            else:
                F_mat=F(N2)
                for i in range(N1):
                    sol[i*N2:(i+1)*N2]=np.dot(F_mat,sol[i*N2:(i+1)*N2])
                return sol
    
    def D(N,N1,N2,i,j):
        if inverse:
            return np.exp(2j*math.pi*j*(N2*i+np.arange(N2))/N)
        else:
            return np.exp(-2j*math.pi*j*(N2*i+np.arange(N2))/N)
    
    def B(sol,N,N1,N2):
        #This code is usually faster for smaller radices, but usually
        #much slower for larger radices.
        sol_b=np.empty(N,dtype=np.complex)
        for i in range(N1):
            sol_b[i*N2:(i+1)*N2]=sol[:N2]+sum([sol[j*N2:(j+1)*N2]*D(N,N1,N2,i,j) for j in range(1,N1)])
        return sol_b
        
        #This code is used for general radices. It is more consistent
        #over the spectrum of possible radices, but is usually slower
        #for smaller radices.
        #sol_b=np.empty(N,dtype=np.complex)
        #res1=F(N1)
        #res2=-2j*math.pi/N*np.outer(np.arange(N1),np.arange(N2))
        #if inverse:
        #    res2*=-1    
        #res2=np.exp(res2)
        #sol_b[0]=sol[0]+np.sum(sol[N2:N:N2])
        #for j in range(N2):
        #    sol_b[j]=np.inner(sol[j:N:N2],res2[:,j])
        #for i in range(1,N1):
        #    sol_b[i*N2]=np.inner(sol[0:N:N2],res1[:,i])
        #    for j in range(1,N2):
        #        sol_b[j+i*N2]=np.inner(sol[j:N:N2],res1[:,i]*res2[:,j])
        #return sol_b
    
    N1,N2=factors(N)
    if inverse:
        return B(G(R(sol,N,N1,N2),N,N1,N2),N,N1,N2)/N
    return B(G(R(sol,N,N1,N2),N,N1,N2),N,N1,N2)

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
