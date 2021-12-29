import math
import numpy as np
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
