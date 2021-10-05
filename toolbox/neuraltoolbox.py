import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

graphsize=9
font = {"family": "serif",
"color": "black",
"weight": "bold",
"size": "16"}
 
#This file contains the classes Layer and NeuralNet.
#Layer contains all the variables and methods needed to push an input through one layer of a feed-forward neural network.
class Layer:
    #The following static classes define several of the vectorized activation functions and their derivatives.
    @staticmethod
    def identity(X):
        return X,np.ones(X.shape)
    @staticmethod
    def binary(X):
        return np.array([0.0 if x<0 else 1.0 for x in X]),np.zeros(X.shape)
    @staticmethod
    def sigmoid(X):
        res=1.0/(1.0+np.exp(-X))
        return res,res*(1-res)
    @staticmethod
    def tanh(X):
        res=np.tanh(X)
        return res,1-res**2
    @staticmethod
    def relu(X):
        res=np.array([0.0 if x<0 else 1.0 for x in X])
        return res*X,res
    @staticmethod
    def leaky_relu(X):
        res=np.array([0.01 if x<0 else 1.0 for x in X])
        return res*X,res
    @staticmethod
    def gelu(X):
        res1=np.sqrt(2)
        res2=np.array([0.5*(1+math.erf(x/res1)) for x in X])
        return X*res2,X*np.exp(-X**2/2)/(res1*np.sqrt(np.pi))+res2
    @staticmethod
    def silu(X):
        res=1.0/(1.0+np.exp(-X))
        return X*res,res*(1+X*(1-res))
    @staticmethod
    def elu(X,a=1.0):
        return np.array([a*(np.exp(x)-1) if x<0 else x for x in X]),np.array([a*np.exp(x) if x<0 else 1.0 for x in X])
    @staticmethod
    def softplus(X):
        res=np.exp(X)
        return np.log(1.0+res),res/(1.0+res)
    @staticmethod
    def softmax(X):
        res1=np.exp(X-max(X))
        res2=res1/sum(res1)
        return res2,np.array([[res2[i]*(1-res2[j]) if i==j else -res2[i]*res2[j] for j in range(len(X))] for i in range(len(X))])
    @staticmethod
    def gauss(X):
        res=np.exp(-X**2)
        return res,-2*X*res
            
    def __init__(self,size,input_size,activ_Func_Title):
        #This initialization method allows the user to designate the following:
        #<size> determines the number of artificial neurons in the layer
        #<input_size> determines the number of inputs each artificial neuron will recieve
        #<activ_Func_Title> determines the activation function each artificial neuron will use.
        self.size=size
        self.activ_Func_Title=activ_Func_Title.lower()
        if self.activ_Func_Title in ["bin","binary"]:
            self.activ_Func=Layer.binary
        elif self.activ_Func_Title in ["sig","sigmoid"]:
            self.activ_Func=Layer.sigmoid
        elif self.activ_Func_Title in ["tanh","hyperbolic tagent"]:
            self.activ_Func=Layer.tanh
        elif self.activ_Func_Title in ["relu"]:
            self.activ_Func=Layer.relu
        elif self.activ_Func_Title in ["lrelu","leaky relu"]:
            self.activ_Func=Layer.leaky_relu
        elif self.activ_Func_Title in ["gelu"]:
            self.activ_Func=Layer.gelu
        elif self.activ_Func_Title in ["silu"]:
            self.activ_Func=Layer.silu
        elif self.activ_Func_Title in ["elu"]:
            self.activ_Func=Layer.elu
        elif self.activ_Func_Title in ["softplus"]:
            self.activ_Func=Layer.softplus
        elif self.activ_Func_Title in ["softmax"]:
            self.activ_Func=Layer.softmax
        elif self.activ_Func_Title in ["gauss"]:
            self.activ_Func=Layer.gauss
        else:
            self.activ_Func="identity"
            self.activ_Func=Layer.identity
            
        self.input=np.append(np.zeros(input_size),1)
        #The variable W determines the weights of the artificial neurons in the layer (including biases). 
        self.W=np.array([np.append(2*np.random.rand(input_size)-1,0) for p in range(self.size)])
        #The variable V determines the "momentum" of the weights of the artificial neurons in the layer (including biases) as the network trains.
        self.V=np.zeros(self.W.shape)
        #These variables are necessary for efficient back propagation (full explanations are mathematically lengthy and thus not provided here).
        self.dEdW=np.zeros(self.W.shape)
        self.aux_sum_1=np.zeros(self.W.shape)
        self.aux_sum_2=np.zeros(self.W.shape)
        self.output=(np.zeros(self.size),np.zeros(self.size))
    
    def activate(self,inputs,one_to_one=False):
        #This method pushes the inputs through the layer to result in an output.
        #Distinction is made whether the layer is an input layer or not.
        if one_to_one:
            self.input=inputs
            self.output=self.activ_Func(self.W[:,0]*self.input+self.W[:,1])
        else:
            self.input[0:-1]=inputs
            self.output=self.activ_Func(self.W.dot(self.input))
        return self.output
        
    def __repr__(self):
        #This method returns a string representation of the layer.
        res="LAYER: Number of Neurons: %i, Activation Function: %s\n"%(self.size,self.activ_Func_Title)
        for i in range(self.size):
            res+="NEURON %i: Number of Weights: %i\n"%(i,self.W.shape[1]-1)
            res+="Weights: "
            for j in range(len(self.W[i,:-1])):
                if j==len(self.W[i,:-1])-1:
                    res+=str(round(self.W[i,j],3))+"\n"
                else:
                    res+=str(round(self.W[i,j],3))+", "
            res+="Bias: "+str(round(self.W[i,-1],3))+"\n"
        return res

#NeuralNet contains all the variables and methods needed to design, operate, save, and load a feed-forward neural network.
#It also contains the infastructure needed to train a neural network effeciently using optimizers.
class NeuralNet:
    #The following static methods compute various error functions used for back propagation.
    @staticmethod
    def Mean_Square_Error(prediction,target,deriv=False):
        if deriv:
            return 2*(prediction-target)/len(target)
        else:
            return ((prediction-target)**2).mean()

    @staticmethod
    def Cross_Entropy_Error(prediction,target,deriv=False,epsilon=0.1):
        if deriv:
            return (epsilon-1)*((epsilon+1)*target+(epsilon-1)*prediction-epsilon)/(len(target)*((1-epsilon)**2*prediction*(1-prediction)+epsilon))
        else:
            return -(target*np.log((1-epsilon)*prediction+epsilon)+(1-target)*np.log((1-epsilon)*(1-prediction)+epsilon)).mean()

    def __init__(self,layer_structure,activ_Func_Titles,importfile=None):
        #This initialization method either loads a network from a pre-existing file or constructs a new one.
        #The user can designate the following:
        #<layer_structure> is a list of numbers, where the length of the list equals the total number of layers (including input layer),
        #and every element of the list represents the number of artificial neurons in the corresponding layer.
        #<activ_Func_Titles> is a list of strings, where the length of the list equals the total number of layers (including input layer),
        #and every element of the list represents the activation function each of the artificial neurons in the corresponding layer will use.
        if importfile is not None:
            self.read(importfile)
        else:
            self.layer_structure=layer_structure
            self.inputsize=self.layer_structure[0]
            self.activ_Func_Titles=activ_Func_Titles
            self.size=len(self.layer_structure)
            self.layers=[Layer(self.layer_structure[0],1,self.activ_Func_Titles[0])]+[Layer(self.layer_structure[i],self.layer_structure[i-1],self.activ_Func_Titles[i]) for i in range(1,self.size)]
            self.input=np.zeros(self.layer_structure[0])
            self.output=(np.zeros(self.layer_structure[-1]),np.zeros(self.layer_structure[-1]))
            if self.layers[-1].activ_Func_Title=="softmax":
                self.error_Func=NeuralNet.Cross_Entropy_Error
            else:
                self.error_Func=NeuralNet.Mean_Square_Error
    
    def __repr__(self):
        #This method returns a string representation of the neural network.
        res="NEURAL NETWORK: Number of Layers: %i, Total Number of Neurons: %i\n\n"%(self.size,sum([layer.size for layer in self.layers]))
        for i in range(self.size):
            res+="LAYER %i: Number of Neurons: %i, Activation Function: %s\n"%(i,self.layers[i].size,self.layers[i].activ_Func_Title)
            for j in range(self.layers[i].size):
                res+="NEURON %i: Number of Weights: %i\n"%(j,self.layers[i].W.shape[1]-1)
                res+="Weights: "
                for k in range(len(self.layers[i].W[j,:-1])):
                    if k==len(self.layers[i].W[j,:-1])-1:
                        res+=str(round(self.layers[i].W[j,k],3))+"\n"
                    else:
                        res+=str(round(self.layers[i].W[j,k],3))+", "
                res+="Bias: "+str(round(self.layers[i].W[j,-1],3))+"\n"
            if i!=self.size-1:
                res+="\n\n"
        return res
        
    def draw(self,fade=False):
        #This method visualizes of the neural network by plotting it.
        #Distinction is made whether to fade the artificial neurons based on the strength of the bias.
        #Distinction is also made whether to fade the weights based on their strength.
        fig=plt.figure(1,figsize=(graphsize,graphsize))
        ax=fig.add_subplot(111)
        r=1
        ax.set_xlim([-1,(6*self.size-2)*r+1])
        ax.set_ylim([-1,(3*max([self.inputsize]+[self.layer_structure[i] for i in range(self.size)])-1)*r+1])
        plt.axis("off")
        if fade:
            max_weight=max([np.amax(layer.W) for layer in self.layers[1:]])
            min_weight=min([np.amin(layer.W[:,:-1]) for layer in self.layers])
            max_weight=1.1*max_weight-0.1*min_weight
            min_weight=1.1*min_weight-0.1*max_weight
            max_bias=max([np.amax(layer.W[:,-1]) for layer in self.layers])
            min_bias=min([np.amin(layer.W[:,-1]) for layer in self.layers])
            max_bias=1.1*max_bias-0.1*min_bias
            min_bias=1.1*min_bias-0.1*max_bias
            if max_bias==min_bias:
                max_bias=1
                min_bias=0
        for i in range(self.size):
            for j in range(self.layer_structure[i]):
                if fade:
                    if i==0:
                        c=plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="orangered",alpha=(self.layers[i].W[j,-1]-min_bias)/(max_bias-min_bias))
                        ax.add_artist(plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="orangered",fill=False))
                    elif i==self.size-1:
                        c=plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="green",alpha=(self.layers[i].W[j,-1]-min_bias)/(max_bias-min_bias))
                        ax.add_artist(plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="green",fill=False))
                    else:
                        c=plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="dodgerblue",alpha=(self.layers[i].W[j,-1]-min_bias)/(max_bias-min_bias))
                        ax.add_artist(plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="dodgerblue",fill=False))
                else:
                    if i==0:
                        c=plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="orangered")
                    elif i==self.size-1:
                        c=plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="green")
                    else:
                        c=plt.Circle(((6*i+3)*r,(3*j+1)*r),r,color="dodgerblue")
                ax.add_artist(c)
            if i==0:
                for j in range(self.layer_structure[0]):
                    if fade:
                        plt.plot([0,3*r],[(3*j+1)*r,(3*j+1)*r],
                              color="black",alpha=(self.layers[i].W[j,0]-min_weight)/(max_weight-min_weight))
                    else:
                        plt.plot([0,3*r],[(3*j+1)*r,(3*j+1)*r],color="black")
            else:
                for j in range(self.layer_structure[i]):
                    for k in range(self.layer_structure[i-1]):
                        if fade:
                            plt.plot([(6*i-3)*r,(6*i+3)*r],[(3*k+1)*r,(3*j+1)*r],
                                  color="black",alpha=(self.layers[i].W[j,k]-min_weight)/(max_weight-min_weight))
                        else:
                            plt.plot([(6*i-3)*r,(6*i+3)*r],[(3*k+1)*r,(3*j+1)*r],color="black")
                            
    def write(self):
        #This method writes the structure, weights, and biases of the neural network to a user-designated text file.
        #Files written this way can be read and converted back into a neuralnet object with the read method.
        lyr_string=str(self.layer_structure[0])
        func_string=str(self.activ_Func_Titles[0])
        for i in range(1,self.size):
            lyr_string+=",%i"%self.layer_structure[i]
            func_string+=",%s"%self.activ_Func_Titles[i]
        filename="neural_net["+lyr_string+"],["+func_string+"].txt"
            
        file=open(filename,"w")
        
        file.write(lyr_string+"\n")
        file.write(func_string+"\n\n")
        
        for i in range(self.size):
            for j in range(self.layers[i].W.shape[0]):
                wght_string="%f"%self.layers[i].W[j,0]
                for k in range(1,self.layers[i].W.shape[1]):
                    wght_string+=",%f"%self.layers[i].W[j,k]
                if i!=self.size-1:
                    file.write(wght_string+"\n")
                else:
                    file.write(wght_string)
            if i!=self.size-1:
                file.write("\n")
        file.close()
        print("Neural Network printed to "+os.getcwd()+"\\"+filename)
    
    def read(self,file_name):
        #This method reads a user-designated neural network structure file and builds a neuralnet object from it.
        #Neuralnet objects can be rewritten back into text file format with the write method.
        file=open(file_name,"r")
        layer_structure_string=file.readline().replace("\n","").split(",")
        self.layer_structure=[int(elem) for elem in layer_structure_string]        
        activ_Func_Titles_string=file.readline().replace("\n","").split(",")
        self.activ_Func_Titles=[str(elem) for elem in activ_Func_Titles_string]
        
        self.inputsize=self.layer_structure[0]
        self.size=len(self.layer_structure)
        self.layers=[Layer(self.layer_structure[0],1,self.activ_Func_Titles[0])]+[Layer(self.layer_structure[i],self.layer_structure[i-1],self.activ_Func_Titles[i]) for i in range(1,self.size)]
        self.input=np.zeros(self.layer_structure[0])
        self.output=np.zeros(self.layer_structure[-1]),np.zeros(self.layer_structure[-1])
        if self.layers[-1].activ_Func_Title=="softmax":
            self.error_Func=NeuralNet.Cross_Entropy_Error
        else:
            self.error_Func=NeuralNet.Mean_Square_Error
        
        for l in self.layers:
            file.readline()
            for i in range(l.W.shape[0]):
                wghts=file.readline().replace("\n","").split(",")
                l.W[i,:]=np.array([float(w) for w in wghts])
                l.V[i,:]=np.zeros(len(wghts))
        file.close()
        
    def push(self,inputs):
        #This method pushes an input through the entire neural network and returns the result.
        if len(inputs)==self.inputsize:
            self.input=np.asarray(inputs)
            self.output=self.layers[0].activate(self.input,one_to_one=True)
            for i in range(1,self.size):
                self.output=self.layers[i].activate(self.output[0])
            return self.output[0]
        else:
            raise Exception("Neural network can take %i inputs but recieved %i"%(self.inputsize,len(inputs)))
            
    def calculate_dEdPhi(self,target_set):
        #For each artificial neuron, this method calculates the derivative of the error between the neural network's prediction and target
        #with respect to the neuron's activation function.
        dEdPhi=[1 for i in range(self.size-1)]+[self.error_Func(self.layers[-1].output[0],target_set,deriv=True)]
        for i in range(self.size-2,-1,-1):
            if self.layers[i+1].activ_Func_Title=="softmax":
                dEdPhi[i]=(dEdPhi[i+1]*np.diag(self.layers[i+1].output[1])).dot(self.layers[i+1].W[:,:-1])
            else:
                dEdPhi[i]=(dEdPhi[i+1]*self.layers[i+1].output[1]).dot(self.layers[i+1].W[:,:-1])
        return dEdPhi

    def calculate_dEdW(self,target_set,addition=False):
        #For each artificial neuron, this method calculates the derivative of the error between the neural network's prediction and target
        #with respect to the neuron's weights and bias.
        dEdPhi=self.calculate_dEdPhi(target_set)
        if self.layers[0].activ_Func_Title=="softmax":
            res=self.layers[0].output[1].T.dot(dEdPhi[0])
        else:
            res=self.layers[0].output[1]*dEdPhi[0]
        if addition:
            self.layers[0].dEdW+=np.transpose([res*self.layers[0].input,res])
        else:
            self.layers[0].dEdW=np.transpose([res*self.layers[0].input,res])
        for i in range(1,self.size):
            if addition:
                out=np.empty((self.layers[i].size,self.layers[i-1].size+1))
                if self.layers[i].activ_Func_Title=="softmax":
                    np.outer(self.layers[i].output[1].T.dot(dEdPhi[i]),self.layers[i].input,out)
                    self.layers[i].dEdW+=out
                else:
                    np.outer(dEdPhi[i]*self.layers[i].output[1],self.layers[i].input,out)
                    self.layers[i].dEdW+=out
            else:
                if self.layers[i].activ_Func_Title=="softmax":
                    np.outer(self.layers[i].output[1].T.dot(dEdPhi[i]),self.layers[i].input,self.layers[i].dEdW)
                else:
                    np.outer(dEdPhi[i]*self.layers[i].output[1],self.layers[i].input,self.layers[i].dEdW)

    def backpropagate_sgd(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with no optimizer.
        for l in self.layers:
            l.V=train_rate*l.dEdW
            l.W-=l.V

    def backpropagate_momentum(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the momentum optimizer.
        for l in self.layers:
            l.V=0.9*l.V+train_rate*l.dEdW
            l.W-=l.V
    
    def backpropagate_nesterov(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the nesterov accelerated gradient optimizer.
        for l in self.layers:
            l.V=0.9*l.V+train_rate*l.dEdW
            l.W-=0.9*l.V+train_rate*l.dEdW
            
    def backpropagate_adagrad(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the adagrad optimizer.
        for l in self.layers:
            l.aux_sum_1+=l.dEdW**2
            l.V=train_rate*l.dEdW/np.sqrt(l.aux_sum_1+1e-6)
            l.W-=l.V
    
    def backpropagate_rmsprop(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the rmsprop optimizer.
        for l in self.layers:
            l.aux_sum_1=0.9*l.aux_sum_1+0.1*l.dEdW**2
            l.V=train_rate*l.dEdW/np.sqrt(l.aux_sum_1+1e-6)
            l.W-=l.V
        
    def backpropagate_adadelta(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the adadelta optimizer.
        for l in self.layers:
            l.aux_sum_1=0.9*l.aux_sum_1+0.1*l.V**2
            l.aux_sum_2=0.9*l.aux_sum_2+0.1*l.dEdW**2
            l.V=np.sqrt(l.aux_sum_1+1e-6)*l.dEdW/np.sqrt(l.aux_sum_2+1e-6)
            l.W-=l.V
        
    def backpropagate_adam(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the adam optimizer.
        for l in self.layers:
            l.aux_sum_1=0.9*l.aux_sum_1+0.1*l.dEdW
            l.aux_sum_2=0.999*l.aux_sum_2+0.001*l.dEdW**2
            l.V=train_rate*(l.aux_sum_1/(1.0-0.9**tau))/(np.sqrt(l.aux_sum_2/(1.0-0.999**tau))+1e-6)
            l.W-=l.V
    
    def backpropagate_adamax(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the adamax optimizer.
        for l in self.layers:
            l.aux_sum_1=0.9*l.aux_sum_1+0.1*l.dEdW
            l.aux_sum_2=np.maximum(0.999*l.aux_sum_2,np.abs(l.dEdW))
            l.V=train_rate*(l.aux_sum_1/(1.0-0.9**tau))/(l.aux_sum_2+1e-6)
            l.W-=l.V
        
    def backpropagate_nadam(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with the nadam optimizer.
        for l in self.layers:
            l.aux_sum_1=0.9*l.aux_sum_1+0.1*l.dEdW
            l.aux_sum_2=0.999*l.aux_sum_2+0.001*l.dEdW**2
            l.V=train_rate*(0.9*l.aux_sum_1+0.1*l.dEdW)/((1.0-0.9**tau)*np.sqrt(l.aux_sum_2/(1.0-0.999**tau))+1e-6*(1.0-0.9**tau))
            l.W-=l.V
    
    def backpropagate_amsgrad(self,train_rate,tau):
        #This method performs the back propagation algorithm on the neural network with amsgrad optimizer.
        for l in self.layers:
            l.aux_sum_1=0.9*l.aux_sum_1+0.1*l.dEdW
            res=0.999*l.aux_sum_2+0.001*l.dEdW**2
            l.V=train_rate*l.aux_sum_1/(np.sqrt(np.maximum(l.aux_sum_2,res))+1e-6)
            l.aux_sum_2=res
            l.W-=l.V
        
    def train(self,feature_set,target_set,train_rate=None,max_error=1e-3,max_epoch=1000,batch_size=50,inform=True,optimizer="adam"):
        #This method takes as input a feature set and a target set and trains the neural network on it using back propagation.
        #<train_rate> determines the speed at which the neural network learns and is usually pre-determined.
        #<max_error> determines the maximum error the neural network must achieve per epoch for the training to finish.
        #<max_epoch> determines the maximum number of epoch the neural network can train before the training halts.
        #<batch_size> allows for training to be done in mini-batches.
        #<optimizer> determines which optimizer scheme will be used to train the network.
        errors_list=[2*max_error]
        training_set_size=len(feature_set)
        k=0
        for i in range(training_set_size):
            target_set[i]=np.asarray(target_set[i])
        
        if train_rate is None:
            train_rate=1e-3
        else:
            train_rate=abs(float(train_rate))
        
        optimizer=optimizer.lower()
        if "moment" in optimizer:
            optimizer="momentum"
            backprop_func=self.backpropagate_momentum
        elif optimizer in ["nesterov","nag"]:
            optimizer="nesterov"
            backprop_func=self.backpropagate_nesterov
        elif "adagrad" in optimizer:
            optimizer="adagrad"
            backprop_func=self.backpropagate_adagrad
        elif "rmsprop" in optimizer:
            optimizer="rmsprop"
            backprop_func=self.backpropagate_rmsprop
        elif "adadelta" in optimizer:
            optimizer="adadelta"
            backprop_func=self.backpropagate_adadelta
        elif optimizer=="adam":
            optimizer="adam"
            backprop_func=self.backpropagate_adam
        elif optimizer=="adamax":
            optimizer=="adamax"
            backprop_func=self.backpropagate_adamax
        elif optimizer=="nadam":
            optimizer=="nadam"
            backprop_func=self.backpropagate_nadam
        elif "amsgrad" in optimizer:
            optimizer=="amsgrad"
            backprop_func=self.backpropagate_amsgrad
        else:
            optimizer="sgd"
            backprop_func=self.backpropagate_sgd
        
        for l in self.layers:
            l.aux_sum_1=np.zeros(l.W.shape)
            l.aux_sum_2=np.zeros(l.W.shape)
        
        if inform:
            master_tic=time.time()
        
        while errors_list[-1]>max_error and k<max_epoch:
            if inform:
                tic=time.time()
            errors=np.empty(training_set_size)
            for i in range(training_set_size-1):
                self.push(feature_set[i])
                errors[i]=self.error_Func(self.output[0],target_set[i])
                
                if i%batch_size==0:
                    self.calculate_dEdW(target_set[i],addition=False)
                else:
                    self.calculate_dEdW(target_set[i],addition=True)
                
                if (i+1)%batch_size==0:
                    for l in self.layers:
                        l.dEdW/=batch_size
                    backprop_func(train_rate,k*training_set_size+i+1)
            
            self.push(feature_set[i])
            errors[-1]=self.error_Func(self.output[0],target_set[-1])
            self.calculate_dEdW(target_set[-1],addition=True) 
            for l in self.layers:
                l.dEdW/=training_set_size-np.floor(training_set_size/batch_size)
            backprop_func(train_rate,(k+1)*training_set_size)
            
            #If <inform>, the code will interact with the user by showing progress and basic result statistics.
            if inform:
                errors_list.append(errors.mean())
            else:
                errors_list[-1]=errors.mean()
            k+=1
            if inform and k!=max_epoch:
                res_time=time.time()-tic
                if k!=0 and k!=max_epoch and k%100==0:
                    print("Algorithm minimally %0.1f%% complete. Remaining time is at most %0.1f min."%(100*k/max_epoch, res_time*(max_epoch-k)/60))
        if inform:
            res_time=time.time()-master_tic
            if res_time<60:
                print("Time needed: %0.2f seconds"%res_time)
            else:
                print("Time needed: %0.2f minutes"%(res_time/60))
            print("Epochs needed: %i"%k)
            print("Total mean error: %f"%errors_list[-1])

            plt.figure(figsize=(graphsize,graphsize))
            plt.title("Error per Epoch with "+optimizer[0].upper()+optimizer[1:]+" Optimizer",fontdict=font)
            plt.xlabel("Epoch",fontdict=font)
            plt.ylabel("Error",fontdict=font)
            plt.plot([i for i in range(1,k+1)],errors_list[1:],label="Error per Epoch")
            plt.plot([1,k],[max_error,max_error],color="red",label="Error Goal")
            plt.legend(fontsize=font["size"])
