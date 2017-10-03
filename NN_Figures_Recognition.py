# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:59:00 2016
@author: benjaminbraun
"""
from mnist import MNIST
import numpy as np
import scipy as sp
import sklearn.metrics as metrics
import random
import csv

NUM_CLASSES = 10
NUM_HIDDEN = 200

sigma = 0.05 # This is the variance used when initializing V and W with random normal distributions

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless,
    # since we are replacing the official MNIST test set with our own test set
    X_test, _ = map(np.array, mndata.load_testing())
    # We reshuffle the training data (X_train), as well as the training labels:
    arr = (np.arange(X_train.shape[0]))
    np.random.shuffle(arr)
    X_train = X_train [arr]
    labels_train = labels_train [arr]    
    return X_train, labels_train, X_test
    
# We create below different preprocessing functions

# Standard pre-processing
def standardize(Xdata):
    for i in range(Xdata.shape[1]):
        Xdata[:,i] = (Xdata[:,i]-np.mean(Xdata[:,i]))/np.std(Xdata[:,i])
    return Xdata

# Pre-processing for logistic regression:
def transformlog(Xdata):
    for i in range(Xdata.shape[1]):
        Xdata[:,i]=np.log(Xdata[:,i]+0.1)
    return Xdata

# Transforms our labels yi into vectors of length 10
def one_hot(labels_train):
    Mat = np.zeros((labels_train.shape[0],NUM_CLASSES));
    i=0;
    for item in labels_train:
        Mat[i,np.int_(item)]=1
        i+=1
    return Mat

# softmax function
def softmaxfun(Ydata):
    for i in range(Ydata.shape[0]):
        Ydata[i,:]=np.e**(Ydata[i,:]-np.max(Ydata[i,:]))/np.sum(np.e**(Ydata[i,:]-np.max(Ydata[i,:])))
    return Ydata

def ReLU(Ydata):
    for i in range(Ydata.shape[0]):
        Ydata[i,:]=[np.max([0,Ydata[i,j]]) for j in range(Ydata.shape[1])]
    return Ydata
    
def Log_loss(Xdata,Ydata,V,W):
        one1 = np.ones([Xdata.shape[0],1])
        XT = np.transpose(Xdata)
        z1 = np.dot(V,XT) # input of the hidden layer, using the matrix V
        a1 = np.transpose(ReLU(np.transpose(z1))) # output of the hidden layer
        one2 = np.ones([1,Xdata.shape[0]])
        a1bias = np.r_[a1,one2] # 201 by 50,000
        z2 = np.dot(W,a1bias)  # input of the output layer
        z2T = np.array(np.transpose(z2))
        a2 = np.transpose(softmaxfun(z2T)) # 10 by 50,000
        a2_log = np.log(a2)
        S=0
        for i in range(Xdata.shape[0]):
            S+= np.dot(Ydata[i,:],a2_log[:,i])        
        return -S
         
# We below use the stochastic gradient descent using mini batch. This means that, instead of picking only one data point at random in our dataset, we pick an 
# index at random, and update the gradient using all the data startng from this index to this index + size (minibatch)
def train_sgd(X_train, y_train,v,w,alpha=0.0001, minib=50,reg=0, num_iter=20000, gamma=0.9):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    
    one1 = np.ones([X_train.shape[0],1])
    # We add a column equal to one to our data set, in order to take the bias into account
    X_train = np.c_[X_train,one1]

    #momentw=0
    #momentv=0
    Loss = [] # Vector which will contain the loss
    Accuracy = [] # Vector which will contain the different training accuracy for various iteration steps
    
    for iter in range(num_iter):
        ran = np.random.randint(0,50000-minib)
        #ran=np.random.randint(60000, size=50)
        #Xi = np.mat(X_train[i,:] for i in ran).reshape((50,785)) 
        #w=w-0.9*momentw 
        #v=v-0.9*momentv
        Xi = np.mat(X_train[ran:ran+minib,:]) # We only take into account the dataset from our index to index + size(minibatch)
        XiT = np.transpose(Xi) #  dim: 785 by 50
        z1 = np.dot(v,XiT) # 200 by 50 (this is the input of the hidden layer)
        a1 = np.transpose(ReLU(np.transpose(z1))) # 200 by 50 (output of the hidden layer)
        one2 = np.ones([1,minib])
        a1bias = np.r_[a1,one2] # 201 by 50
        z2 = np.dot(w,a1bias)  # 10 by 50 : input of the output layer
        z2T = np.array(np.transpose(z2))
        a2 = np.transpose(softmaxfun(z2T)) # 10 by 50: output of the output layer 
        residual = a2 - np.transpose(np.mat( y_train[ran:ran+minib,:] )) # This is the matrix of differences between the predicted outputs and the real labels
        grdw = np.dot( residual , np.transpose(a1bias) ) # 10 by 201
        w = w - alpha*grdw # 10 by 201
        #momentv = alpha*grdw
        
        chain_op =  np.dot(np.transpose(residual) , w ) # 50 1 by 201
        chain_opRe = np.zeros((minib,chain_op.shape[1]-1))  # 50 1 by 200
        for j in range (minib):        
            for i in range (chain_opRe.shape[1]):
                if z1[i,j]<=0:
                    chain_opRe[j,i]=0
                else:
                    chain_opRe[j,i]=chain_op[j,i]
        grdv = np.dot(np.transpose(np.mat(chain_opRe)),Xi) # 200 by 785
        v = v - alpha*grdv

        #Each 5000 iterations, we calculate the logistic loss and training accuracy:
        if (iter%5000==0):
            loss = Log_loss (X_train,y_train,v,w)
            print (loss)
            Loss=np.append(Loss,loss)
            
            pred_labels_train_iter = predict(v,w,X_train[:,:X_train.shape[1]-1])
            accuracy = metrics.accuracy_score(labels_train, pred_labels_train_iter)
            print(accuracy)
            Accuracy=np.append(Accuracy,accuracy)
            
        # After each epoch, we decay the learning rate by gamma
        if(iter%1000==0):
            alpha=alpha*gamma

    return v,w,Loss,Accuracy
    
def predict(v,w, X):
    ''' From model and data points, output predicted y's '''
    one1 = np.ones([X.shape[0],1])
    X = np.c_[X,one1]
    predM= ReLU(np.dot(X,np.transpose(v))) # output of the hidden layer
    one2 = np.ones([X.shape[0],1])
    predM = np.c_[predM,one2] # 60000 by 201
    predO = softmaxfun(np.array(np.dot(predM,np.transpose(w)))) # output of the final layer
    predL = np.zeros(X.shape[0])
    for index3 in range(predO.shape[0]):
        '''P=np.abs(predY[index3,:]-1)'''
        predL[index3]=np.argmax(predO[index3,:]) # Thiw will return the maximum element in the vector y, which corresponds to the predicted label
    return predL

if __name__ == "__main__":   
    X_train, labels_train, X_test = load_dataset()
    X_train=transformlog(X_train)
    X_test=transformlog(X_test)
    X_validation = X_train[50000:,]
    labels_validation = labels_train[50000:,]
    X_train = X_train[:50000,]
    labels_train = labels_train[:50000,]
    y_train = one_hot(labels_train)
# We then initialize both our matrixes V and W
    Id=np.eye(X_train.shape[1]+1)
    v=np.random.multivariate_normal(np.zeros(X_train.shape[1]+1),sigma**2*Id,NUM_HIDDEN)
    Id=np.eye(NUM_HIDDEN+1)
    w=np.random.multivariate_normal(np.zeros(NUM_HIDDEN+1),sigma**2*Id,NUM_CLASSES)
# We then modify the two matrices V and W as the outputs of the stochastic gradient descent algorithm:
    v1,w1,Loss,Accuracy = train_sgd(X_train, y_train,v,w)
    pred_labels_train = predict(v1,w1,X_train)
    pred_labels_validation = predict(v1,w1,X_validation)
    pred_labels_test = predict(v1,w1,X_test)
    pred_labels_test = pred_labels_test.reshape(10000,1)
    
    with open("output.csv","w") as f:
        writer = csv.writer(f)
        for i in range(pred_labels_test.shape[0]):
            writer.writerow(pred_labels_test[i])
        
    np.save('v.npy',v1)
    np.save('w.npy',w1)
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Validation accuracy: {0}".format(metrics.accuracy_score(labels_validation, pred_labels_validation)))