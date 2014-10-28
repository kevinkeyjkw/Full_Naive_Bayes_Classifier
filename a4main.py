import sys
import numpy as np
#import scipy.stats
import random
import pdb

#Correct
def bayes_params(dataset):
    numAttributes=len(dataset[0])-1
    classes = np.unique(dataset[:,numAttributes])
    Pc=[0 for i in range(len(classes))]
    mean=[0 for i in range(len(classes))]
    coMat=[0 for i in range(len(classes))]
    centMat=[0 for i in range(len(classes))]
    for i in range(len(classes)):
        d=np.array([a[:numAttributes] for a in dataset if a[numAttributes]==classes[i]],dtype='float')
        cardinality=len(d)
        Pc[i]=cardinality/len(dataset)
        mean[i]=(1/cardinality)*np.sum(d,axis=0)#sum vertically
        centMat[i]=np.array(d-mean[i])
        coMat[i]=(1/cardinality)*np.dot(centMat[i].transpose(),centMat[i])
    return (classes, Pc, mean, coMat)

def bayes_naive_params(dataset):
    numAttributes=len(dataset[0])-1
    classes = np.unique(dataset[:,numAttributes])
    Pc=[0 for i in range(len(classes))]
    mean=[0 for i in range(len(classes))]
    coMat=[0 for i in range(len(classes))]
    centMat=[0 for i in range(len(classes))]
    for i in range(len(classes)):
        d=np.array([a[:numAttributes] for a in dataset if a[numAttributes]==classes[i]],dtype='float')
        cardinality=len(d)
        Pc[i]=cardinality/len(dataset)
        mean[i]=(1/cardinality)*np.sum(d,axis=0)#sum vertically
        centMat[i]=np.array(d-mean[i])
        coMat[i]=( (1/cardinality)*np.dot(centMat[i].transpose(),centMat[i]) )*np.identity(numAttributes)
    return (classes, Pc, mean, coMat)

# def bayes_naive_params(dataset):
#     numAttributes=len(dataset[0])-1
#     classes = np.unique(dataset[:,numAttributes])
#     Pc=[0 for i in range(len(classes))]
#     var=[0 for i in range(len(classes))]
#     mean=[0 for i in range(len(classes))]
#     centMat=[0 for i in range(len(classes))]
#     pdb.set_trace()
#     for i in range(len(classes)):
#         d=np.array([a[:numAttributes] for a in dataset if a[numAttributes]==classes[i]],dtype='float')
#         cardinality=len(d)
#         Pc[i]=cardinality/len(dataset)
#         mean[i]=(1/cardinality)*np.sum(d,axis=0)
#         centMat[i]=np.array(d-mean[i])
#         varDiag=[]
#         for j in range(numAttributes):#Num of attributes
#             varDiag.append((1/cardinality)*np.dot(centMat[i][:,j],centMat[i][:,j]))
#         var[i]=varDiag
#     return Pc, mean, var

def multi_var_normal_pdf(x, mean, cov):
    return ( 1 / (( np.sqrt(2*np.pi)**len(x) )*np.sqrt(np.linalg.det(cov)) ) )*np.exp((-0.5)* np.dot(np.dot(x-mean,1/cov),x-mean )  )   

def multi_var_normal_pdf2(x, mean, cov):
    return ( 1 / (( np.sqrt(2*np.pi)**len(x) )*np.sqrt(np.linalg.det(cov)) ) )*np.exp((-0.5)* np.dot( (x-mean)*(1/np.diag(cov)),x-mean)   )   

def bayes_test2(x, classes, Pc, mean, cov):
    argmax = -1
    y=classes[0]#default val
    #pdb.set_trace()
    for i in range(len(classes)):
        tmp = Pc[i]*multi_var_normal_pdf2(x,mean[i],cov[i])
        if tmp > argmax: 
            argmax=tmp
            y=classes[i]
    return y

def bayes_test(x, classes, Pc, mean, cov):
    argmax = -1
    y=classes[0]#default val
    #pdb.set_trace()
    for i in range(len(classes)):
        tmp = Pc[i]*multi_var_normal_pdf(x,mean[i],cov[i])
        if tmp > argmax: 
            argmax=tmp
            y=classes[i]
    return y

def paired_t_test(X, Y, K, alpha):
    n = len(X)
    diff = np.zeros(K)
    for i in range(K):
        # 1 compute training set X_i using bootstrap resampling
        ...

        # 2 train both full and naive Bayes on sample X_i
        classes, Pc, mean, cov = bayes_params(X_i, Y_i)

        # 3 compute testing set X - X_i
        ...

        # 4 assess both on X - X_i
        ...
        print('sample, full, naive:', i, num_err_full, num_err_naive)

        # 5 compute difference in error rates
        diff[i] = err_rate_full - err_rate_naive
        
    print('all differences:'); print(diff)
 
    # compute mean, variance, and z-score
    ...
    print('z-score:', z_score)

    # compute interval bound using inverse survival function of t distribution
    bound = scipy.stats.t.isf((1-alpha)/2.0, K-1) 
    print('bound:', bound)

    # output conclusion based on tests
    if ... :
        print('accept: classifiers have similar performance')
    else:
        print('reject: classifiers have significantly different performance')
def readData(f):
    blah=[]
    y=[]
    for x in f.readlines():
        temp = x.split(',')
        length=len(temp)
        word=temp[length-1]
        y.append(word[1:len(word)-2])
        temp = [ float(a) for a in temp[:length-1] ]
        blah.append(temp)
    return blah,y

def bootstrap(rows,y,k):
    d=[]
    for i in range(k):
        temp = [0 for i in range(n)] 
        for j in range(n):
            randNum=random.randint(0,n-1)
            temp[j]=[float(x) for x in rows[ randNum ]]
            temp[j].append(y[randNum])
        d.append(temp)
    return np.array(d)

def testError(correct, predicted,typeClassifier):
    errors=0
    for i in range(len(correct)):
        if correct[i]==predicted[i]:
            print('True')
        else:
            print('False')
            errors = errors + 1
    print(typeClassifier,' Error rate: ',errors/len(correct))

f = open(sys.argv[1],'r')
rows,y = readData(f)
n = len(rows)
k = 5
dataset = bootstrap(rows,y,k)

classes, Pc, mean, coMat = bayes_naive_params(dataset[0])#pass in dataset[i] i=0,1,2...
#Pc, mean, var = bayes_naive_params(dataset[0])
testingSet=rows
result=[]
#bayes_test(x, classes, Pc, mean, coMa) returns predicted class for x
for x in testingSet:
    result.append(bayes_test2(x,classes,Pc,mean,coMat))
pdb.set_trace()
testError(y,result,'Naive Bayes Classifier')
# read in data and command-line arguments, and compute X and Y
classes, Pc, mean, coMat = bayes_params(dataset[0])
result=[]
for x in testingSet:
    result.append(bayes_test(x,classes,Pc,mean,coMat))
testError(y,result,'Full Bayes Classifier')
pdb.set_trace()
paired_t_test(X, Y, K, alpha)
