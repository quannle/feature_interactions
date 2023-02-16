import numpy.random as r
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import time
from scipy import stats
from sklearn.model_selection import KFold
import random
from sklearn import preprocessing
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
import vimpy
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
keras.backend.clear_session()
from scipy.stats import *
from random import sample
 



def logitridge(X,Y,ridge_mult=1000):
    reg=LogisticRegression(penalty='l2',solver='saga',max_iter=10,C = ridge_mult).fit(X,Y)
    return reg
# Define Baseline Regressor to be used later
def DecisionTreeClass(X,Y,fun_para):
    tree = DecisionTreeClassifier(min_samples_leaf=5).fit(X,Y)
    return tree


def logitridgecv(X,Y,ridge_mult=0.001):
    reg=LogisticRegressionCV(cv=5,penalty='l2',solver='saga').fit(X,Y)
    # named ridge2 to distinguish it from the ridge regressor in sklearn.
    return reg

def RFclass(X,Y,ntree=200):
    M =len(X[0])    # when bootstrap=False, it means each tree is trained on all rows of X and only
    #      subsamples its columns (which are features).
    rf = RandomForestClassifier(n_estimators=200, bootstrap=False).fit(X,Y)
    return rf
def getNC(true_y,prob,method = 'prob1'):
    if method=='prob2':
        if len(true_y)==1:
            true_y=true_y[0]
            py=prob[true_y]
            pz = max(prob.drop(true_y,axis=1))
            nc = (1- py+pz)/2
        else:
            py=[prob[item][i] for i,item in enumerate(true_y)] ##prob of true label
            pz=[max(prob.iloc[i].drop(true_y[i])) for i in range(len(true_y))] ## max prob of other label 
            nc = [(1- py[i]+pz[i])/2 for i in range(len(py))]
    if method=='prob1':
        if len(true_y)==1:
            true_y=int(true_y[0])
            py=prob[true_y]
            nc = (1- py)
        else:
            py=[prob[item][i] for i,item in enumerate(true_y)] ##prob of true label
            nc = [(1- py[i]) for i in range(len(py))]
    return np.array(nc)


def buildMPClass(X,Y,n_ratio,m_ratio):
    N = len(X)
    M = len(X[0])
    n = np.int(np.ceil(n_ratio * N))
    m = np.int(np.ceil(m_ratio * M))
    r = np.random.RandomState()
    ## index of minipatch
    #3 stratified sampling
    Y_pd=pd.DataFrame(Y.reshape((len(Y),1)))

    idx_I =Y_pd.groupby(0, group_keys=False).apply(lambda x: x.sample(frac=n_ratio))
    idx_I = np.sort(list(idx_I.index)) # stratified sampling of subset of observations

    #idx_I = np.sort(r.choice(N, size=n, replace=False)) # uniform sampling of subset of observations
    idx_F = np.sort(r.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]



def predictMPClass(X,Y,X1, n_ratio,m_ratio,B,fit_func,fun_para):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    clas=set(Y)

        
    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
#     predictions = np.zeros((B, N+N1), dtype=float)
    predictions=[]
    for b in range(B):
        [idx_I,idx_F,x_mp,y_mp] = buildMPClass(X,Y,n_ratio,m_ratio)
        model = fit_func(x_mp,y_mp,fun_para)
        prob = pd.DataFrame(model.predict_proba(X1[:, idx_F]), columns=set(y_mp))
        for i in (clas):
            if i not in prob.columns:
                prob[i]=0
    ############################################
        predictions.append(np.array(prob))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]



## mean inference 
      
def ztest(z,alpha,MM=1,bonf_correct=True):
    l = len(z)
    s = np.std(z)
    m = np.mean(z)
    pval1 = 1-norm.cdf(m/s*np.sqrt(l))

    pval2 = 2*(1-norm.cdf(np.abs(m/s*np.sqrt(l))))

    # Apply Bonferroni correction for M tests
    if bonf_correct:
        pval1= min(MM*pval1,1)
        pval2= min(MM*pval2,1)
        alpha = alpha/MM
    q = norm.ppf(1-alpha/2)
    left  = m - q*s/np.sqrt(l)
    right = m + q*s/np.sqrt(l)
    return [pval1,pval2, left,right]

## median inference 
def signtest(z,alpha,MM=1,bonf_correct=False):

    l = len(z)
    if l==0:
        return [0]*3
    s = np.sum(z>0)
    pval = 1-binom.cdf(s-1,l,0.5)
    # Apply Bonferroni correction for M tests
    if bonf_correct:
        pval = min(MM*pval,1)
        alpha = alpha/MM
    pp = int(l - binom.ppf(1-alpha/2,l,0.5)) # Want P(B <= j) <= alpha/2
    zz=sorted(z)
    if pp==0:
        return [nan]*3
    left  = zz[pp-1]
    right = zz[l-pp]
    return [pval,left,right]



############################
## LOCO on MP ensemble
#############################
def LOCOsplitMPClass(X,Y,n_ratio,m_ratio,B,fit_func,fun_para,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    ## SPLIT 
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    ## fit model on first part, predict on second 
    clas=set(y_train)
    #### fit MP model on 1st part 
    [predictions,in_mp_obs,in_mp_feature]= predictMPClass(x_train,y_train,x_val,n_ratio,m_ratio,B,fit_func,fun_para)
    ##############
    ## average     
    predictions = pd.DataFrame(predictions.mean(0), columns=clas)
    ## get nonconformal on second 
    resids_split = getNC(y_val, predictions)
    
    
    #Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z= np.zeros((len(ff),3))
    z={}
    quantile_z=np.zeros((len(ff),2))    
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        [out_js,in_mp_obs,in_mp_feature]= predictMPClass(np.delete(x_train,j,1),y_train,np.delete(x_val,j,1),n_ratio,m_ratio,B,fit_func,fun_para)
        out_j = pd.DataFrame(out_js.mean(0), columns=clas)

        resids_drop=getNC(y_val, out_j)
        z[idd] = resids_drop - resids_split

        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = [np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)]
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res



       
###################
####### LOCO
###################

def LOCOsplitClass(X,Y,fit_func,fun_para,selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    ## SPLIT 
    r = np.random.RandomState()
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    ## fit model on first part, predict on second 
    clas=set(y_train)
    model = fit_func(x_train,y_train,fun_para)
    prob = pd.DataFrame(model.predict_proba(x_val), columns=clas)
    resids_split = getNC(y_val, prob)

 # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
        
    inf_z= np.zeros((len(ff),3))
    z={}
    quantile_z=np.zeros((len(ff),2))
    
    for idd,j in enumerate(ff):
        model_out_j = fit_func(np.delete(x_train,j,1),y_train,fun_para) 
        prob_j = pd.DataFrame(model_out_j.predict_proba(np.delete(x_val,j,1)), columns=clas)
        resids_drop=getNC(y_val, prob_j)
        z[idd] = resids_drop - resids_split
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = ([np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)])
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res


 

def SimuAutoregressiveExactSparseClass(N,M,N1,SNR, seed=1):
    M1=int(0.05*M)
    np.random.seed(seed)
    mu = [0]*M
    cov = [[0]*M for _ in range(M)]
    for i in range(M-1):
        cov[i][i+1]=0.5
        cov[i+1][i]=0.5
    cov=np.array(cov)
    np.fill_diagonal(cov, 1)

    X = np.random.multivariate_normal(mu, cov,N)
    np.random.seed(seed*2+1)
    X1 =  np.random.multivariate_normal(mu, cov,N1)
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))
    pr = np.dot(X,beta)+np.random.normal(0, 1,N)
    pr=(np.exp(pr)/(1+np.exp(pr)))
    Y = np.random.binomial(1,pr)
    pr1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)
    pr1= (np.exp(pr1))/(1+np.exp(pr1))
    Y1 = np.random.binomial(1,pr1)

    return [X,Y,X1,Y1]

def SimuExactSparseClass(N,M,N1,SNR,M1=5,seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.rand(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    M1=int(0.05*M)
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))
    pr = np.dot(X,beta)+np.random.normal(0, 1,N)
    pr=(np.exp(pr)/(1+np.exp(pr)))
    Y = np.random.binomial(1,pr)

    pr1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)
    pr1= (np.exp(pr1))/(1+np.exp(pr1))
    Y1 = np.random.binomial(1,pr1)

    return (X,Y,X1,Y1)
def SimuNonlinearExactSparseClass(N,M,N1,SNR,M1=5,seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.rand(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    beta = np.append([SNR],np.random.normal(5,1,5))
    pr = beta[0]*X[:,0]*(X[:,0]<2 &(X[:,0]>-2))+beta[1]*X[:,1]*(X[:,1]<0)+(X[:,2]>0)*X[:,2]*beta[2]+(X[:,3]>0)*X[:,3]*beta[3]+ (np.sign(X[:,4]))*beta[4]+np.random.normal(0, 1,N)
    pr=(np.exp(pr)/(1+np.exp(pr)))
    Y = np.random.binomial(1,pr)
    pr1 = beta[0]*X1[:,0]*(X1[:,0]<2 &(X1[:,0]>-2))+beta[1]*X1[:,1]*(X1[:,1]<0)+(X1[:,2]>0)*X1[:,2]*beta[2]+(X1[:,3]>0)*X1[:,3]*beta[3]+ (np.sign(X1[:,4]))*beta[4]+np.random.normal(0, 1,N1)
    pr1= (np.exp(pr1))/(1+np.exp(pr1))
    Y1 = np.random.binomial(1,pr1)

    return (X,Y,X1,Y1)


def LOCOMPClass(X,Y,X1,Y1, n_ratio,m_ratio,B,fit_func,fun_para,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)
    clas=np.unique(Y)
    start_time=time.time()                
    [predictions,in_mp_obs,in_mp_feature]= predictMPClass(X,Y,X,n_ratio,m_ratio,B,fit_func,fun_para)
    predictions_train = predictions
    times= time.time()-start_time
    print(times)
    # Re-fit after dropping each feature
    zeros=False


    diff=[]
    print('find loo')
    start_time=time.time()                
    b_keep = pd.DataFrame(~in_mp_obs).apply(lambda i: np.array(i[i].index))

    #############################
    ## Find LOO
    ############################
    for i in range(N):
        ## find MP has no i but has j
#         b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
        #####################
        ###### estimate B
        sel_2 = np.array(sample(list(b_keep[i]),20))
        sel_2.shape = (2,10)
        diff.append(np.square(predictions_train[sel_2[0],i][:,0] - predictions_train[sel_2[1],i][:,0]).mean())
        
        
    with_j = map(lambda i: predictions_train[b_keep[i],i].mean(0),range(N))
    with_j = pd.DataFrame(list(with_j), columns=clas)

#         if len(b_keep)>0:
#             with_j[i]= predictions_train[b_keep[i],i].mean(0)

#     with_j = pd.DataFrame(with_j, columns=clas)

    resids_LOO = getNC(Y, with_j)
    times= time.time()-start_time
    print(times)

    ################################
    ######## FIND LOCO
    #############################

    print('find loco')
    start_time=time.time()                

    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    z={}
    resids_LOCOs={}
    inf_z = np.zeros((len(ff),4))
    for idd,j in enumerate(ff):
        out_j = np.zeros((N,len(clas)))
#         b_keep_f = pd.DataFrame(~in_mp_obs).apply(lambda i:np.sort(list(set(np.array(i[i].index))&set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)))))
#         out_j = map(lambda i: predictions_train[b_keep_f[i],i].mean(0),range(N))
#         out_j = pd.DataFrame(list(out_j), columns=clas)
        
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            out_j[i] = predictions_train[b_keep_f,i].mean(0)
        out_j = pd.DataFrame(out_j, columns=clas)
        resids_LOCO = getNC(Y, out_j)

        zz = resids_LOCO - resids_LOO
        z[idd] = zz[~np.isnan(zz)]

        resids_LOCOs[idd] = resids_LOCO.copy()
        if len(z)==0:
            inf_z[idd]= [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =bonf)
    times= time.time()-start_time
    print(times)

    ###########################
    res= {}
    res['loco_ci']=inf_z
    res['z']=z
    res['diff']=diff
    return res    
def LOCOSplitClass(X,Y,X1,Y1, fit_func,fun_para,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)
    clas=np.unique(Y)
    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    clas=set(y_train)


    model = fit_func(x_train,y_train,fun_para)
    prob = pd.DataFrame(model.predict_proba(x_val), columns=clas)
    prob_test = pd.DataFrame(model.predict_proba(X1), columns=clas)
    resids_split = getNC(y_val, prob)
    resids_split_test = getNC(Y1, prob_test)
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features

    inf_z= np.zeros((len(ff),4))
    z={}
    quantile_z=np.zeros((len(ff),2))
    resids_drop,resids_drop_test = {},{}
    for idd,j in enumerate(ff):
        model_out_j = fit_func(np.delete(x_train,j,1),y_train,fun_para)
        prob_j = pd.DataFrame(model_out_j.predict_proba(np.delete(x_val,j,1)), columns=clas)
        prob_j_test = pd.DataFrame(model_out_j.predict_proba(np.delete(X1,j,1)), columns=clas)
        resids_drop[idd]=getNC(y_val, prob_j)
        resids_drop_test[idd]=getNC(Y1, prob_j_test)

        z[idd] = resids_drop[idd] - resids_split
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)

    ###########################
    res= {}
    res['loco_ci']=inf_z
    res['z']=z


    return res


from joblib import Parallel, delayed
def LOCOMPClass_pare(X,Y,X1,Y1, n_ratio,m_ratio,B,fit_func,fun_para,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)
    clas=np.unique(Y)
    [predictions,in_mp_obs,in_mp_feature]= predictMPClass(X,Y,X,n_ratio,m_ratio,B,fit_func,fun_para)
    predictions_train = predictions

    zeros=False


    diff=[]
    print('find loo')
    start_time=time.time()                
    b_keep = pd.DataFrame(~in_mp_obs).apply(lambda i: np.array(i[i].index))

    #############################
    ## Find LOO
    ############################
    for i in range(N):
        #####################
        ###### estimate B
        sel_2 = np.array(sample(list(b_keep[i]),20))
        sel_2.shape = (2,10)
        diff.append(np.square(predictions_train[sel_2[0],i][:,0] - predictions_train[sel_2[1],i][:,0]).mean())
        
        
    with_j = map(lambda i: predictions_train[b_keep[i],i].mean(0),range(N))
    with_j = pd.DataFrame(list(with_j), columns=clas)
    resids_LOO = getNC(Y, with_j)

    ################################
    ######## FIND LOCO
    #############################
    def get_loco(i,j):
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
        return predictions_train[b_keep_f,i].mean(0)

    print('find loco')

    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    results = Parallel(n_jobs=-1)(delayed(get_loco)(i,j) for i in range(N) for j in range(M))
    ress = pd.DataFrame(results)
    ress['i'] = np.repeat(range(N),M)
    ress['j'] = np.tile(range(M),N)
    ress['true_y'] = np.repeat(Y,M)
    ress['resid_loco'] = getNC(ress['true_y'], ress[[0,1]])
    ress['resid_loo'] = np.repeat(resids_LOO,M)
    ress['zz'] = ress['resid_loco'] -ress['resid_loo']


    inf_z = np.zeros((len(ff),4))
    for idd,j in enumerate(ff): 
        inf_z[idd] = ztest(ress[ress.j==idd].zz,alpha,MM=len(ff),bonf_correct =bonf)

    ###########################
    res= {}
    res['loco_ci']=inf_z
    res['info']=ress
    res['diff']=diff
    return res    