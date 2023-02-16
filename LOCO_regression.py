import numpy as np
import pandas as pd
from scipy.stats import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy.random as r
from scipy.stats import *
from random import sample

from joblib import Parallel, delayed
def LOCOMPReg(X,Y,n_ratio,m_ratio,B,fit_func, selected_features=[0],alpha=0.1,bonf=True):

    N=len(X)
    M = len(X[0])

    [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,X,n_ratio,m_ratio,B,fit_func)
    predictions_train = predictions



    # Re-fit after dropping each feature
    zeros=False

    #############################
    ## Find LOO
    ############################
    diff= []
    b_keep = pd.DataFrame(~in_mp_obs).apply(lambda i: np.array(i[i].index))
    for i in range(N):
        sel_2 = np.array(sample(list(b_keep[i]),20))
        sel_2.shape = (2,10)
        diff.append(np.square(predictions_train[sel_2[0],i] - predictions_train[sel_2[1],i]).mean())
         ##################################

    resids_LOO = list(map(lambda i: np.abs(Y[i] - predictions_train[b_keep[i],i].mean()),range(N)))


    ################################
    ######## FIND LOCO
    #############################
    def get_loco(i,j):
        b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
        return predictions_train[b_keep_f,i].mean()
    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    results = Parallel(n_jobs=-1)(delayed(get_loco)(i,j) for i in range(N) for j in range(M))
    ress = pd.DataFrame(results)
    ress['i'] = np.repeat(range(N),M)
    ress['j'] = np.tile(range(M),N)
    ress['true_y'] = np.repeat(Y,M)
     
    ress['resid_loco'] =np.abs(ress['true_y'] - ress[0])
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


def buildMP(X,Y,n_ratio,m_ratio):
    N = len(X)
    M = len(X[0])
    n = np.int(np.round(n_ratio * N))
    m = np.int(np.round(m_ratio * M))
    r = np.random.RandomState()
    ## index of minipatch
    idx_I = np.sort(r.choice(N, size=n, replace=False)) # uniform sampling of subset of observations
    idx_F = np.sort(r.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp=X[np.ix_(idx_I, idx_F)]
    y_mp=Y[np.ix_(idx_I)]
    return [idx_I,idx_F,x_mp,y_mp]
def predictMP(X,Y,X1, n_ratio,m_ratio,B,fit_func):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B):        
        [idx_I,idx_F,x_mp,y_mp] = buildMP(X,Y,n_ratio,m_ratio)
        predictions.append(fit_func(x_mp,y_mp,X1[:, idx_F]))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]


def ztest(z,alpha,MM=1,bonf_correct=True):
    try:
        s = np.std(z)
    except:
        return [0,0,0,0]
    
    l = len(z)
    s = np.std(z)
    if s==0:
        return [0,0,0,0]
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

