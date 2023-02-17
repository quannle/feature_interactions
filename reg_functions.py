import vimpy
import numpy.random as r
from tensorflow import keras
# import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from sklearn.linear_model import RidgeCV
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import time
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from scipy.stats import *
from sklearn import preprocessing
# keras.backend.clear_session()
from random import sample
### autoregressive model 
 

      
def ztest(z,alpha,MM=1,bonf_correct=True):
    try:
        s = np.std(z)
    except:
        return [0,0,0,0]
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



# Define Baseline Regressor to be used later
def DecisionTreeReg(X,Y,X1,fun_para):
    tree = DecisionTreeRegressor().fit(X,Y)
    return tree.predict(X1)
def ridgecv(X,Y,X1,ridge_mult=0.001):
    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], fit_intercept = False).fit(X, Y)
    # named ridge2 to distinguish it from the ridge regressor in sklearn.
    
    return clf.predict(X1)

def ridge2(X,Y,X1,ridge_mult=0.001):
    clf = Ridge(fit_intercept = False,alpha=ridge_mult).fit(X, Y)
    # named ridge2 to distinguish it from the ridge regressor in sklearn.
    
    return clf.predict(X1)

def RFreg(X,Y,X1,ntree=1000):
    # when bootstrap=False, it means each tree is trained on all rows of X and only
    #      subsamples its columns (which are features).
    rf = RandomForestRegressor(n_estimators=200,criterion='mse',bootstrap=False).fit(X,Y)
    return rf.predict(X1)
from joblib import Parallel, delayed

# kernel svm 
from sklearn.svm import SVR
def kernelSVR(X,Y,X1,ridge_mult=0.001):
    krr = SVR(kernel='rbf',C=1/ridge_mult).fit(X, Y)
    return krr.predict(X1)
## knn 
from sklearn.neighbors import KNeighborsRegressor
def knnReg(X,Y,X1,ridge_mult=0.001):
    neigh = KNeighborsRegressor(n_neighbors=10,weights='distance').fit(X, Y)
    return neigh.predict(X1)
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense

def MLPreg(X,Y,X1,ridge_mult=0.001):
    M = len(X[0])
    model = Sequential()
    model.add(Dense(M, input_dim=M, activation='relu'))
    model.add(Dense(M, input_dim=M, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X,Y, epochs=50, batch_size=10,verbose=0)
    yhat=[i[0] for i in (model.predict(X1))]
    return yhat

def bestLambdaReg(X,Y, test_size=0.333, tune_iter = 5, lambdas = np.logspace(-3, 1, 60)):
    mse = pd.DataFrame(columns = ['lambda','iter','mse'])
    for lamb in lambdas: 
            for iii in range(tune_iter):
                x, x1, y, y1 = train_test_split(X,Y, test_size=test_size)
                clf = Ridge(normalize = True,alpha=lamb)
                y_hat = clf.fit(x, y).predict(x1)
                msee = np.sqrt(np.mean((y1-y_hat)**2))
                mse.loc[len(mse)]=[lamb,iii,msee]
    ss=pd.DataFrame(mse.groupby(['lambda'],as_index=False)['mse'].mean())
    best_lambda = ss.loc[ss['mse'].idxmin()]
    lamb = float(best_lambda['lambda'])

    return lamb
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
def predictMP(X,Y,X1, n_ratio,m_ratio,B,fit_func,fun_para):
    N = len(X)
    M = len(X[0])
    N1 = len(X1)
    in_mp_obs,in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    predictions=[]
    for b in range(B):        
        [idx_I,idx_F,x_mp,y_mp] = buildMP(X,Y,n_ratio,m_ratio)
        predictions.append(fit_func(x_mp,y_mp,X1[:, idx_F],fun_para))
        in_mp_obs[b,idx_I]=True
        in_mp_feature[b,idx_F]=True  
    return [np.array(predictions),in_mp_obs,in_mp_feature]



## mean inference 

## median inference 
def signtest(z,alpha,MM=1,bonf_correct=True):

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

# Define Baseline Regressor to be used later
def DecisionTree(X,Y,X1,fun_para):
    tree = DecisionTreeRegressor().fit(X,Y)
    return tree.predict(X1)
########################
## our methods
########################


from joblib import Parallel, delayed



###################
####### LOCO
###################

def LOCOsplitReg(X,Y,fit_func,fun_para,selected_features=[0],alpha=0.1,bonf=False):
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
    predictions = fit_func(x_train,y_train,(x_val),fun_para)
    ## get residual on second 
    resids_split = np.abs(y_val - predictions)
 # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z= np.zeros((len(ff),4))
    z={}
    quantile_z=np.zeros((len(ff),2))
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        out_j = fit_func(np.delete(x_train,j,1),y_train,np.delete(x_val,j,1),fun_para)
        resids_drop=np.abs(y_val - out_j)
        z[idd] = resids_drop - resids_split
        inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)
        quantile_z[idd] = ([np.quantile(z[idd],alpha/2),np.quantile(z[idd],1-alpha/2)])
    res= {}
    res['loco_ci']=inf_z
    res['loco_q']=quantile_z
    res['z']=z
    return res

        
 


from scipy.stats import *
 
def LOCOMPReg(X,Y,X1, Y1, n_ratio,m_ratio,B,fit_func,fun_para,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):

    N=len(X)
    M = len(X[0])
    N1=len(X1)

    [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,X,n_ratio,m_ratio,B,fit_func,fun_para)
#     [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,np.vstack((X,X1)),n_ratio,m_ratio,B,fit_func,fun_para)
    predictions_train = predictions



    # Re-fit after dropping each feature
    zeros=False

    #############################
    ## Find LOO
    ############################
    diff= []
    b_keep = pd.DataFrame(~in_mp_obs).apply(lambda i: np.array(i[i].index))
    for i in range(N):
#         ## find MP has no i but has j
#         b_keep = list(set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))

#         #####################
#         ###### estimate B
        sel_2 = np.array(sample(list(b_keep[i]),20))
        sel_2.shape = (2,10)
        diff.append(np.square(predictions_train[sel_2[0],i][:,0] - predictions_train[sel_2[1],i][:,0]).mean())
         ##################################

    resids_LOO = map(lambda i: Y[i] - predictions_train[b_keep[i],i].mean(),range(N))


        ################################
    ######## FIND LOCO
    #############################
    if len(selected_features)==0:
        ff = list(range(M))
    else:
        ff=selected_features
    inf_z= np.zeros((len(ff),4))
    z={}
    resids_LOCOs={}

    for idd,j in enumerate(ff):
    #    print(idd,j)
        for i in range(N):
            b_keep_f = list(set(np.argwhere(~(in_mp_feature[:,j])).reshape(-1)) & set(np.argwhere(~(in_mp_obs[:,i])).reshape(-1)))
            resids_LOCO[i]= np.abs(Y[i] - predictions_train[b_keep_f,i].mean())
        zz = resids_LOCO - resids_LOO
        z[idd] = zz[~np.isnan(zz)]
        resids_LOCOs[idd] = resids_LOCO.copy()

        zz = resids_LOCO_sq - resids_LOO_sq
        z_sq[idd] = zz[~np.isnan(zz)]


        if len(z)==0:
            inf_z[idd]= [0]*4
        else:
            inf_z[idd] = ztest(z[idd],alpha,MM=len(ff),bonf_correct =True)

    ###########################
    res= {}
    res['diff']=diff
    res['loco_ci']=inf_z
    res['z']=z

    return res
 

    
def SimuAutoregressiveExactSparse(N,M,N1,SNR, seed=1):
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
    np.random.rand(seed*2+1)
    X1 =  np.random.multivariate_normal(mu, cov,N1)
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))

    Y = np.dot(X,beta)++np.random.normal(0, 1,N)

    Y1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)
    Y = preprocessing.scale(Y)
    Y1 = preprocessing.scale(Y1)
    return [X,Y,X1,Y1]

def SimuRealLinearExactSparse5per(N,M,N1,SNR, seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.rand(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    M1=int(0.05*M)
    if M1 == 0: 
        M1 = 1
    beta = np.append(np.append(np.array([SNR]),np.random.normal(5, 1,M1-1)),np.array([0]*(M-M1)))

    Y = np.dot(X,beta)+np.random.normal(0, 1,N)
    Y1 = np.dot(X1,beta)+np.random.normal(0, 1,N1)

    Y = preprocessing.scale(Y)
    Y1 = preprocessing.scale(Y1)

    return [X,Y,X1,Y1]

 
def SimuNonlinearExactSparse(N,M,N1,SNR, seed=1):
    np.random.seed(seed)
    X =  np.random.normal(0, 1, (N, M))
    np.random.seed(seed*2+1)
    X1 =  np.random.normal(0, 1, (N1, M))
    beta = np.append([SNR],np.random.normal(5,1,5))
#     beta_noise = np.array(np.random.normal(0,0.1,M-6))

    Y = beta[0]*X[:,0]*(X[:,0]<2 &(X[:,0]>-2))+beta[1]*X[:,1]*(X[:,1]<0)+(X[:,2]>0)*X[:,2]*beta[2]+(X[:,3]>0)*X[:,3]*beta[3]+ (np.sign(X[:,4]))*beta[4]+np.random.normal(0, 1,N)
#     Y= Y+ np.dot(X[:,6:],beta_noise)
    Y1 = beta[0]*X1[:,0]*(X1[:,0]<2 &(X1[:,0]>-2))+beta[1]*X1[:,1]*(X1[:,1]<0)+(X1[:,2]>0)*X1[:,2]*beta[2]+(X1[:,3]>0)*X1[:,3]*beta[3]+ (np.sign(X1[:,4]))*beta[4]+np.random.normal(0, 1,N1)
#     Y1= Y1+ np.dot(X1[:,6:],beta_noise)

    Y = preprocessing.scale(Y)
    Y1 = preprocessing.scale(Y1)


    print(beta)
    return [X,Y,X1,Y1]


def LOCOSplitReg(X,Y,X1, Y1,fit_func,fun_para,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):
    N=len(X)
    M = len(X[0])
    N1=len(X1)

    idx_I = np.sort(r.choice(N, size=int(0.5*N), replace=False)) # uniform sampling of subset of observations
    x_train=X[idx_I,:]
    y_train=Y[np.ix_(idx_I)]
    out_I =list(set(range(N))-set(idx_I))
    x_val = X[out_I]
    y_val = Y[out_I]
    
    
    ## fit model on first part, predict on second 
    prediction = fit_func(x_train,y_train,np.vstack((x_val,X1)),fun_para)
    predictions  = prediction[:len(y_val)]
    predictions_test = prediction[len(y_val):]
    ## get residual on second 
    resids_split = np.abs(y_val - predictions)
    resids_split_sq = (y_val - predictions)**2

    resids_split_test = np.abs(Y1 - predictions_test)
    resids_split_test_sq = np.square(Y1 - predictions_test)

    # Re-fit after dropping each feature    
    if len(selected_features)==0:
        ff = range(M)
    else:
        ff=selected_features
    inf_z,inf_z_sq= np.zeros((len(ff),4)),np.zeros((len(ff),4))
    
    uhat_j_test={}
    
    for idd,j in enumerate(ff):
        # Train on the first part, without variable j, predict on 2nd without j 
        out_js = fit_func(np.delete(x_train,j,1),y_train,np.delete(np.vstack((x_val,X1)),j,1),fun_para) 
        
        out_j  = out_js[:len(y_val)]
        uhat_j_test[idd] = out_js[len(y_val):]
    
        
        resids_drop=np.abs(y_val - out_j)
        resids_drop_sq = np.mean((y_val - out_j)**2)
        z = resids_drop - resids_split
        z_sq= resids_drop_sq - resids_split_sq
        inf_z[idd] = ztest(z,alpha,bonf_correct =True)
        inf_z_sq[idd] = ztest(z_sq,alpha,bonf_correct =True)


    ###########################
    res= {}
    res['loco_ci']=inf_z
    res['z']=z



        
    return res 
 
from joblib import Parallel, delayed

def LOCOMPReg_pare(X,Y,X1, Y1, n_ratio,m_ratio,B,fit_func,fun_para,LOCO = True, selected_features=[0],alpha=0.1,bonf=False):

    N=len(X)
    M = len(X[0])
    N1=len(X1)

    [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,X,n_ratio,m_ratio,B,fit_func,fun_para)
#     [predictions,in_mp_obs,in_mp_feature]= predictMP(X,Y,np.vstack((X,X1)),n_ratio,m_ratio,B,fit_func,fun_para)
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