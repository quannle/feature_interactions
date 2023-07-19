import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

def kSparseLinearModel(N, M, K):
    """
    Generates a linear model. The coefficients, beta, are K-sparse 
    and distributed ~ N(0, 2). Returns:
    - X = a numpy array with M rows and N columns.
    - Y = X @ beta + N(0, 1). A linear combination of the features in X with standard normal error. 
    """
    X = np.random.normal(0, 1, (N, M))
    # k-sparse array of coefficients
    beta = np.concatenate((np.random.normal(0, 2, K), np.zeros((M - K)))) 
    Y = X @ beta + np.random.normal(0, 1, N)
    # Y = (Y - np.mean(Y)) / np.std(Y)
    return X, Y


def buildMP(X, Y, n_ratio, m_ratio):
    """
    Builds a minipatch. Returns:
    - idx_I = the chosen subset of observation indices
    - idx_F = the chosen subset of feature indices
    - x_mp  = the minipatch of observations
    - y_mp  = the minipatch of responses
    """

    N, M = len(X), len(X[0])
    n = np.int(n_ratio * N)
    m = np.int(m_ratio * M)
    
    # uniformly sample a subset of observations
    idx_I = np.random.choice(N, size=n, replace=False)
    idx_I.sort()
    # uniformly sample a subset of features
    idx_F = np.random.choice(M, size=m, replace=False)
    idx_F.sort()

    ## record which obs/features are subsampled 
    x_mp = X[np.ix_(idx_I, idx_F)]
    y_mp = Y[np.ix_(idx_I)]
    return idx_I, idx_F, x_mp, y_mp


class Ensemble:
    def __init__(self, model):
        self.base = model

    def fit(self, X, Y, n_ratio, m_ratio, B):
        N, M = X.shape
        self.mp_observations = np.zeros((N, B), dtype=bool)
        self.mp_features = np.zeros((M, B), dtype=bool)
        self.ensemble = [None] * B
        for b in range(B):
            idx_I, idx_F, x_mp, y_mp = buildMP(X, Y, n_ratio, m_ratio)
            self.ensemble[b] = self.base.fit(x_mp, y_mp) 
            self.mp_observations[idx_I, b] = True
            self.mp_features[idx_F, b] = True  
        return self
        
    def predict(self, X):
        predictions = np.empty((len(X), len(self.ensemble)))
        for b, m in enumerate(self.ensemble):
            predictions[:, b] = m.predict(X[:, self.mp_features[:, b]])
        return predictions
    


def predict(X, Y, n_ratio, m_ratio, B, model, model_param):
    """
    Fits and predicts models
    """
    N, M = X.shape
    mp_observations = np.zeros((N, B), dtype=bool)
    mp_features = np.zeros((M, B), dtype=bool)
    predictions = np.empty((N, B))
    for b in range(B):
        idx_I, idx_F, x_mp, y_mp = buildMP(X, Y, n_ratio, m_ratio)
        predictions[:, b] = model.fit(x_mp, y_mp).predict(X[:, idx_F])
        mp_observations[idx_I, b] = True
        mp_features[idx_F, b] = True  
    return predictions, mp_observations, mp_features




def computeDeltaCap(Y, j1, j2, predictions, mp_observations, mp_features, metric=np.square):
    """
    TODO: apply bonferroni correction if more than one test is made.
    Computes the squared error vectors from LOCO and LOO predictions.
    """
    
    loo = 1 - mp_observations
    mu_loo = np.sum(predictions * loo, axis=1) / np.sum(loo, axis=1)
    
    loco1 = loo * (1 - mp_features[j1, :])
    mu_loco1 = np.sum(predictions * loco1, axis=1) / np.sum(loco1, axis=1)

    loco2 = loo * (1 - mp_features[j2, :])
    mu_loco2 = np.sum(predictions * loco2, axis=1) / np.sum(loco2, axis=1)

    loco12 = loco1 * loco2
    mu_loco12 = np.sum(predictions * loco12, axis=1) / np.sum(loco12, axis=1)

    # fig, axs = plt.subplots(4, 1, figsize=(10,10))
    # axs[0].spy(loo)
    # axs[1].spy(loco1)
    # axs[2].spy(loco2)
    # axs[3].spy(loco12)
    # plt.show()
    # exit(0)


    residual_loo = metric(Y - mu_loo)
    residual_loco1 = metric(Y - mu_loco1)
    residual_loco2 = metric(Y - mu_loco2)
    residual_loco12 = metric(Y - mu_loco12)

    return residual_loco12, residual_loco1,  residual_loco2, residual_loo    

def getCI(delta_cap, alpha=0.1):
    """Get confidence interval width / 2
    """
    sigma = np.std(delta_cap, ddof=1)
    ci = norm.ppf(1 - alpha / 2) * sigma / np.sqrt(len(delta_cap))
    return ci


def featureInteractions(X, Y, n_ratio, m_ratio, B, model, model_param, j1, j2, alpha=0.1):
    predictions, mp_observations, mp_features = predict(X, Y, n_ratio, m_ratio, B, model, model_param)
    r12, r1, r2, r = computeDeltaCap(Y, j1, j2, predictions, mp_observations, mp_features)
    dc = r12 - r1 - r2 + r
    # dc = r1 - r
    ci = getCI(dc, alpha)
    return np.mean(dc), ci


