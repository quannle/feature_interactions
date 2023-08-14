
import sys
import functions as mp
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def main():
    M = 25
    N = 1000
    K = 0
    J1 = 0
    J2 = 1

    SNR = np.linspace(0, 5, 10)

    m_ratio = np.sqrt(M) / M
    n_ratio = np.sqrt(N) / N

    B = 1000 # from 1k to 10k
    num_trials = 1
    models = [
        MLPRegressor(solver="lbfgs", max_iter=500),
        RandomForestRegressor(n_estimators=50, max_depth=None),
        KernelRidge(alpha=.001, kernel="polynomial", degree=2),
        KernelRidge(alpha=.001, kernel="rbf")
    ]

    if len(sys.argv) == 1:
        return h(SNR, models, M, N, K, n_ratio, m_ratio, B, J1, J2, num_trials)

    with open("importance.npy", "rb") as f:
        importance = np.load(f)
        plot_importance(models, SNR, importance)




def f(X, Y, n_ratio, m_ratio, B, model, J1, J2, metric): 
    """
    Accepts numpy arrays X, Y, integers n_ratio, m_ratio, ...
    Performs a one-sided significance test for each metric, including iLOCO
    """
    ensemble = mp.Ensemble(model).fit(X, Y, n_ratio, m_ratio, B)
    predictions = ensemble.predict(X)
    r12, r1, r2, r = mp.computeDeltaCap(Y, J1, J2, 
                                        predictions,
                                        ensemble.mp_observations, 
                                        ensemble.mp_features, 
                                        metric=metric)

    deltas = [r1 + r2 - r12 - r, 
              r12 - r,
              r1 - r, 
              r2 - r,
              r12 - r1,
              r12 - r2]

    result = {}
    result["!h_0"] = len(deltas) * [0]
    result["delta"] = len(deltas) * [0]

    for i, delta in enumerate(deltas):
        ci = mp.getCI(delta, alpha=0.1)
        delta_bar = np.mean(delta)
        result["!h_0"][i] = 1 if (delta_bar - ci > 0) else 0
        result["delta"][i] = delta_bar

    result["errors"] = [np.mean(r), np.mean(r1), np.mean(r2), np.mean(r12)]


    return result



def g(M, N, K, n_ratio, m_ratio, B, model, J1, J2, snr):
    X, Y = mp.kSparseLinearModel(N, M, K)

    Y += X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4]
    # Y += snr * X[:, J1] 
    Y += (X[:, 1] * X[:, 2]) + (X[:, 2] * X[:, 3]) + (X[:, 3] * X[:, 4])
    Y += snr * (X[:, J1] * X[:, J2])

    # Y += snr * (X[:, J1] > 0) * (X[:, J2] > 0)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    return f(X, Y, n_ratio, m_ratio, B, model, J1, J2, np.abs) 

def h(SNR, models, M, N, K, n_ratio, m_ratio, B, J1, J2, num_trials):
    residuals   = np.empty((len(models), len(SNR), 4))
    powers      = np.empty((len(models), len(SNR), 6))
    errors      = np.empty((len(models), len(SNR), 6))
    importance  = np.empty((len(models), len(SNR), 6))

    iterable = [(M, N, K, n_ratio, m_ratio, B, model, J1, J2, snr)
                for model in models for snr in SNR]

    with Pool() as p:
        iterator = iter(p.starmap(g, iterable))

    if num_trials != 1:
        print("you neglected to implement this")
        sys.exit(1)


    for i, model in enumerate(models):
        for j, snr in enumerate(SNR):
            result = next(iterator)
            importance[i, j, :] = result["delta"]
    
    print(importance)

    with open("importance.npy", "wb") as f:
        np.save(f, importance)
    return 0

def plot_importance(models, SNR, importance):
    model_names = [model.__class__.__name__ 
            if model.__class__.__name__ != KernelRidge.__name__ 
            else model.__class__.__name__ + " " + model.get_params()["kernel"]
            for model in models]

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    for i, model_name in enumerate(model_names):   
        ax1.plot(SNR, importance[i, :, 0])
        # ax1.plot(SNR, importance[:, 2, j] + importance[:, 3, j] - importance[:, 1, j])
        

    ax1.set_title("Feature Importance")
    ax1.set_xlabel("SNR values")
    ax1.set_ylabel(f"Feature Importance")
    ax1.legend(model_names)

    plt.show()


if __name__ == "__main__":
    main()
