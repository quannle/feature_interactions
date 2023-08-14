import functions as mp
import numpy as np
from multiprocessing import Pool

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
        # MLPRegressor(solver="lbfgs", max_iter=500),
        RandomForestRegressor(n_estimators=50, max_depth=None),
        KernelRidge(alpha=.001, kernel="polynomial", degree=2),
        KernelRidge(alpha=.001, kernel="rbf")
    ]



    return 0


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
    # Y += snr * X[:, J1] 
    Y += X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4]
    Y += (X[:, 1] * X[:, 2]) + (X[:, 2] * X[:, 3]) + (X[:, 3] * X[:, 4])
    Y += snr * (X[:, J1] * X[:, J2])

    # Y += snr * (X[:, J1] > 0) * (X[:, J2] > 0)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    return f(X, Y, n_ratio, m_ratio, B, model, J1, J2, np.abs) 

def h(SNR, models, M, N, K, n_ratio, m_ratio, B, J1, J2, num_trials)
    residuals   = np.empty((len(SNR), 4, len(models)))
    powers      = np.empty((len(SNR), 6, len(models)))
    errors      = np.empty((len(SNR), 6, len(models)))
    importance  = np.empty((len(SNR), 6, len(models)))

    iterable = [(M, N, K, n_ratio, m_ratio, B, model, J1, J2, snr)
                for model in models for i in range(num_trials)]

    with Pool() as p:
        iterator = p.starmap(g, iterable)

    if num_trials != 1:
        print("you neglected to implement this")
        sys.exit(1)

    for result in iterable:
        null_rejection = np.array([r["!h_0"] for r in result])
        residual = np.array([r["errors"] for r in result])

        errors[i, :, j] = np.std(null_rejection, axis = 0)
        powers[i, :, j] = np.mean(null_rejection, axis = 0)
        residuals[i, :, j] = np.mean(residual, axis = 0)
        importance[i, :, j] = np.mean([r["delta"] for r in result], axis = 0)


    for i, snr in enumerate(SNR):
        results = Parallel(n_jobs=-1)(
                  delayed(h)(M, N, K, n_ratio, m_ratio, B, model, J1, J2, snr)
                  for i in range(num_trials) for model in models)


        for j, model in enumerate(models):
            result = results[j::len(models)]


            # for k, hmetric in enumerate(["H21", "H22"]):
            #     h2results = np.array(r[hmetric] for r in result)
            #     H2[i, k, j] = np.mean(h2results)
            #     H2err[i, k, j] = np.std(h2results)



        print(f"\rfinished run {i} {snr}", end="")

if __name__ == "__main__":
    main()
