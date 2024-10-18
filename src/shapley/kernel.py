import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from time import perf_counter

import itertools
import scipy.special


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 10000
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


def exact_regression(df_X, predict_fct, reference, do_print=False):
    """computes kernel shap exact regression."""
    features = df_X.columns
    if do_print:
        print("--------------------------")
        print("Building prediction dataset", end=" ")
        t1 = perf_counter()
    N, M = df_X.shape[0], df_X.shape[1]
    X = np.zeros((2**M * N, (M + 1)))
    X[:, -1] = 1
    weights = np.zeros(2**M)
    
    if reference is None: #random vector each time
        V = background.iloc[np.random.choice(background.shape[0], 2**M * N, replace=True)].values
    else:
        V = np.zeros((2**M * N, M))
        for i in range(2**M * N):
            V[i, :] = reference.copy()

    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        weights[i] = shapley_kernel(M, len(s))
        for k in range(N):
            V[k*2**M+i, s] = df_X.values[k,s]
            X[k*2**M+i, s] = 1
    wsq = np.sqrt(weights)
    if do_print:
        t2 = perf_counter()
        print(f"({(t2-t1):.3f}s)")

    if do_print:
        print("--------------------------")
        print("Computing predictions", end=" ")
        t1 = perf_counter()
    y = predict_fct(pd.DataFrame(V,columns=features))
    if do_print:
        t2 = perf_counter()
        print(f"({(t2-t1):.3f}s)")

    if do_print:
        print("--------------------------")
        print("Computing regressions",end=" ")
        t1 = perf_counter()
    shapley_values = []
    base_values = []
    for k in range(N):
        result = np.linalg.lstsq(wsq[:, None] * X[k*2**M:(k+1)*2**M], wsq * y[k*2**M:(k+1)*2**M], rcond=None)[0]
        shapley_values.append(result[:-1])
        base_values.append(result[-1])
    if do_print:
        t2 = perf_counter()
        print(f"({(t2-t1):.3f}s)")
    return shapley_values