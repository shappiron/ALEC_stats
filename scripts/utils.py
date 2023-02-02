import numpy as np

def convert_to_lifetimes(x, y, n):
    coef = 100 / n
    y_true = np.array(y) // coef * coef
    number_of_died = np.abs(np.diff(y_true) / coef)
    niter = np.concatenate([np.array([0]), number_of_died])

    lifetimes = []
    for i, d in enumerate(niter):
        if d != 0:
            lifetimes = lifetimes + [x[i]] * int(np.round(d))
    if len(lifetimes) != n:
        print('Warning: Computed number of lifetimes != n')
    #assert len(lifetimes) == n, 'Something wrong!'
    return lifetimes