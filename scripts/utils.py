import autograd.numpy as np
from lifelines.fitters import ParametricUnivariateFitter

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

    
def generate_sample_edf(sample, n=100):
    # Compute the empirical distribution function (EDF)
    sample_sorted = np.sort(sample)
    sample_size = len(sample_sorted)
    y = np.arange(1, sample_size + 1) / sample_size
    #rand_num = np.random.uniform(0, 1, size=n)
    rand_num = np.linspace(0, 1, n)
    index = np.searchsorted(y, rand_num)
    return sample_sorted[index]

class GompertzFitter(ParametricUnivariateFitter):
    # this parameterization is slightly different than wikipedia.
    _fitted_parameter_names = ['nu_', 'b_']
    def __init__(self, times):
        super(GompertzFitter, self).__init__()
        self.Tmax = np.max(times)
    def _cumulative_hazard(self, params, times):
        nu_, b_ = params
        return nu_ * (np.expm1(times * b_/self.Tmax))

def generate_gompertz_sample(A, B, Tmax, n=100):
    u = np.random.uniform(0, 1, n)
    x = Tmax / B * np.log(1 - np.log(1 - u) / A)
    return x