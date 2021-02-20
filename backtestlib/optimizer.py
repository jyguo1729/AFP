import logging as log
import sys
log.basicConfig(level = log.INFO, stream=sys.stdout,format='backtest-%(asctime)s-%(funcName)s: %(message)s', datefmt='%Y-%b-%d %H:%M:%S')

import numpy as np
import numpy.linalg as LA
import scipy.optimize

class Optimizer:
    """ An optimizer converts a signal to positions
    """
    
    def __init__(self, **kw):
        """ Pass into parameters (e.g. max position), if any.
        """
        pass

    def optimize(self, signal, **kw):
        """ 
        Pass into signals and other parames, if any.
        Returns position of each symbol.
        """
        pass
        
class BenchMark(Optimizer):
    """
    This optimizer longs 1 dollar, with equal position for each stock.
    """

    def __init__(self, **kw):
        log.info(f"BenchMark initialized")
        
    def optimize(self, signal=None, **kw):
        """ Builds equal position for all stocks.
        """
        
        pos = 1 / len(signal)*np.ones_like(signal)
        return np.nan_to_num(pos, 0)
        
        
class SampleBased(Optimizer):
    """
    This optimizer given postion as variance inverse signal
    """

    def __init__(self, **kw):
        log.info(f"SampleBased initialized")
        
    def optimize(self, signal=None, sample_var: np.ndarray = None, **kw):
        """ Builds variance inverse signal position for all stock  sample_variance is n*n square matrix, with n = len(signal)
        """
        assert sum(np.isnan(signal)) == 0
        pos = LA.inv(sample_var)@np.nan_to_num(signal.values, 0)
        pos/=np.sum(pos)
        return np.nan_to_num(pos, 0)
        
class ShrinkToIdentity(Optimizer):
    """
    This optimizer given postion as variance inverse signal
    """

    def __init__(self, **kw):
        log.info(f"ShrinkToIdentity initialized")
        
    def optimize(self, signal=None, sample_var: np.ndarray = None, **kw):
        """ Builds variance inverse signal position for all stock  sample_variance is n*n square matrix, with n = len(signal)
        """
        assert sum(np.isnan(signal)) == 0
        raise
        pos = LA.inv(sample_var)@np.nan_to_num(signal.values, 0)
        pos/=np.sum(pos)
        return np.nan_to_num(pos, 0)
        
        
class SpectralCut(Optimizer):
    """
    This optimizer uses SpectralCut
    """

    def __init__(self,delta = 0.1, **kw):
        log.info(f"SpectralCut initialized with  delta {delta}")
        self.delta = delta
    def optimize(self, signal=None, sample_var: np.ndarray = None, **kw):
        """ Builds variance inverse signal position for all stock  sample_variance is n*n square matrix, with n = len(signal)
        """
        assert sum(np.isnan(signal)) == 0
        n = len(sample_var)
        u,s, vh = LA.svd(sample_var)
        mu = signal.values
        delta = self.delta

        def get_appx_error(K):
            cut = np.zeros(n)
            cut[:K] = 1.
            A = u@np.diag(cut)@vh
            mu_K = A@mu
            appx_error = LA.norm(mu- mu_K)/LA.norm(mu)
            return appx_error
        appx_errors = np.array([get_appx_error(K) for K in range(n+1)])
        K_opt = np.argmax(appx_errors < delta)

        log.info(f"Optimal K is {K_opt}")
        if K_opt == 0:
            log.info(f"delta too big")
        
        cut = np.zeros(n)
        cut[:K_opt] = 1.
        
        a = cut * (u.T@mu)
        log.info(f" return compent {a}")
        B = u@np.diag(1/(s+1e-8)*cut)@vh
        pos =  B@mu
        pos /= np.sum(pos)
        return np.nan_to_num(pos, 0)
        
class SpectralSelection(Optimizer):
    """
    This optimizer uses SpectralSelection
    """

    def __init__(self,delta = 0.1, c = 1, **kw):
        log.info(f"SpectralCut initialized with  delta {delta}, c {c}")
        self.delta = delta
        self.c = c
        
    def optimize(self, signal=None, sample_var: np.ndarray = None, **kw):
        """ Builds SpectralSelection 
        """
        assert sum(np.isnan(signal)) == 0
        n = len(sample_var)
        u,s, vh = LA.svd(sample_var)
        mu = signal.values
        delta = self.delta
        c = self.c

        a_ls = u.T@mu

        def get_appx_error(gamma):
            
            a_sel = np.sign(a_ls)*np.maximum(np.abs(a_ls) - gamma *np.power(s,-c),0)
            mu_sel = u@a_sel
            appx_error = LA.norm(mu- mu_sel)/LA.norm(mu)
            return abs(delta - appx_error)
        sol = scipy.optimize.minimize(get_appx_error, x0 = [0])
        
        gamma_opt = sol.x[0]
        log.info(f"Optimal gamma is {round(gamma_opt,3)}")
        a_sel = np.sign(a_ls)*np.maximum(np.abs(a_ls) - gamma_opt *np.power(s,-c),0)
        log.info(f" return compent {a_sel}")
        pos = u@np.diag(1/(s+1e-8))@a_sel
        pos /= np.sum(pos)
        return np.nan_to_num(pos, 0)
