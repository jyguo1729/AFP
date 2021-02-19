import logging as log
import numpy as np
import numpy.linalg as LA

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
        return np.nan_to_num(pos, 0)
        