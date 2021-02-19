import logging as log

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
        
        pos = 1 / len(signal)
        return np.nan_to_num(pos, 0)
        