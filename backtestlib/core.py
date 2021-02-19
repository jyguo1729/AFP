from . import optimizer as opt
from . import io
from .io import log
from typing import Union, List, Optional, Dict
import pandas as pd
from tqdm import tqdm

                 
class BackTest:
    
    def __init__(self,
                 *,
                 start: str=None, # start month , like "2018-01"
                 end: str=None, # start month , like "2018-01"
                 universe_file: str=None, # file path , like "/a/b/c/univ.csv"
                 optimizers: Dict[str, opt.Optimizer]  # optimizers
                 ):
        """
        Parameters
        ----------
        start
        end
        universe_file: csv that stores the universe
        optimizers: optimizers to use
        """
        
        self.start  = pd.to_datetime(start)
        self.end    = pd.to_datetime(end)
        self.data   = io.read_csv(start=self.start, end=self.end, file_path=universe_file)
        self.optimizers = optimizers
        self.strategies = list(optimizers.keys())
        log.info(f"back tester initialized; start = {self.start}, end = {self.end}, strategies = {self.strategies}")
        
    def read_signal(self,
                    file_path: str = None,
                    terms: Union[List[str], str]="all",
                    labels: Union[List[str], str]=None,
                    shift=0,
                    ):

        sig = io.read_csv(start=self.start,
                          end=self.end,
                          file_path=file_path,
                          terms=terms,
                          labels=labels,
                          )
        self.data = pd.merge(self.data, sig, how="left", on=["date", "id"])

    def construct_portfolio(self, signal_var="signal"):

        for date in tqdm(self.data["date"].unique()):
            for opt_name, opter in self.optimizers.items():
                self.data.loc[self.data["date"] == date, f"pos_{opt_name}"] = opter.optimize(signal=self.data.loc[self.data["date"] == date, signal_var])
    
    def serialize(self, 
                  out_file: str=None,
                  output_vars: Union[List[str], str]="all"):
        
        if output_vars == "all":
            output_vars = self.data.columns
        io.write_csv(self.data[output_vars], file_path=out_file)