import logging as log
import sys, os
log.basicConfig(level=log.INFO, format='backtest-%(asctime)s-%(funcName)s: %(message)s', datefmt='%Y-%b-%d %H:%M:%S')
import pandas as pd
from typing import List, Union
from uuid import uuid4


def read_csv(*,
             start=None,
             end=None,
             date_var="DATADATE",
             id_var="PERMNO",
             file_path=None,
             terms=None,
             labels=None,
             forward=0,
             ):
    """
    Parameters
    ----------
    start: str or pandas.Timestamp
    end: str or pandas.Timestamp, end date, inclusive
    date_var: name of the column of date
    id_var: identifier
    file_path: path of the csv
    terms: terms of the table to be kept
    labels: if not None, rename the terms
    forward: if == n, date will be shifted in such a way that each date will corresponds to the variable during the next n period

    """
    log.info(f"reading {file_path}")
    res = pd.read_csv(file_path)
    res["date"] = pd.to_datetime(res[date_var])
    res["id"] = res[id_var]
    res = res.drop([date_var, id_var], axis=1)

    if forward:
        log.info(f"forward period = {forward}")
        res["date"] = res.groupby("id")["date"].shift(forward)

    if start and end:
        log.info(f" start = {start}, end = {end} ")
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        res = res[(res["date"] >= start) & (res["date"] <= end)].reset_index(drop=True)

    if terms is None:
        terms = [var for var in res.columns if var not in ["date", "id"]]

    log.info(f"terms = {terms}")
    if labels is None: labels = terms
    res = res[["date", "id"] + terms].rename(columns=dict(zip(terms, labels)))

    return res


def write_csv(df: pd.DataFrame,
              file_path=None):
    """ Writes table to csv.
    """
    
    make_parent_dir(file_path)
    log.info(f"writing to: {file_path}")
    df.to_csv(file_path)


def make_parent_dir(file_path):

    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        log.info(f"creating dir {parent_dir}")
        os.makedirs(parent_dir, exist_ok=True)
