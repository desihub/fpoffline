"""Load and interpret positioner hardware tables.
"""
import os
import pathlib

import numpy as np

import pandas as pd


def load_hwtable(expid, petal_id=None, exp_iter=None, path=None, verbose=False):
    """Load the positioner hardware move table(s) for the specified exposure.

    Parameters
    ----------
    expid : int
        Exposure ID to load
    petal_id : int or None
        Load the specified petal ID or load all petals when None
    exp_iter : int or None
        Load the specified exposure iteration or load all available when None.
        For science exposures, the initial blind move has a value of 0
        followed by the correction move with a value of 1.
    path : str or Path or None
        Load from the specified path or try NERSC and KPNO defaults when None
        and either the NERSC_HOST or DOS_HOME env var is set.
    verbose : bool
        Print information about progress

    Returns
    -------
    df : pandas dataframe
        Dataframe created with the pandas ``read_csv`` function, with an
        additional petal_id column.
    """
    # Get the path to use.
    if path is None:
        if os.getenv('NERSC_HOST') is not None:
            path = '/global/cfs/cdirs/desi/engineering/focalplane/hwtables'
        elif os.getenv('DOS_HOST') is not None:
            if expid >= 91163:
                path = '/global/cfs/cdirs/desi/engineering/focalplane/hwtables'
            else:
                path = '/data/msdos/focalplane/fp_temp_files/'
    if path is None:
        raise ValueError('Must specify a path except at NERSC or KPNO.')
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError('Invalid path: "{path}".')
    if verbose:
        print(f'Using path: "{path}".')
    # Look for any files associated with this exposure.
    files = sorted(path.glob(f'hwtables_ptlid??_{expid}_*.csv'))
    if len(files) == 0:
        raise ValueError(f'No files found for expid {expid}.')
    # Decode petal_id and exp_iter from each filename.
    petal_ids, exp_iters = [], []
    expstr = str(expid)
    for file in files:
        name = file.name
        petal_ids.append(int(name[14:16]))
        lo = name.index(expstr) + len(expstr) + 1
        hi = name.index('_', lo)
        exp_iters.append(int(name[lo:hi]) if hi > lo else 0)
    if verbose:
        print(f'Found petal_ids {set(petal_ids)} and exp_iters {set(exp_iters)}.')
    # Are the request petal_id and exp_iter available?
    if petal_id is not None and exp_iter is not None:
        if (petal_id, exp_iter) not in zip(petal_ids, exp_iters):
            raise ValueError(f'Requested petal_id={petal_id} and exp_iter={exp_iter} not found for expid {expid}.')
    elif petal_id is not None:
        if petal_id not in petal_ids:
            raise ValueError(f'Requested petal_id={petal_id} not found for expid {expid}.')
    elif exp_iter is not None:
        if exp_iter not in exp_iters:
            raise ValueError(f'Requested exp_iter={exp_iter} not found for expid {expid}.')
    # Read CSV files.
    dfs = []
    for i,file in enumerate(files):
        if petal_id not in (petal_ids[i], None):
            continue
        if exp_iter not in (exp_iters[i], None):
            continue
        if verbose:
            print(f'Reading {file}')
        df = pd.read_csv(file)
        # Add petal_id, exp_iter columns.
        df['petal_id'] = petal_ids[i]
        df['exp_iter'] = exp_iters[i]
        dfs.append(df)
    # Combine all CSVs.
    return pd.concat(dfs, ignore_index=True)
