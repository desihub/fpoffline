"""Load positioner constants.
"""
import os
import json
import pathlib
import collections
import importlib.resources

import numpy as np
import pandas as pd


# Array of petal_id values indexed by petal_loc=0-9.
PETAL_ID_MAP = [4, 5, 6, 3, 8, 10, 11, 2, 7, 9]


# List of the 18 devices removed during the summer 2021 shutdown
REMOVED_2021 = ['M04182', 'M02725', 'M03709', 'M01722', 'M03996', 'M06848',
                'M07550', 'M06931', 'M06891', 'M05743', 'M03567', 'M03918',
                'M03824', 'M03236', 'M03912', 'M03556', 'M04024', 'M03648']


def load_constants(version=None):
    """Load fiber positioner constants.

    This currently only works at NERSC by reading
    /global/cfs/cdirs/desi/engineering/focalplane/constants but might query the
    database in future.

    Parameters
    ----------
    version : int or None
        Load the specified version, or else the most recent available when None.

    Returns
    -------
    dict
        Dictionary of constants index by positioner ID.  Each value is a
        dictionary indexed by upper-case property names.
    """
    if os.getenv('NERSC_HOST') is None:
        raise RuntimeError('Constants are currently only accessible at NERSC.')
    path = pathlib.Path('/global/cfs/cdirs/desi/engineering/focalplane/constants')
    if not path.exists():
        raise RuntimeError('Non-existent directory: {path}.')
    if version is not None:
        file = path / f'constants-{version}.json'
    else:
        file = sorted(path.glob('constants-*.json'))[-1]
    with open(file) as f:
        data = json.load(f)
    return {elem['name']: elem['constants'] for elem in data['elements']}


PETAL_DESIGN = None


def get_petal_design():
    global PETAL_DESIGN
    if PETAL_DESIGN is None:
        with importlib.resources.path('fpoffline.data', 'DESI-0530-v18-coords.csv') as path:
            PETAL_DATA = pd.read_csv(path)
        HOLES = PETAL_DATA[np.isin(PETAL_DATA.DEVICE_TYPE, ('POS','ETC','FIF','GIF'))]
        X, Y = HOLES.X_PTL.to_numpy(), HOLES.Y_PTL.to_numpy()
        LOCS = HOLES.DEVICE_LOC.to_numpy()
        LOCMAP = np.full(LOCS.max() + 1, -1, int)
        LOCMAP[LOCS] = np.arange(len(LOCS))
        XFP, YFP = np.zeros((2, 10, len(LOCS)), np.float32)
        for petal_loc in range(10):
            alpha = (petal_loc - 3) * np.pi / 5
            C, S = np.cos(alpha), np.sin(alpha)
            XFP[petal_loc] = C * X - S * Y
            YFP[petal_loc] = S * X + C * Y
        PETAL_DESIGN = collections.namedtuple('PETAL_DESIGN',
            ['holes','locmap','xfp','yfp'])(holes=HOLES, locmap=LOCMAP, xfp=XFP, yfp=YFP)
    return PETAL_DESIGN
