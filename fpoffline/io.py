"""Utilities for reading focal plane engineering files
"""
import pathlib
import datetime

import numpy as np

import astropy.time
import astropy.table

import pandas as pd

import fpoffline.scripts.endofnight


FP_ENG = pathlib.Path('/global/cfs/cdirs/desi/engineering/focalplane')
DATA = pathlib.Path('/global/cfs/cdirs/desi/spectro/data')


def get_snapshot(timestamp=None, maxage_days=5, path=FP_ENG / 'calibration'):
    DIR = pathlib.Path(path)
    oneday = astropy.time.TimeDelta(1, format='jd')
    timestamp = timestamp or astropy.time.Time.now()
    for age in range(0, maxage_days+1):
        date = timestamp - age * oneday
        for snaps in sorted(DIR.glob(date.strftime('%Y%m%dT*.ecsv')), reverse=True):
            if not snaps.name.endswith('_fp_calibs.ecsv'):
                raise ValueError(f'Invalid snapshot name: {snaps.name}')
            when = astropy.time.Time.strptime(snaps.name[:-15], '%Y%m%dT%H%M%S%z')
            if when >= timestamp:
                continue
            try:
                table = astropy.table.Table.read(snaps)
                table.meta['name'] = snaps.name
                return table, when
            except ValueError as e:
                #import astropy
                if astropy.__version__ == '5.0':
                    print('You need astropy >= 5.0.4.  If you are getting 5.0 from the desiconda module "unset PYTHONPATH" may help.')
                raise e


def get_index(before=None, path=FP_ENG / 'PositionerIndexTable/index_files'):
    """Locate and read the latest positioner index table.

    The returned table is augmented by FIBER_ID calculated as

      FIBER_ID = 500*PETAL_LOC + 25*SLITBLOCK_ID + BLOCKFIBER_ID

    when SLITBLOCK_ID and BLOCKFIBER_ID are both valid, or else -1.

    Parameters
    ----------
    before : int or None
        When not None, restrict the search for the most recent index table
        to dates <= before, specified as YYYYMMDD.

    Returns
    -------
    astropy.table.Table
    """
    DIR = pathlib.Path(path)
    files = sorted(DIR.glob('desi_positioner_indexes_????????.csv'), reverse=True)
    dates = [int(file.name[-12:-4]) for file in files]
    if before is not None:
        dates = [date for date in dates if date <= before]
    if len(dates) == 0:
        raise ValueError(f'No index table found with before={before}')
    file = DIR / f'desi_positioner_indexes_{dates[0]}.csv'
    table = astropy.table.Table.read(file)
    nrows = len(table)
    fiber_id = np.full(nrows, -1, dtype=int)
    for irow in range(nrows):
        row = table[irow]
        try:
            slit, fiber = row['SLITBLOCK_ID'], row['BLOCKFIBER_ID']
            if slit.startswith('B') and fiber.startswith('F'):
                fiber_id[irow] = 500*int(row['PETAL_LOC']) + 25*int(slit[1:]) + int(fiber[1:])
        except AttributeError:
            pass
    table['FIBER_ID'] = fiber_id
    table['LOCATION'] = table['PETAL_LOC'] * 1000 + table['DEVICE_LOC']
    table.sort('LOCATION')
    table.meta['index_name'] = file.name
    return table


def load_endofnight(night, assets='fp-{night}.ecsv,moves-{night}.csv.gz', parent_dir=FP_ENG / 'endofnight', path_only=False):
    """Load end-of-night assets for a given night.

    The possible assets are:
    - fp-{night}.ecsv: focal plane status from the offline dump used for fiber assignment during this night
    - moves-{night}.csv.gz: moves data for the night (will be uncompressed using fpoffline.scripts.endofnight.uncompress_moves)
    - calib-{night}.csv: any changes to focal plane calibration during this night (usually just changes to keepouts by FP setup)
    - fvc-back-{night}.jpg: back-illuminated FVC image taken at the end of the night during the park robots script
    - fvc-front-{night}.jpg: front-illuminated FVC image taken at the end of the night during the park robots script
    - hwtables-{night}.csv.gz: dump of the hardware move tables used during the night

    Parameters
    ----------
    night : str or int
        Night to load, specified as YYYYMMDD.
    assets : str
        Comma-separated list of assets to load, selected from those listed above.
        The night is substituted into the asset names for each night. A night is considered
        to be available if all of the specified assets exist as files.
    parent_dir : str or pathlib.Path
        Directory containing subdirectories for each night with names YYYYMMDD.
        The default is appropriate for NERSC.
    path_only : bool
        If True, return the path to the directory containing the assets
        instead of loading them.

    Returns
    -------
    A list of loaded assets, or the path to the directory containing
    the assets if path_only is True.
    """
    night = str(night)
    assets = [ asset.format(night=night) for asset in assets.split(',') ]
    path = pathlib.Path(parent_dir) / night
    if path.exists():
        found = [ (path / asset).exists() for asset in assets ]
    elif (DATA / night).exists() and int(night) >= 20241201:
        # Look for results in DATA / night / EEEEEEEE /
        exptags = sorted((path.name for path in (DATA / night).glob("????????")))
        for exptag in exptags[::-1]:
            path = DATA / night / exptag
            found = [ (path / asset).exists() for asset in assets ]
            if any(found):
                break
    else:
        raise ValueError(f'No data found for {night}')
    if not all(found):
        missing = [ asset for (asset,ok) in zip(assets,found) if not ok ]
        raise ValueError(f'Missing some assets for {night}: {",".join(missing)}')
    if path_only:
        return path
    loaded = [ ]
    for asset in assets:
        if asset.endswith('.ecsv'):
            loaded.append(astropy.table.Table.read(path / asset))
        elif asset.endswith('.csv') or asset.endswith('.csv.gz'):
            loaded.append(pd.read_csv(path / asset, low_memory=False))
            if asset.startswith('moves'):
                fpoffline.scripts.endofnight.uncompress_moves(loaded[-1], night)
        else:
            raise ValueError(f'No loader implemented for {asset}')
    return loaded