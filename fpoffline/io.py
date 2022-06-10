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


def load_endofnight(night, verbose=True, parent_dir=FP_ENG / 'endofnight'):

    path = pathlib.Path(parent_dir)
    if not path.exists():
        raise ValueError('Non-existent parent_dir "{parent_dir}".')
    path = path / str(night)
    if not path.exists():
        raise ValueError('No directory found for night {night}.')

    summary = path / f'fp-{night}.ecsv'
    if not summary.exists():
        print(f'WARNING: missing end-of-night summary {summary}')
        summary = None
    else:
        summary = astropy.table.Table.read(summary)

    moves = path / f'moves-{night}.csv.gz'
    if not moves.exists():
        print(f'WARNING: missing moves table {moves}')
        moves = None
    else:
        moves = pd.read_csv(moves)
        fpoffline.scripts.endofnight.uncompress_moves(moves, night)

    if verbose:
        print(f'Found {np.count_nonzero(moves.blocked)} moves blocked for bad comms or collision avoidance.')
        print(f'Found {np.count_nonzero(summary["FUNC"])} non-functional devices.')
        inspect = summary['INSPECT']
        groups = summary.meta['inspect_groups']
        for bit, description in enumerate(groups):
            nbit = np.count_nonzero(summary['INSPECT'] & (1 << bit) != 0)
            print(f'Found {nbit} {description}.')

    return summary, moves