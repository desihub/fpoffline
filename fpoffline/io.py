"""Utilities for reading focal plane files.
"""
import pathlib

import astropy.time
import astropy.table


def get_snapshot(timestamp=None, maxage_days=5, path='/global/cfs/cdirs/desi/engineering/focalplane/calibration'):
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
            table = astropy.table.Table.read(snaps)
            table.meta['name'] = snaps.name
            return table, when
