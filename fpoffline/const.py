"""Load positioner constants.
"""
import os
import json
import pathlib


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
        return {E['name']: E['constants'] for E in data['elements']}
