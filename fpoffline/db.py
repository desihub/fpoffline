"""Tools to query the online positioner databases.
"""
import pathlib
import logging

import numpy as np
import pandas as pd

import desimeter.transform.pos2ptl
import desimeter.transform.ptl2fp

import fpoffline.const

try:
    import requests
except ImportError:
    # We will flag this later if it matters.
    pass


class DB(object):
    """Initialize a connection to the database.

    To force a direct connection using sqlalchemy and pyscopg2, set ``http_fallback``
    to ``False``. To force an indirect http connection using requests,
    set ``config_name`` to ``None``.  By default, will attempt a
    direct connection then fall back to an indirect connection.

    Direct connection parameters are stored in the SiteLite package.

    An indirect connection reads authentication credentials from
    your ~/.netrc file. Refer to this internal trac page for details:
    https://desi.lbl.gov/trac/wiki/Computing/AccessNerscData#ProgrammaticAccess

    Parameters
    ----------
    config_path : str
        Path of yaml file containing direct connection parameters to use.
    http_fallback : bool
        Use an indirect http connection when a direct connection fails
        if True.
    """
    def __init__(self, config_name='/global/cfs/cdirs/desi/engineering/focalplane/db.yaml', http_fallback=True):
        self.method = 'indirect'
        if pathlib.Path(config_name).exists():
            # Try a direct connection.
            try:
                import yaml
            except ImportError:
                raise RuntimeError('The pyyaml package is not installed.')
            with open(config_name, 'r') as f:
                db_config = yaml.safe_load(f)
            try:
                import sqlalchemy
                self.engine = sqlalchemy.create_engine(
                    'postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(**db_config))
                self.method = 'direct'
            except ImportError:
                if not http_fallback:
                    raise RuntimeError('The sqlalchemy package is not installed.')
            except Exception as e:
                if not http_fallback:
                    raise RuntimeError(f'Unable to establish a database connection:\n{e}')
        if self.method == 'indirect' and http_fallback:
            try:
                import requests
            except ImportError:
                raise RuntimeError('The requests package is not installed.')
        logging.info(f'Established {self.method} database connection.')

    def query(self, sql, maxrows=10, dates=None):
        """Perform a query using arbitrary SQL. Returns a pandas dataframe.
        Use maxrows=None to remove any limit on the number of returned rows.
        """
        logging.debug(f'SQL: {sql}')
        if 'limit ' in sql.lower():
            raise ValueError('Must specify SQL LIMIT using maxrows.')
        if maxrows is None:
            maxrows = 'NULL'
        if self.method == 'direct':
            return pd.read_sql(sql + f' LIMIT {maxrows}', self.engine, parse_dates=dates)
        else:
            return self.indirect(dict(sql_statement=sql, maxrows=maxrows), dates)

    def indirect(self, params, dates=None):
        """Perform an indirect query using an HTTP request. Returns a pandas dataframe."""
        url = 'https://replicator.desi.lbl.gov/QE/DESI/app/query'
        params['dbname'] = 'desi'
        # Use tab-separated output since the web interface does not escape embedded
        # special characters, and there are instances of commas in useful
        # string columns like PROGRAM.
        #params['output_type'] = 'text,' # comma separated
        params['output_type'] = 'text' # tab separated
        logging.debug(f'INDIRECT PARAMS: {params}')
        req = requests.get(url, params=params)
        if req.status_code != requests.codes.ok:
            if req.status_code == 401:
                raise RuntimeError('Authentication failed: have you setup your .netrc file?')
            req.raise_for_status()
        # The server response ends each line with "\t\r\n" so we replace that with "\n" here.
        text = req.text.replace('\t\r\n', '\n')
        return pd.read_csv(io.StringIO(text), sep='\t', parse_dates=dates)


def get_calib(DB, at=None, verbose=True):
    """Get the most recent calibration data available, at the specified time or now, for each positioner on all petals.

    Parameters
    ----------
    DB : fpoffline.db.DB
        Database connection to use.
    at : datetime-like
        A value that pd.Timestamp can interpet to specify when the calibration data should be valid.
    verbose : bool
        Print a one-line summary for each petal when True.

    Returns
    -------
    pd.Dataframe
        A pandas dataframe containing the most recent available data for each positioner, augumented by
        petal_loc, offset_x,y_cs5 and location=1000*petal_loc+device_loc.
    """
    tables = []
    for petal_loc, petal_id in enumerate(fpoffline.const.PETAL_ID_MAP):
        table_name = f'posmovedb.positioner_calibration_p{petal_id}'
        before = '' if at is None else f" where time_recorded<=TIMESTAMP '{pd.Timestamp(at)}'"
        sql = f'''
            select * from {table_name}
            where (pos_id,time_recorded) in
            (
                select pos_id,max(time_recorded)
                from {table_name}{before}
                group by pos_id
            )'''
        table = DB.query(sql, 600)
        table['petal_loc'] = petal_loc
        table['location'] = 1000*petal_loc + table['device_loc']
        # Convert offset_xy from flatXY to CS5 using desimeter.
        ptl_x, ptl_y = desimeter.transform.pos2ptl.flat2ptl(table.offset_x, table.offset_y)
        table['offset_x_cs5'], table['offset_y_cs5'], _ = desimeter.transform.ptl2fp.ptl2fp(petal_loc, ptl_x, ptl_y)
        tables.append(table)
        if verbose:
            print(f'Found calibration data for {len(table)} positioners on petal loc[{petal_loc}] id[{petal_id}]')
    return pd.concat(tables, axis='index', ignore_index=True)


def get_moves(DB, at=None, expid=None, maxrows=10000, verbose=True):
    """Get the most recent moves recorded for each positioner at a specified time, or all moves for one exposure.

    Parameters
    ----------
    DB : fpoffline.db.DB
        Database connection to use.
    at : datetime-like
        A value that pd.Timestamp can interpet to specify a time to fetch moves for.  Only the most recent
        move for this time will be returned.  Either ``at`` or ``expid`` must be specified.
    expid : int
        An exposure id to fetch moves for. All moves (usually a blind + correction move) are returned.
        Either ``at`` or ``expid`` must be specified.
    maxrows : int
        Maximum number of rows to return for each petal. This function will print a warning if this needs
        to be increased, which might happen if there are many move steps for a single exposure.
    verbose : bool
        Print a one-line summary for each petal when True.

    Returns
    -------
    pd.Dataframe
        A pandas dataframe containing the most recent available data for each positioner, augumented by
        petal_loc, rot_T,P and location=1000*petal_loc+device_loc.
    """
    tables = []
    if expid and at:
        print('do not specify expid and at params')
        return
    before = '' if at is None else f" where time_recorded<=TIMESTAMP '{pd.Timestamp(at)}'"
    for petal_loc, petal_id in enumerate(fpoffline.const.PETAL_ID_MAP):
        table_name = f'posmovedb.positioner_moves_p{petal_id}'
        if expid is not None:
            sql = f'''
                select * from {table_name}
                where exposure_id={expid}
                order by time_recorded asc'''
        else:
            sql = f'''
                select * from {table_name}
                where (pos_id,time_recorded) in
                (
                    select pos_id,max(time_recorded)
                    from {table_name}{before}
                    group by pos_id
                )'''
        table = DB.query(sql, maxrows)
        if len(table)==maxrows:
            print(f'Need to increase maxrows={maxrows} for petal_id {petal_id}')
            return None
        table['petal_loc'] = petal_loc
        table['location'] = 1000*petal_loc + table['device_loc']
        # Calculate net T,P rotation by parseing move_val1,2 fields.
        ##table['rot_T'] = [0 if v=='' else np.sum([float(t.split(' ')[1]) for t in v.split('; ')]) for v in table.move_val1]
        ##table['rot_P'] = [0 if v=='' else np.sum([float(t.split(' ')[1]) for t in v.split('; ')]) for v in table.move_val2]
        tables.append(table)
        if verbose:
            print(f'Found {len(table)} rows of move data for petal loc[{petal_loc}] id[{petal_id}]')
    return pd.concat(tables, axis='index', ignore_index=True)
