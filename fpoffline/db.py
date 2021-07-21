"""Tools to query the online positioner databases.
"""
import pandas as pd

import desimeter.transform.pos2ptl
import desimeter.transform.ptl2fp


# Array of petal_id values indexed by petal_loc=0-9.
petal_ids = [4, 5, 6, 3, 8, 10, 11, 2, 7, 9]


def get_calib(DB, at=None, verbose=True):
    """Get the most recent calibration data available, at the specified time or now, for each positioner on all petals.

    Parameters
    ----------
    DB : desietc.db.DB
        Database connection to use.
    at : datetime-like
        A value that pd.Timestamp can interpet to specify when the calibration data should be valid.
    verbose : bool
        Print a one-line summary for each petal when True.

    Returns
    -------
    pd.Dataframe
        A pandas dataframe containing the most recent available data for each positioner, augumented by petal_loc
        and offset_x,y_cs5.
    """
    tables = []
    for petal_loc, petal_id in enumerate(petal_ids):
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
        # Convert offset_xy from flatXY to CS5 using desimeter.
        ptl_x, ptl_y = desimeter.transform.pos2ptl.flat2ptl(table.offset_x, table.offset_y)
        table['offset_x_cs5'], table['offset_y_cs5'], _ = desimeter.transform.ptl2fp.ptl2fp(petal_loc, ptl_x, ptl_y)
        tables.append(table)
        if verbose:
            print(f'Found calibration data for {len(table)} positioners on petal loc[{petal_loc}] id[{petal_id}]')
    return pd.concat(tables, axis='index', ignore_index=True)
