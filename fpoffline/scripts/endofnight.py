import argparse
import logging
import pdb
import traceback
import sys
import os
import pathlib
import datetime
import math
from ast import literal_eval as safe_eval

import numpy as np

import pandas as pd

import fitsio

import astropy.time

#import desimeter
import desimeter.io
import desimeter.processfvc
import desimeter.transform.ptl2fp
import desimeter.transform.pos2ptl
import desimeter.transform.xy2qs

import fpoffline.io
import fpoffline.db
import fpoffline.fvc
import fpoffline.const


def run(args):

    # raise an exception here to flag any error
    # return value is propagated to the shell

    if args.night is None:
        # Use yesterday's date by default.
        args.night = int((datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y%m%d'))
        logging.info(f'Using default night {args.night}')

    # Check we have a valid parent path.
    if not args.parent_dir.exists():
        raise RuntimeError(f'Non-existent parent_dir: {args.parent_dir}')

    # Create a night subdirectory if necessary.
    output = args.parent_dir / str(args.night)
    if not output.exists():
        logging.info(f'Creating {output}')
        output.mkdir()
    logging.info(f'Output for {args.night} will be stored in {output}')

    # Calculate local midnight for this observing midnight.
    N = str(args.night)
    year, month, day = int(N[0:4]), int(N[4:6]), int(N[6:8])
    midnight = astropy.time.Time(datetime.datetime(year, month, day, 12) + datetime.timedelta(hours=19))
    logging.info(f'Local midnight on {N} is at {midnight}')

    # Calculate local noons before and after local midnight.
    twelve_hours = astropy.time.TimeDelta(12 * 3600, format='sec')
    noon_before = midnight - twelve_hours
    noon_after = midnight + twelve_hours

    # Load the most recent database snapshot.
    snapshot, snap_time = fpoffline.io.get_snapshot(astropy.time.Time(midnight, format='datetime'))
    snapshot['LOCATION'] = snapshot['PETAL_LOC']*1000 + snapshot['DEVICE_LOC']
    snapshot.sort('LOCATION')
    logging.info(f'Loaded snapshot {snapshot.meta["name"]}')
    snap_age = (midnight - snap_time).sec / 86400
    if snap_age > 0.6:
        logging.warning(f'Snapshot is {snap_age:.1f} days old.')

    # Initialize per-device summary table with metrology.
    metrology = desimeter.io.load_metrology() # returns an astropy table, not a pandas dataframe.
    summary = (metrology[np.isin(metrology['DEVICE_TYPE'],['POS','ETC','FIF','GIF'])][['LOCATION','X_FP','Y_FP']]
               .group_by('LOCATION').groups.aggregate(np.mean)) # use mean of fiducial locations
    summary.meta = {} # Do not use an OrderedDict
    summary.meta['night'] = args.night
    summary.meta['midnight'] = str(midnight)
    summary.meta['snapshot'] = snapshot.meta['name']
    summary.meta['snap_time'] = str(snap_time)

    # Add indexing info. The left join will drop 5 fidicuals that are missing metrology:
    # P074, P030, P029, P123, P058.
    I = fpoffline.io.get_index(args.night)
    logging.info(f'Using index {I.meta["index_name"]}')
    I['LOCATION'] = 1000 * I['PETAL_LOC'] + I['DEVICE_LOC']
    summary = astropy.table.join(summary, I, keys='LOCATION', join_type='left')
    summary.sort('LOCATION')

    # Add calibration info from the snapshot.
    summary = reduce_snapshot(snapshot, summary)

    # Compute interpolated constants to transform from flatXY to ptlXY.
    s0 = np.hypot(summary['OFFSET_X'], summary['OFFSET_Y'])
    r0 = desimeter.transform.xy2qs.s2r(s0)
    summary['R0_OVER_S0'] = r0 / s0

    # Initialize online database access.
    DB = fpoffline.db.DB()

    # Look up the exposures taken on this night.
    # Not all exposures have night set (e.g. the FP setup) so we
    # search on a 24-hour window of update_time.
    expvars = 'id,update_time,tileid,etcteff,etcreal,flavor,program'
    cond = f"""
    (update_time > timestamp '{noon_before.strftime("%Y-%m-%dT%H:%M:%S+0000")}') and
    (update_time < timestamp '{noon_after.strftime("%Y-%m-%dT%H:%M:%S+0000")}')
    """
    sql = f'select {expvars} from exposure.exposure where {cond} order by update_time asc'
    exps = DB.query(sql, maxrows=1000)

    # Find the (first) FP_setup exposure.
    setups = exps.query("program=='FP_setup'")
    if len(setups) != 2:
        logging.warning(f'Expected 2 setups but found {len(setups)}:')
        if len(setups) > 0:
            logging.warning(setups[['id','update_time']])
    if args.setup_id is None:
        if len(setups) == 0:
            logging.error('Giving up with no setups.')
            return -1
        args.setup_id = setups.id.min()
    summary.meta['setup_id'] = args.setup_id
    setup_exp = exps.query(f"id=={args.setup_id}")
    summary.meta['setup_time'] = str(exps.query(f"id=={args.setup_id}").iloc[0].update_time)
    if not args.setup_id or len(setup_exp) != 1:
        logging.error('Missing FP_setup exposure')
        return -1
    summary.meta['setup_time'] = str(setup_exp.iloc[0].update_time)
    logging.info(f'FP_setup is expid {args.setup_id}')

    # Find the (last) end park exposures.
    fronts = exps[exps.program.str.endswith("(front illuminated image)").fillna(False)]
    if len(fronts) != 1:
        logging.warning(f'Expected 1 front-illuminated image but got {len(fronts)}')
        if len(fronts) > 0:
            logging.warning(fronts[['id','update_time']])
    if args.front_id is None:
        args.front_id = fronts.id.min()
    if np.isfinite(args.front_id):
        logging.info(f'Front-illuminated image is expid {args.front_id}')
        summary.meta['front_id'] = args.front_id
        summary.meta['front_time'] = str(exps.query(f"id=={args.front_id}").iloc[0].update_time)
    else:
        logging.warning('Missing end-of-night front-illuminated exposure')
        summary.meta['front_id'] = None
        args.front_id = None

    backs = exps[exps.program.str.endswith("(back illuminated image)").fillna(False)]
    if len(backs) != 1:
        logging.warning(f'Expected 1 back-illuminated image but got {len(backs)}')
        if len(backs) > 0:
            logging.warning(backs[['id','update_time']])
    if args.back_id is None:
        args.back_id = backs.id.min()
    if np.isfinite(args.back_id):
        logging.info(f'Back-illuminated image is expid {args.back_id}')
        summary.meta['back_id'] = args.back_id
        summary.meta['back_time'] = str(exps.query(f"id=={args.back_id}").iloc[0].update_time)
    else:
        logging.warning('Missing end-of-night back-illuminated exposure')
        summary.meta['back_id'] = None
        args.back_id = None

    if args.park_id is None:
        args.park_id = exps.query("program=='FP_setup'").id.max()
    if np.isfinite(args.park_id) and args.park_id > args.setup_id:
        logging.info(f'End-night park image is expid {args.park_id}')
        summary.meta['park_id'] = args.park_id
        summary.meta['park_time'] = str(exps.query(f"id=={args.park_id}").iloc[0].update_time)
    else:
        logging.warning('Missing end-of-night park exposure')
        summary.meta['park_id'] = None
        args.park_id = None

    # Locate raw data products.
    DATA = args.data_dir
    logging.info(f'Reading FVC images from {args.data_dir}')

    end_time = None
    if args.front_id:
        if args.back_id and (args.back_id - args.front_id != 2):
            logging.warning(f'Unexpected back_id - front_id = {args.back_id - args.front_id}')
        # Generate processed images.
        front_img =  output / f'fvc-front-{args.night}.jpg'
        ftag = str(args.front_id).zfill(8)
        front_fits = DATA / str(args.night) / ftag / f'fvc-{ftag}.fits.fz'
        if not front_fits.exists():
            logging.warning(f'Missing front-illuminated FVC image {front_fits}')
            summary.meta['front_id'] = None
        else:
            fhdr = fitsio.read_header(str(front_fits), ext=0)
            if end_time is None and 'DATE-OBS' in fhdr and fhdr['DATE-OBS']:
                end_time = fhdr['DATE-OBS'] + '+0000'
            if args.overwrite or not front_img.exists():
                logging.info(f'Generating {front_img} from expid {args.front_id}...')
                data = fitsio.read(str(front_fits), ext='F0000')
                data = fpoffline.fvc.process_front_illuminated(data)
                fpoffline.fvc.plot_fvc(data, color='cividis', save=front_img, quality=75)

    if args.back_id or args.park_id:
        # There are normally two final back-illuminated images.  Use the first in this case.
        if args.park_id and args.back_id and args.front_id and (args.park_id == args.front_id - 2):
            logging.info(f'Using the park back-illuminated expid {args.park_id} instead of {args.back_id}')
            args.back_id = args.park_id
        if args.park_id and not args.back_id:
            logging.info(f'Using the park expid {args.park_id} since no final back-illuminated image available.')
            args.back_id = args.park_id
        back_img =  output / f'fvc-back-{args.night}.jpg'
        btag = str(args.back_id).zfill(8)
        back_fits = DATA / str(args.night) / btag / f'fvc-{btag}.fits.fz'
        if not back_fits.exists():
            logging.warning(f'Missing back-illuminated FVC image {back_fits}')
            summary.meta['back_id'] = None
            back_fits = None
        else:
            bhdr = fitsio.read_header(str(back_fits), ext=0)
            if end_time is None and 'DATE-OBS' in bhdr and bhdr['DATE-OBS']:
                end_time = bhdr['DATE-OBS'] + '+0000'
            if end_time is None and 'MJD-OBS' in bhdr and bhdr['MJD-OBS']:
                # Park FVC image is missing DATE-OBS but has MJD-OBS.
                end_time = astropy.time.Time(bhdr['MJD-OBS'], format='mjd').iso + '+0000'
                logging.warning('Back image missing DATE-OBS so using MJD-OBS instead.')
            if args.overwrite or not back_img.exists():
                logging.info(f'Generating {back_img} from expid {args.back_id}...')
                try:
                    data = fitsio.read(str(back_fits), ext='F0000')
                    data = fpoffline.fvc.process_back_illuminated(data)
                    fpoffline.fvc.plot_fvc(data, color=(0,1,1), save=back_img, quality=85)
                except Exception as e:
                    logging.warning(f'Error reading back-illuminated FVC image {back_fits}')
                    logging.warning(e)
                    summary.meta['back_id'] = None
                    back_fits = None

        if back_fits:
            # Less verbose desimeter logging.
            desi_loglevel = os.getenv('DESI_LOGLEVEL', 'INFO')
            os.putenv('DESI_LOGLEVEL', 'ERROR')
            try:
                # Use desimeter to find the back-illuminated spots.
                spots = desimeter.processfvc.process_fvc(str(back_fits), use_subprocess=False)
                logging.info(f'Fit {len(spots)} spots')
                # Fit the FVC <-> FP transforms to the spots.
                tx = desimeter.transform.fvc2fp.FVC2FP.read_jsonfile(desimeter.io.fvc2fp_filename())
                tx.fit(spots, metrology, update_spots=False, zbfit=True)
                # Record per-location info of all fidicials and positioners.
                fp = np.stack((summary['X_FP'], summary['Y_FP']))
                fvc = np.stack(tx.fp2fvc(fp[0], fp[1]))
                # Save X_FVC,Y_FVC measured from top-left corner.
                fvc_img_size = 6000
                summary['X_FVC'], summary['Y_FVC'] = fvc_img_size - fvc
                # Calculate a local linear transformation from FP coords to FVC pixels.
                dfvc_dx = np.stack(tx.fp2fvc(fp[0] + 0.5, fp[1])) - np.stack(tx.fp2fvc(fp[0] - 0.5, fp[1]))
                dfvc_dy = np.stack(tx.fp2fvc(fp[0], fp[1] + 0.5)) - np.stack(tx.fp2fvc(fp[0], fp[1] - 0.5))
                summary['DXFVC_DXFP'], summary['DYFVC_DXFP'] = -dfvc_dx
                summary['DXFVC_DYFP'], summary['DYFVC_DYFP'] = -dfvc_dy
                # Transform GFA, PTL keepouts from FP to FVC.
                for petal_loc in range(10):
                    xfp, yfp = summary.meta['keepout']['gfa'][petal_loc]
                    xfvc, yfvc = tx.fp2fvc(np.array(xfp), np.array(yfp))
                    summary.meta['keepout']['gfa'][petal_loc] = [(6000-xfvc).tolist(), (6000-yfvc).tolist()]
                    xfp, yfp = summary.meta['keepout']['ptl'][petal_loc]
                    xfvc, yfvc = tx.fp2fvc(np.array(xfp), np.array(yfp))
                    summary.meta['keepout']['ptl'][petal_loc] = [(6000-xfvc).tolist(), (6000-yfvc).tolist()]
            except Exception as e:
                logging.warning(f'Failed to fit spots in expid {args.back_id}:\n{e}')
                if args.traceback:
                    raise e
            os.putenv('DESI_LOGLEVEL', desi_loglevel)

    # Determine the end time to use for DB queries if not already set.
    snap_time = snap_time.strftime("%Y-%m-%dT%H:%M:%S+0000")
    if end_time is None:
        end_time = noon_after.strftime("%Y-%m-%dT%H:%M:%S+0000")
        logging.warning(f'using default end_time {end_time} in absence of any park FVC images')
    # Save the end time to use for DB queries.
    summary.meta['end_time'] = str(end_time)

    # Look for any calib updates since the snapshot.
    calib_csv = output / f'calib-{args.night}.csv'
    if args.overwrite or not calib_csv.exists():
        logging.info(f'Fetching calib DB updates during {snap_time} - {end_time}...')
        tables = []
        for petal_loc, petal_id in enumerate(fpoffline.const.PETAL_ID_MAP):
            table_name = f'posmovedb.positioner_calibration_p{petal_id}'
            sql = f'''
                select * from {table_name} where
                    (time_recorded > timestamp '{snap_time}') and
                    (time_recorded < timestamp '{end_time}')
                order by time_recorded asc
            '''
            table = DB.query(sql, maxrows=1000)
            if len(table) > 0:
                logging.info(f'Read {len(table)} rows for PETAL_LOC {petal_loc}')
            table['petal_loc'] = petal_loc
            table['location'] = 1000*petal_loc + table['device_loc']
            tables.append(table)
        calib = pd.concat(tables, axis='index', ignore_index=True)
        calib.to_csv(calib_csv, index=False)
        logging.info(f'Wrote {calib_csv.name} with {len(calib)} rows.')
    else:
        calib = pd.read_csv(calib_csv, parse_dates=['time_recorded'])
        logging.info(f'Read {calib_csv.name} with {len(calib)} rows.')

    # Look for move updates since the snapshot.
    moves_csv = output / f'moves-{args.night}.csv.gz'
    if args.overwrite or not moves_csv.exists():
        print(f'Fetching moves DB updates during {snap_time} - {end_time}...')
        tables = []
        for petal_loc, petal_id in enumerate(fpoffline.const.PETAL_ID_MAP):
            table_name = f'posmovedb.positioner_moves_p{petal_id}'
            move_cols = '''
                time_recorded,device_loc,pos_id,pos_t,pos_p,ctrl_enabled,move_cmd,move_val1,move_val2,log_note,
                exposure_id,exposure_iter,flags,ptl_x,ptl_y,obs_x,obs_y
                '''
            # We don't bother sorting by time_recorded in the query since we do it globally after concatenating all petals.
            sql = f'''
                select {move_cols} from {table_name} where
                    (time_recorded > timestamp '{snap_time}') and
                    (time_recorded < timestamp '{end_time}')
            '''
            table = DB.query(sql, maxrows=200000)
            if len(table) > 0:
                logging.info(f'Read {len(table)} rows for PETAL_LOC {petal_loc}')
            table['location'] = 1000*petal_loc + table['device_loc']
            table.drop(columns='device_loc', inplace=True)
            tables.append(table)
        moves = pd.concat(tables, axis='index', ignore_index=True)
        # Sort by increasing time.
        moves.sort_values('time_recorded', inplace=True, ignore_index=True)
        # Calculate and save the sum of commanded T,P moves.
        def sum_move(col_in, col_out):
            moves[col_out] = 0.
            valid = moves[col_in].notna()
            angle = moves[col_in].dropna().str.split('; | ').apply(lambda d: np.sum(np.array(list(map(float, d[1::2])))))
            moves.loc[valid, col_out] = angle
            moves.drop(columns=col_in, inplace=True) # Drop the original string column
        sum_move('move_val1', 'req_dt')
        sum_move('move_val2', 'req_dp')
        # Compress the log notes
        moves.log_note = (
            moves.log_note
            .str.replace("Positioner is disabled", "$D", regex=False)
            .str.replace("Target request denied", "$R", regex=False)
            .str.replace("handle_fvc_feedback", "$F", regex=False)
            .str.replace("Stored new: OBS_X OBS_Y PTL_X PTL_Y PTL_Z FLAGS", "$S", regex=False)
            .str.replace("tp_update_posTP", "$U", regex=False)
            .str.replace("end of night park_positioners observer script", "$E", regex=False)
            .str.replace("req_posintTP=", "TP=", regex=False)
            .str.replace("req_ptlXYZ=", "XYZ=", regex=False)
        )
        # Compress the move_cmd
        moves.move_cmd = (
            moves.move_cmd
            .str.replace("; (auto backlash backup); (auto final creep)", "$A", regex=False)
            .str.replace("obsdXdY=", "dXY=", regex=False)
        )
        # Transform PTL_X,Y corresponding to OBS_X,Y into nominal FP coords
        # for a direct probe of the petal alignments.
        valid = ~(moves.ptl_x.isna() | moves.ptl_y.isna() | moves.location.isna())
        moves.loc[valid, 'ptl_x'], moves.loc[valid, 'ptl_y'] = ptl2fp_nominal(
            moves.ptl_x[valid], moves.ptl_y[valid], moves.location[valid] // 1000)
        # Extract and save ptlXYZ from the log_note.
        sel = moves.log_note.str.contains('XYZ=') & ~moves.location.isna()
        msel = moves[sel]
        petal_locs = np.array(msel.location // 1000)
        def splitXYZ(note):
            i1 = note.index('XYZ=(')
            i2 = note.index(')', i1)
            return [float(val) for val in note[i1+5:i2].split(', ')]
        ptlXYZ = msel.log_note.apply(splitXYZ)
        x_ptl = ptlXYZ.apply(lambda d: d[0])
        y_ptl = ptlXYZ.apply(lambda d: d[1])
        # Transform the requested PTL coords into nominal FP coords.
        x_req, y_req = ptl2fp_nominal(x_ptl, y_ptl, petal_locs)
        moves['req_x'] = np.nan
        moves['req_y'] = np.nan
        moves.loc[sel, 'req_x'] = x_req
        moves.loc[sel, 'req_y'] = y_req
        # Calculate predicted petal and focal-plane (x,y) for each valid internal (t,p).
        valid = ~(moves.pos_t.isna() | moves.pos_p.isna() | moves.location.isna())
        locs = moves[valid].location
        idx = np.searchsorted(summary['LOCATION'], locs)
        assert np.all(summary['LOCATION'][idx] == locs)
        petal_locs = summary['PETAL_LOC'][idx]
        moves['pred_x'] = np.nan
        moves['pred_y'] = np.nan
        x_ptl_from_angles, y_ptl_from_angles = int2ptl(
            moves.pos_t[valid], moves.pos_p[valid],
            summary['OFFSET_T'][idx], summary['OFFSET_P'][idx],
            summary['LENGTH_R1'][idx], summary['LENGTH_R2'][idx],
            summary['OFFSET_X'][idx], summary['OFFSET_Y'][idx])
        x_pred, y_pred = ptl2fp_nominal(x_ptl_from_angles, y_ptl_from_angles, petal_locs)
        moves.loc[valid, 'pred_x'] = x_pred
        moves.loc[valid, 'pred_y'] = y_pred
        # Identify rows with FVC feedback.
        moves['fvc_feedback'] = moves.log_note.str.contains("\$F")
        # Flag rows that are followed immediately by an FVC feedback row.
        byloc = moves.groupby('location')
        moves['has_fvc_feedback'] = byloc.fvc_feedback.shift(-1, fill_value=False)
        # Store the feedback POS_T,P as FVC_T,P or NaN if there is no FVC feedback.
        moves['fvc_t'] = np.nan
        moves['fvc_p'] = np.nan
        moves.loc[moves['has_fvc_feedback'], 'fvc_t'] = byloc.pos_t.shift(-1, fill_value=np.nan)
        moves.loc[moves['has_fvc_feedback'], 'fvc_p'] = byloc.pos_p.shift(-1, fill_value=np.nan)
        # Calculate the actual change in angles using fvc_t,p
        byloc = moves.groupby('location')
        moves['last_fvc_t'] = byloc.fvc_t.shift(+1, fill_value=np.nan)
        moves['last_fvc_p'] = byloc.fvc_p.shift(+1, fill_value=np.nan)
        moves['act_dt'] = moves.fvc_t - moves.last_fvc_t
        moves['act_dp'] = moves.fvc_p - moves.last_fvc_p
        # Remove temporary columns and only keep fvc_t,p and act_dt,dp
        moves.drop(columns=['fvc_feedback', 'has_fvc_feedback', 'last_fvc_t', 'last_fvc_p'], inplace=True)

    # Save the summary table as ECSV (so the metadata is included)
    # TODO: round float values
    summary.meta = dict(summary.meta) # Don't use an OrderedDict
    summary.write(output / f'fp-{args.night}.ecsv', overwrite=True)

    return 0


def reduce_snapshot(snapshot, summary):
    """Utility function used to merge snapshot calibration and keepout info into the summary table and its metadata.
    """
    canonical = lambda k: np.round(safe_eval(k), 3)

    cols = (
        'LENGTH_R1', 'LENGTH_R2', 'OFFSET_T', 'OFFSET_P', 'OFFSET_X', 'OFFSET_Y', 'PHYSICAL_RANGE_T', 'PHYSICAL_RANGE_P',
        'DEVICE_CLASSIFIED_NONFUNCTIONAL', 'FIBER_INTACT', 'LOCATION')
    reduced = astropy.table.Table(snapshot[cols])
    reduced.meta = {}
    reduced.sort('LOCATION')

    keepouts_t, keepout_t_idx = get_keepouts(snapshot, 'T', canonical)
    keepouts_p, keepout_p_idx = get_keepouts(snapshot, 'P', canonical)
    keepouts = dict(theta=keepouts_t.tolist(), phi=keepouts_p.tolist())

    # Transform GFA and PTL keepouts to focal-plane coords for each petal.
    gfa_x, gfa_y = desimeter.transform.pos2ptl.flat2ptl(
        *canonical(snapshot.meta['keepout_GFA']))
    ptl_x, ptl_y = desimeter.transform.pos2ptl.flat2ptl(
        *canonical(snapshot.meta['keepout_PTL']))
    gfa, ptl = [], []
    for petal_loc in range(10):
        x, y, _ = desimeter.transform.ptl2fp.ptl2fp(petal_loc, gfa_x, gfa_y)
        gfa.append(np.array([x, y]).tolist())
        x, y, _ = desimeter.transform.ptl2fp.ptl2fp(petal_loc, ptl_x, ptl_y)
        ptl.append(np.array([x, y]).tolist())
    keepouts['gfa'] = gfa
    keepouts['ptl'] = ptl

    # Save the indices of the keepouts used by each device.
    reduced['KEEPOUT_T'] = keepout_t_idx
    reduced['KEEPOUT_P'] = keepout_p_idx

    # Merge with the summary. The snapshot has no entries for the fidicuals, so their new fields
    # will be empty after the left join.
    summary = astropy.table.join(summary, reduced, keys='LOCATION', join_type='left')

    summary.meta['keepout'] = keepouts

    petals = []
    for petal_loc in range(10):
        petal_id = fpoffline.const.PETAL_ID_MAP[petal_loc]
        alignment = snapshot.meta['PETAL_ALIGNMENTS'][petal_id]
        gamma = alignment['gamma']
        petals.append(dict(
            Tx=alignment['Tx'], Ty=alignment['Ty'],
            cosGamma=math.cos(gamma), sinGamma=math.sin(gamma)))
    summary.meta['petals'] = petals

    return summary


def get_keepouts(snap, arm, canonical):

    nsnap = len(snap)
    keepouts = [ canonical(snap.meta[f'general_keepout_{arm}']) ]
    keepout_idx = np.full(nsnap, -1, int)

    for j, row in enumerate(snap):
        k = canonical(row[f'KEEPOUT_{arm}'])
        for i, ki in enumerate(keepouts):
            if np.array_equal(k, ki):
                keepout_idx[j] = i
                break
        if keepout_idx[j] == -1:
            keepout_idx[j] = len(keepouts)
            keepouts.append(k)

    logging.info(f'Found {len(keepouts)} unique {arm} keepouts for {nsnap} devices')
    return np.array(keepouts), keepout_idx


#def ptl2fp(x_ptl, y_ptl, petal_locs):
#    """Transform from petal flat (x,y) to focal-plane (x,y) with desimeter translate and rotate.
#    """
#    x_fp, y_fp = np.zeros(shape=(2,)+x_ptl.shape)
#    for petal_loc in np.unique(petal_locs):
#        sel = (petal_locs == petal_loc)
#        x_fp[sel], y_fp[sel], _ = desimeter.transform.ptl2fp.ptl2fp(
#            petal_loc, x_ptl[sel], y_ptl[sel])
#    return x_fp, y_fp


def int2ptl(t_int, p_int, offset_t, offset_p, length_r1, length_r2, offset_x, offset_y):
    """Transform from internal angles (t,p) to ptl (x,y).
    """
    t_ext = desimeter.transform.pos2ptl.int2ext(t_int, offset_t)
    p_ext = desimeter.transform.pos2ptl.int2ext(p_int, offset_p)
    x_loc, y_loc = desimeter.transform.pos2ptl.ext2loc(t_ext, p_ext, length_r1, length_r2)
    x_flat = desimeter.transform.pos2ptl.loc2flat(x_loc, offset_x)
    y_flat = desimeter.transform.pos2ptl.loc2flat(y_loc, offset_y)
    return desimeter.transform.pos2ptl.flat2ptl(x_flat, y_flat)


def ptl2fp_nominal(x_ptl, y_ptl, petal_locs):
    """Transform from petal flat (x,y) to focal-plane (x,y) with nominal rotation.
    """
    phi = (np.arange(10) - 3) * np.pi / 5
    x_fp, y_fp = np.zeros(shape=(2,)+x_ptl.shape)
    for petal_loc in np.unique(petal_locs):
        sel = (petal_locs == petal_loc)
        C, S = np.cos(phi[petal_loc]), np.sin(phi[petal_loc])
        x_fp[sel] = x_ptl[sel] * C - y_ptl[sel] * S
        y_fp[sel] = x_ptl[sel] * S + y_ptl[sel] * C
    return x_fp, y_fp


def main():
    # https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser(
        description='Run the focal-plane end of night analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--night', type=str, metavar='YYYYMMDD',
        help='night to process or use the most recent night if not specified')
    parser.add_argument('--overwrite', action='store_true',
        help='overwrite any existing output files')
    parser.add_argument('--setup-id', type=int, metavar='NNNNNNNN',
        help='exposure ID that starts the observing night')
    parser.add_argument('--front-id', type=int, metavar='NNNNNNNN',
        help='exposure ID to use for the front-illuminated image')
    parser.add_argument('--back-id', type=int, metavar='NNNNNNNN',
        help='exposure ID to use for the back-illuminated image')
    parser.add_argument('--park-id', type=int, metavar='NNNNNNNN',
        help='exposure ID to use for the park robots script image')
    parser.add_argument('--parent-dir', type=pathlib.Path, metavar='PATH',
        default=pathlib.Path('/global/cfs/cdirs/desi/engineering/focalplane/endofnight'),
        help='parent directory for per-night output directories')
    parser.add_argument('--data-dir', type=pathlib.Path, metavar='PATH',
        default=pathlib.Path('/global/cfs/cdirs/desi/spectro/data'),
        help='directory containing raw data products under NIGHT/EXPID/')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='provide verbose output on progress')
    parser.add_argument('--debug', action='store_true',
        help='provide verbose and debugging output')
    parser.add_argument('--traceback', action='store_true',
        help='print traceback and enter debugger after an exception')
    args = parser.parse_args()

    # Configure logging.
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s %(message)s')

    try:
        retval = run(args)
        sys.exit(retval)
    except Exception as e:
        if args.traceback:
            # https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            print(e)
            sys.exit(-1)

if __name__ == '__main__':
    main()
