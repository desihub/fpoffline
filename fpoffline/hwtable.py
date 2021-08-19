"""Load and interpret positioner hardware tables.
"""
import os
import pathlib
import json

import numpy as np

import pandas as pd

import fpoffline.const


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


class Scheduler:
    """Initialize a new move scheduler for a positioner.
    """
    CLOCK = 18000                 # tick frequency in Hz
    PAUSE = 18                    # clock ticks per unit of pause
    CREEP_STEP = 0.1              # motor rotation in degrees for a single creep step
    CRUISE_RATIO = 33             # ratio of cruise and creep motor rotations for a single step
    GEAR_RATIO = (46.0/14.0+1)**4 # Namiki "337:1" gear box (GEAR_TYPE_P,T in constants db)

    default_const = dict(
        CREEP_PERIOD=2, SPINUPDOWN_PERIOD=12, MOTOR_CCW_DIR_T=-1, MOTOR_CCW_DIR_P=-1,
        GEAR_TYPE_T='namiki', GEAR_TYPE_P='namiki')

    default_calib = dict(
        length_r1=3.0, length_r2=3.0, offset_t=0.0, offset_p=0.0,
        offset_x=0.0, offset_y=0.0, gear_calib_t=1.0, gear_calib_p=1.0)

    class Motor:
        """
        """
        def __init__(self, CREEP_PERIOD=2, SPINUPDOWN_PERIOD=12, ccw=-1, ratio=1):
            self.CREEP_PERIOD = CREEP_PERIOD
            self.SPINUPDOWN_PERIOD = SPINUPDOWN_PERIOD
            self.dw = ccw * ratio * Scheduler.CREEP_STEP * Scheduler.CLOCK / (self.CREEP_PERIOD * Scheduler.GEAR_RATIO)
            self.increment = []
            self.duration = []
            self.nticks = 0

        def creep(self, steps):
            if steps==0: return
            direction, nsteps = (1,steps) if steps > 0 else (-1,-steps)
            self.increment.append(direction)
            self.duration.append(self.CREEP_PERIOD * nsteps)
            self.nticks += self.duration[-1]

        def cruise(self, steps):
            if steps==0: return
            direction, nsteps = (1,steps) if steps > 0 else (-1,-steps)
            nupdown = Scheduler.CRUISE_RATIO
            spin_up_down = [(i + 1) * direction * self.CREEP_PERIOD for i in range(nupdown)]
            self.increment.extend(spin_up_down + spin_up_down[-1:] + spin_up_down[::-1])
            self.duration.extend([self.SPINUPDOWN_PERIOD] * nupdown + [nsteps] + [self.SPINUPDOWN_PERIOD] * nupdown)
            self.nticks += 2 * self.SPINUPDOWN_PERIOD * Scheduler.CRUISE_RATIO + nsteps

        def pause(self, steps, unit=None):
            unit = unit or Scheduler.PAUSE
            if steps < 0: raise ValueError('Invalid steps < 0.')
            if steps == 0: return
            self.increment.append(0)
            self.duration.append(steps * unit)
            self.nticks += self.duration[-1]

        def move(self, steps, mode, pause):
            if mode == 'creep': self.creep(steps)
            elif mode == 'cruise': self.cruise(steps)
            else: raise ValueError(f'Invalid move mode "{mode}".')
            self.pause(pause)
            assert len(self.increment) == len(self.duration)

        def finalize(self):
            omega = np.array(self.increment) * self.dw
            dt = np.array(self.duration) / Scheduler.CLOCK
            n = len(self.increment) + 1
            self.time = np.empty(n, np.float32)
            self.angle = np.empty(n, np.float32)
            self.time[0] = self.angle[0] = 0
            self.time[1:] = np.cumsum(dt)
            self.angle[1:] = np.cumsum(omega * dt)

    def __init__(self, const=default_const, calib=default_calib):
        # Convert pandas series to dictionary if necessary.
        try:
            calib = calib.iloc[0].to_dict()
        except AttributeError:
            pass
        if const['GEAR_TYPE_T'] != 'namiki' or const['GEAR_TYPE_P'] != 'namiki':
            raise ValueError('Only "namiki" gears are supported.')
        self.const = const
        self.calib = calib

    def plan(self, move, hwtable):
        """
        """
        # Convert pandas series to dictionary if necessary.
        try:
            move = move.iloc[0].to_dict()
        except AttributeError:
            pass
        try:
            hwtable = hwtable.iloc[0].to_dict()
        except AttributeError:
            pass
        # Convert args from strings to arrays of python types.
        args = [json.loads(hwtable[name].replace("'", '"')) for name in
                ('motor_steps_T', 'speed_mode_T', 'motor_steps_P', 'speed_mode_P', 'postpause')]
        # Check that arrays have matching lengths.
        try:
            lens = [len(arg) for arg in args]
            assert min(lens) == max(lens)
        except (TypeError, AssertionError):
            raise RuntimeError('Arguments must be arrays of the same length.')

        # Schedule the move table.
        self.motor_T = Scheduler.Motor(
            self.const['CREEP_PERIOD'], self.const['SPINUPDOWN_PERIOD'],
            self.const['MOTOR_CCW_DIR_T'], self.calib['gear_calib_t'])
        self.motor_P = Scheduler.Motor(
            self.const['CREEP_PERIOD'], self.const['SPINUPDOWN_PERIOD'],
            self.const['MOTOR_CCW_DIR_P'], self.calib['gear_calib_p'])
        for (steps_T, mode_T, steps_P, mode_P, pause) in zip(*args):
            self.motor_T.move(steps_T, mode_T, pause)
            self.motor_P.move(steps_P, mode_P, pause)
            # Add extra delay to one motor, if needed, to keep both in synch.
            delta = self.motor_T.nticks - self.motor_P.nticks
            if delta > 0:
                self.motor_P.pause(+delta, 1)
            elif delta < 0:
                self.motor_T.pause(-delta, 1)
        self.motor_T.finalize()
        self.motor_P.finalize()

        # Calculate theta,phi angles [in deg] with theta relative to CS5 +x and phi relative to theta,
        # such that the move ends up at T=move['pos_t'], P=move['pos_p'].
        petal_loc = fpoffline.const.PETAL_ID_MAP.index(move['petal_id'])
        petal_T = (petal_loc - 3) * 36
        T0 = petal_T + self.calib['offset_t'] + move['pos_t'] - self.motor_T.angle[-1]
        P0 = self.calib['offset_p'] + move['pos_p'] - self.motor_P.angle[-1]
        self.time = np.unique(np.concatenate((self.motor_T.time, self.motor_P.time)))
        self.theta = np.interp(self.time, self.motor_T.time, self.motor_T.angle + T0)
        self.phi = np.interp(self.time, self.motor_P.time, self.motor_P.angle + P0)

        # Calculate CS5 coords [in mm] of the theta, phi arm endpoints.
        tstep = self.const['CREEP_PERIOD'] / self.CLOCK
        nt = int(np.ceil(self.time[-1] / tstep)) + 1
        self.t = np.linspace(0, self.time[-1], nt)
        T = np.interp(self.t, self.time, np.deg2rad(self.theta))
        TP = np.interp(self.t, self.time, np.deg2rad(self.theta + self.phi))
        C, S = np.cos(np.deg2rad(petal_T)), np.sin(np.deg2rad(petal_T))
        x0, y0 = self.calib['offset_x'], self.calib['offset_y']
        self.x0, self.y0 = x0 * C - y0 * S, x0 * S + y0 * C
        self.rT, self.rP = self.calib['length_r1'], self.calib['length_r2']
        self.xT = self.x0 + self.rT * np.cos(T)
        self.yT = self.y0 + self.rT * np.sin(T)
        self.xP = self.xT + self.rP * np.cos(TP)
        self.yP = self.yT + self.rP * np.sin(TP)
