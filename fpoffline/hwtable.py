"""Load and interpret positioner hardware tables.
"""

import os
import pathlib
import json

import numpy as np

import pandas as pd

import desimeter.transform.pos2ptl
import desimeter.transform.ptl2fp


def load_hwtable(
    expid, petal_id=None, exp_iter=None, path=None, verbose=False, compress=True
):
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
    compress : bool
        Remove redundant columns and compress string columns by removing spaces
        and replacing 'creep' and 'cruise' with 0 and 1, respectively,
        in the speed_mode_T/P columns.

    Returns
    -------
    df : pandas dataframe
        Dataframe created with the pandas ``read_csv`` function, with an
        additional petal_id column.
    """
    # Get the path to use.
    if path is None:
        if os.getenv("NERSC_HOST") is not None:
            path = "/global/cfs/cdirs/desi/engineering/focalplane/hwtables"
        elif os.getenv("DOS_SITE") is not None:
            if expid >= 91163:
                path = "/global/cfs/cdirs/desi/engineering/focalplane/hwtables"
            else:
                path = "/data/msdos/focalplane/fp_temp_files/"
    if path is None:
        raise ValueError("Must specify a path except at NERSC or KPNO.")
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError('Invalid path: "{path}".')
    if verbose:
        print(f'Using path: "{path}".')
    # Look for any files associated with this exposure.
    files = sorted(path.glob(f"hwtables_ptlid??_{expid}_*.csv"))
    if len(files) == 0:
        raise ValueError(f"No files found for expid {expid}.")
    # Decode petal_id and exp_iter from each filename.
    petal_ids, exp_iters = [], []
    expstr = str(expid)
    for file in files:
        name = file.name
        petal_ids.append(int(name[14:16]))
        lo = name.index(expstr) + len(expstr) + 1
        hi = name.index("_", lo)
        exp_iters.append(int(name[lo:hi]) if hi > lo else 0)
    if verbose:
        print(f"Found petal_ids {set(petal_ids)} and exp_iters {set(exp_iters)}.")
    # Are the request petal_id and exp_iter available?
    if petal_id is not None and exp_iter is not None:
        if (petal_id, exp_iter) not in zip(petal_ids, exp_iters):
            raise ValueError(
                f"Requested petal_id={petal_id} and exp_iter={exp_iter} not found for expid {expid}."
            )
    elif petal_id is not None:
        if petal_id not in petal_ids:
            raise ValueError(
                f"Requested petal_id={petal_id} not found for expid {expid}."
            )
    elif exp_iter is not None:
        if exp_iter not in exp_iters:
            raise ValueError(
                f"Requested exp_iter={exp_iter} not found for expid {expid}."
            )
    # Read CSV files.
    dfs = []
    for i, file in enumerate(files):
        if petal_id not in (petal_ids[i], None):
            continue
        if exp_iter not in (exp_iters[i], None):
            continue
        if verbose:
            print(f"Reading {file}")
        df = pd.read_csv(file)
        # Add petal_id, exp_iter columns.
        df["petal_id"] = petal_ids[i]
        df["exp_iter"] = exp_iters[i]
        dfs.append(df)
    # Combine all CSVs.
    df = pd.concat(dfs, ignore_index=True)
    # Sort by increasing exp_iter
    df.sort_values("exp_iter", inplace=True, ignore_index=True)
    if compress:
        # Apply optional compression steps
        df = df.drop(
            columns=[
                "busid",
                "canid",
                "petal_id",
                "move_time",
                "total_time",
                "required",
                "failed_to_send",
                "nrows",
            ]
        )
        for col in (
            "motor_steps_P",
            "motor_steps_T",
            "speed_mode_T",
            "speed_mode_P",
            "postpause",
        ):
            df[col] = df[col].str.replace(", ", ",")
        for col in "speed_mode_T", "speed_mode_P":
            df[col] = df[col].str.replace("'creep'", "0").str.replace("'cruise'", "1")
    return df


class Scheduler:
    """Initialize a new move scheduler for a positioner."""

    CLOCK = 18000  # tick frequency in Hz
    PAUSE = 18  # clock ticks per unit of pause
    CREEP_STEP = 0.1  # motor rotation in degrees for a single creep step
    CRUISE_RATIO = 33  # ratio of cruise and creep motor rotations for a single step
    GEAR_RATIO = (
        46.0 / 14.0 + 1
    ) ** 4  # Namiki "337:1" gear box (GEAR_TYPE_P,T in constants db)

    default_const = dict(
        CREEP_PERIOD=2,
        SPINUPDOWN_PERIOD=12,
        MOTOR_CCW_DIR_T=-1,
        MOTOR_CCW_DIR_P=-1,
        GEAR_TYPE_T="namiki",
        GEAR_TYPE_P="namiki",
    )

    default_calib = dict(
        length_r1=3.0,
        length_r2=3.0,
        offset_t=0.0,
        offset_p=0.0,
        offset_x=0.0,
        offset_y=0.0,
        gear_calib_t=1.0,
        gear_calib_p=1.0,
    )

    class Motor:
        """ """

        def __init__(self, CREEP_PERIOD=2, SPINUPDOWN_PERIOD=12, ccw=-1, ratio=1):
            self.CREEP_PERIOD = CREEP_PERIOD
            self.SPINUPDOWN_PERIOD = SPINUPDOWN_PERIOD
            self.dw = (
                ccw
                * ratio
                * Scheduler.CREEP_STEP
                * Scheduler.CLOCK
                / (self.CREEP_PERIOD * Scheduler.GEAR_RATIO)
            )
            self.increment = []
            self.duration = []
            self.nticks = 0

        def creep(self, steps):
            if steps == 0:
                return
            direction, nsteps = (1, steps) if steps > 0 else (-1, -steps)
            self.increment.append(direction)
            self.duration.append(self.CREEP_PERIOD * nsteps)
            self.nticks += self.duration[-1]

        def cruise(self, steps):
            if steps == 0:
                return
            direction, nsteps = (1, steps) if steps > 0 else (-1, -steps)
            nupdown = Scheduler.CRUISE_RATIO
            spin_up_down = [
                (i + 1) * direction * self.CREEP_PERIOD for i in range(nupdown)
            ]
            self.increment.extend(spin_up_down + spin_up_down[-1:] + spin_up_down[::-1])
            self.duration.extend(
                [self.SPINUPDOWN_PERIOD] * nupdown
                + [nsteps]
                + [self.SPINUPDOWN_PERIOD] * nupdown
            )
            self.nticks += 2 * self.SPINUPDOWN_PERIOD * Scheduler.CRUISE_RATIO + nsteps

        def pause(self, steps, unit=None):
            unit = unit or Scheduler.PAUSE
            if steps < 0:
                raise ValueError("Invalid steps < 0.")
            if steps == 0:
                return
            self.increment.append(0)
            self.duration.append(steps * unit)
            self.nticks += self.duration[-1]

        def move(self, steps, mode, pause):
            if mode == "creep":
                self.creep(steps)
            elif mode == "cruise":
                self.cruise(steps)
            else:
                raise ValueError(f'Invalid move mode "{mode}".')
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
        if const["GEAR_TYPE_T"] != "namiki" or const["GEAR_TYPE_P"] != "namiki":
            raise ValueError('Only "namiki" gears are supported.')
        self.const = const
        self.calib = calib
        self.finalized = False

    def plan(self, hwtable):
        """Compute the angular motion specified in a hardware move table.

        Results are stored in the time and angle arrays of our motor_T/P attributes.
        The computed angles specify changes in the theta, phi angles, as a function
        of time, that would result assuming no collisions (with a hard stop or
        other positioner).
        """
        # Convert pandas series to dictionary if necessary.
        try:
            hwtable = hwtable.iloc[0].to_dict()
        except AttributeError:
            pass
        # Convert args from strings to arrays of python types.
        args = [
            json.loads(hwtable[name].replace("'", '"'))
            for name in (
                "motor_steps_T",
                "speed_mode_T",
                "motor_steps_P",
                "speed_mode_P",
                "postpause",
            )
        ]
        # Check that arrays have matching lengths.
        try:
            lens = [len(arg) for arg in args]
            assert min(lens) == max(lens)
        except (TypeError, AssertionError):
            raise RuntimeError("Arguments must be arrays of the same length.")

        # Schedule the move table.
        self.motor_T = Scheduler.Motor(
            self.const["CREEP_PERIOD"],
            self.const["SPINUPDOWN_PERIOD"],
            self.const["MOTOR_CCW_DIR_T"],
            self.calib["gear_calib_t"],
        )
        self.motor_P = Scheduler.Motor(
            self.const["CREEP_PERIOD"],
            self.const["SPINUPDOWN_PERIOD"],
            self.const["MOTOR_CCW_DIR_P"],
            self.calib["gear_calib_p"],
        )
        for steps_T, mode_T, steps_P, mode_P, pause in zip(*args):
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
        assert self.motor_T.time[-1] == self.motor_P.time[-1]
        self.duration = self.motor_T.time[-1]
        self.finalized = True

    def get_path_for_move(self, move):
        """Wrapper for get_path that uses a move record for the final angles to use."""
        # Convert pandas series to dictionary if necessary.
        try:
            move = move.iloc[0].to_dict()
        except AttributeError:
            pass
        return self.get_path(move["pos_t"], move["pos_p"], external=False)

    def get_path(self, t_final, p_final, external=True, max_tstep=0.05):
        """Calculate the theta and phi arm paths in various coordinate systems.

        Parameters
        ----------
        t_final : float
            Final theta angle in degrees.
        p_final : float
            Final phi angle in degrees.
        external : bool
            Final angles are external when True, else internal and will be offset
            by the calibration offset_t,p values.
        max_tstep : float or None
            Maximum time step in seconds to use when tabulating the path. When None,
            no maximum is applied and the returned tabulation exactly captures the
            angular motion in its most compact form (via linear interpolation), but
            the (x,y) components might have large jumps.
        """
        if not self.finalized:
            raise RuntimeError("Call the plan() method before calculating a path.")
        if not external:
            # Convert from internal to external angles.
            t_final = desimeter.transform.pos2ptl.int2ext(
                t_final, self.calib["offset_t"]
            )
            p_final = desimeter.transform.pos2ptl.int2ext(
                p_final, self.calib["offset_p"]
            )
        # The motor rotations are tabulated on independent time grids so combine them
        # to a common grid now.
        if max_tstep is None:
            # Take the union of the two motor grids.
            self.time = np.unique(
                np.concatenate((self.motor_T.time, self.motor_P.time))
            )
        else:
            # Use the smallest tstep < max_tstep that evenly divides this motion's duration.
            nt = int(np.ceil(self.duration / max_tstep)) + 1
            self.time = np.linspace(0, self.duration, nt)
            assert self.time[1] <= max_tstep
        # Interpolate the angles onto the same grid, adjusted to end at t_final, p_final.
        self.t_ext = np.interp(
            self.time,
            self.motor_T.time,
            t_final + self.motor_T.angle - self.motor_T.angle[-1],
        )
        self.p_ext = np.interp(
            self.time,
            self.motor_P.time,
            p_final + self.motor_P.angle - self.motor_P.angle[-1],
        )
        # Convert from angles to petal local (x,y).
        self.x_loc, self.y_loc = desimeter.transform.pos2ptl.ext2loc(
            self.t_ext, self.p_ext, self.calib["length_r1"], self.calib["length_r2"]
        )
        # (xt,yt) refers to the phi axis at the end of the theta arm, calculated by setting r2=0.
        self.xt_loc, self.yt_loc = desimeter.transform.pos2ptl.ext2loc(
            self.t_ext, self.p_ext, self.calib["length_r1"], 0
        )
        # Convert from local (x,y) to flat (x,y).
        self.x_flat = desimeter.transform.pos2ptl.loc2flat(
            self.x_loc, self.calib["offset_x"]
        )
        self.y_flat = desimeter.transform.pos2ptl.loc2flat(
            self.y_loc, self.calib["offset_y"]
        )
        self.xt_flat = desimeter.transform.pos2ptl.loc2flat(
            self.xt_loc, self.calib["offset_x"]
        )
        self.yt_flat = desimeter.transform.pos2ptl.loc2flat(
            self.yt_loc, self.calib["offset_y"]
        )
        # Convert from flat (x,y) to petal (x,y).
        self.x_ptl, self.y_ptl = desimeter.transform.pos2ptl.flat2ptl(
            self.x_flat, self.y_flat
        )
        self.xt_ptl, self.yt_ptl = desimeter.transform.pos2ptl.flat2ptl(
            self.xt_flat, self.yt_flat
        )
        # Convert from petal (x,y) to focal plane (x,y,z).
        self.x_fp, self.y_fp, self.z_fp = desimeter.transform.ptl2fp.ptl2fp(
            self.calib["petal_loc"], self.x_ptl, self.y_ptl
        )
        self.xt_fp, self.yt_fp, self.zt_fp = desimeter.transform.ptl2fp.ptl2fp(
            self.calib["petal_loc"], self.xt_ptl, self.yt_ptl
        )
