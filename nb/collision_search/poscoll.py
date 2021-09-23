#!/usr/bin/env python
# coding: utf-8

# Imports

import os
import re
from functools import partial
from collections import namedtuple
import numbers
import glob
import logging
import getpass
import psycopg2
import h5py
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy
from astropy.stats import biweight_midvariance
from astropy.time import Time

import desimeter.transform
import desimeter.transform.pos2ptl as pos2ptl

# Constants

DESI_DATABASE = "desi_dev"
DESI_DBUSER = "desi_reader"
DESI_DBHOST = "db.replicator.dev-cattle.stable.spin.nersc.org"
DESI_DBPASSWORD = ""
DESI_DBPORT = 60042
MOVE_RE = re.compile(r".*(creep|cruise) (\-?\d+\.?\d*)")
T_MOVE_BINS = np.arange(-100, 100, 5)
P_MOVE_BINS = np.arange(-42.5, 42.5, 2.5)

PETAL_IDS = tuple(range(2, 4))
PETAL_NAME_MAP_FNAME = (
    "/global/cfs/cdirs/desi/users/skent/plate/desi/etc/desi/metrology/desi.map"
)
CONFIG_TEMPLATE = (
    "/global/cfs/cdirs/desi/users/skent/data/desi/{expid}/config-{expid}.*.dat"
)

logging.basicConfig(format="%(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel("DEBUG")
LOGGER.info("Starting")

# Interface functions


def desidb_query(query):
    """Query the DESI database, and return a `pandas.DataFrame` result.

    Parameters
    ----------
    query : str
       The query to be executed.

    Returns
    -------
    result : pandas.DataFrame
       The result of the query.
    """

    dosdb_connect = partial(
        psycopg2.connect,
        database=DESI_DATABASE,
        user=DESI_DBUSER,
        host=DESI_DBHOST,
        port=DESI_DBPORT,
        password=DESI_DBPASSWORD,
    )

    with dosdb_connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def query_nights_with_exposures():
    """Query the online database for all nights with exposures.

    Returns
    -------
    result : pandas.DataFrame
       The result of the query.
    """
    nights = desidb_query(
        """SELECT MIN(date_obs) AS first_time,
                  MAX(date_obs) AS last_time,
                  MIN(id) AS first_id,
                  MAX(id) AS max_id,
                  FLOOR(EXTRACT(JULIAN FROM date_obs) -2400000.5 - 112/360.0 + 0.5)::INT AS night_mjd,
                  COUNT(*) AS num_exposures
           FROM exposure
           GROUP BY FLOOR(EXTRACT(JULIAN FROM date_obs) -2400000.5 - 112/360.0 + 0.5)::INT
           ORDER BY FLOOR(EXTRACT(JULIAN FROM date_obs) -2400000.5 - 112/360.0 + 0.5)::INT
    """
    )
    nights.dropna(subset=["night_mjd"], inplace=True)
    nights["night_mjd"] = nights["night_mjd"].astype(int)
    return nights


def query_night_exposures(night_mjd):
    """Query the online database for ids of all exposures on a night.
    Parameters
    ----------
    night_mjd : int
       The Modified Julian Date of the desired night.

    Returns
    -------
    exposure_ids : tuple [int]
       The result of the query.

    """
    exposures = desidb_query(
        f"""SELECT id
            FROM exposure
            WHERE FLOOR(EXTRACT(JULIAN FROM date_obs) -2400001 - 112/360.0 - 0.5)::INT={night_mjd}
            ORDER BY id"""
    )
    return tuple(exposures.id.values)


def find_exposure_fvcprocs(expid, config_template=CONFIG_TEMPLATE):
    """Find positioner-corr file names of all executions of PlateMaker's fvcproc on an exposure.

    Parameters
    ----------
    expid : `int`
        The DESI exposure id
    config_template : `str`
        The glob template for PlateMaker config files.
        This pattern should include the full path.

    Returns
    -------
    positioner_corr_fnames : `list` `[str]`
        A list of file names of PlateMaker positioner-corr files
    """

    positioner_corr_fnames = []
    for config_file in sorted(glob.glob(config_template.format(expid=expid))):
        with open(config_file) as fp:
            first_line = fp.readline()
            if not first_line.startswith("p fvcproc"):
                continue
            for line in fp:
                if line.startswith("w positioner-corr"):
                    fname = line.split(" ")[1].strip()
                    full_fname = os.path.join(
                        os.path.dirname(config_file), fname
                    )
                    positioner_corr_fnames.append(full_fname)
    return positioner_corr_fnames


def find_night_fvcprocs(night_mjd, config_template=CONFIG_TEMPLATE):
    """Find positioner-corr file names of all executions of PlateMaker's fvcproc on a night.

    Parameters
    ----------
    night_mjd : `int`
        The Modified Julian Date of the night
    config_template : `str`
        The glob template for PlateMaker config files.
        This pattern should include the full path.

    Returns
    -------
    positioner_corr_fnames : `list` `[str]`
        A list of file names of PlateMaker positioner-corr files
    """
    night_exposures = query_night_exposures(night_mjd)
    positioner_corr_fnames = []
    for night_mjd in sorted(night_exposures):
        positioner_corr_fnames += find_exposure_fvcprocs(
            night_mjd, config_template
        )
    return positioner_corr_fnames


def read_night_positions(night_mjd):
    """Read fiber positions from positioner-corr files for a night.

    Parameters
    ----------
    night_mjd : `int`
        The MJD of the night for which to read positions

    Returns
    -------
    positions : `pandas.DataFrame`
        A table of position data

    """
    pos_corr_fnames = find_night_fvcprocs(night_mjd)
    if len(pos_corr_fnames) < 1:
        return None

    position_dfs = []
    for fname in pos_corr_fnames:
        positions = pd.read_table(
            fname, skiprows=(1, 2, 3), delim_whitespace=True
        )
        positions.rename(
            columns={
                "#PETAL_LOC": "petal_loc",
                "DEVICE_LOC": "device_loc",
                "FVC_ID": "fvc_id",
                "DELTA_X": "delta_x",
                "DELTA_Y": "delta_y",
                "X": "obs_x",
                "Y": "obs_y",
                "FLAGS": "flags",
            },
            inplace=True,
        )
        positions["fname"] = os.path.basename(fname)
        positions["row"] = np.arange(len(positions))
        position_dfs.append(positions)

    positions = pd.concat(position_dfs)
    if "flags" not in positions:
        return None

    positions = positions.query("(obs_x != 0.0) or (obs_y != 0.0)").copy()
    positions["flags"] = positions["flags"].astype(int)
    positions["ctrl_disabled"] = bit_flag(positions["flags"], 16)
    positions["nonfunctional"] = bit_flag(positions["flags"], 24)
    positions["ambiguous"] = bit_flag(positions["flags"], 11)
    positions["stationary"] = bit_flag(positions["flags"], 14)
    positions["intact"] = bit_flag(positions["flags"], 17)
    positions.set_index(
        ["petal_loc", "device_loc", "fname", "row"], drop=True, inplace=True
    )
    positions["obs_x_diff"] = positions.groupby(["petal_loc", "device_loc"])[
        "obs_x"
    ].diff()
    positions["obs_y_diff"] = positions.groupby(["petal_loc", "device_loc"])[
        "obs_y"
    ].diff()
    positions["obs_diff"] = np.hypot(
        positions.obs_x_diff, positions.obs_y_diff
    )

    positions["night"] = pd.to_datetime(
        Time(night_mjd, format="mjd", scale="utc").datetime64
    )
    positions["expid"] = (
        positions.reset_index()
        .fname.str.extract(r".*positioner-corr-([0-9]+)\.[0-9]+.dat")
        .values.astype(int)
    )
    positions["configid"] = (
        positions.reset_index()
        .fname.str.extract(r".*positioner-corr-[0-9]+\.([0-9]+).dat")
        .values.astype(int)
    )

    positions["night_mjd"] = night_mjd

    positions["night"] = pd.to_datetime(
        Time(night_mjd, format="mjd", scale="utc").datetime64
    )

    positions["expid"] = (
        positions.reset_index()
        .fname.str.extract(r".*positioner-corr-([0-9]+)\.[0-9]+.dat")
        .values.astype(int)
    )
    positions["configid"] = (
        positions.reset_index()
        .fname.str.extract(r".*positioner-corr-[0-9]+\.([0-9]+).dat")
        .values.astype(int)
    )

    positions.reset_index(inplace=True)
    positions.set_index(
        #            ["night_mjd", "petal_loc", "device_loc", "fname", "row"]
        [
            "night_mjd",
            "petal_loc",
            "device_loc",
            "expid",
            "configid",
            "row",
        ],
        inplace=True,
    )

    return positions


def read_many_nights(nights=None, num_nights=None, move_thresh=None):
    """Read fiber positions for many nights.

    Parameters
    ----------
    nights : `list` of `[int]`
        A list of MJDs of nights to read
    num_nights : `int` or None
        If not None, pick num_nights nights randomly and read only those.
    move_thresh : `float`
        If not None, for any given night, include only positioners
        that moved by move_thresh mm at some point during the night.

    Returns
    -------
    positions : `pandas.DataFrame`
        Position data
    """
    if nights is None:
        nights = query_nights_with_exposures()

    these_nights = (
        nights if num_nights is None else nights.sample(n=num_nights)
    )
    night_position_dfs = []
    for night_mjd in these_nights.night_mjd:
        LOGGER.info(f"Reading {night_mjd}")
        try:
            full_night_position_df = read_night_positions(night_mjd)
        except PermissionError:
            print(f"Permission error on {night_mjd}")
            continue

        if full_night_position_df is None:
            continue

        night_position_df = full_night_position_df.query(
            "nonfunctional and not ambiguous"
        ).copy()

        if move_thresh is not None:
            # If a positioner has any moves over the threshhold on a night
            # keep all moves for the night by that positioner
            keep = (
                night_position_df.query(f"obs_diff>{move_thresh}")
                .reset_index()
                .set_index(["petal_loc", "device_loc"])
                .index.drop_duplicates()
            )
            if len(keep) > 0:
                night_position_df["keep"] = pd.DataFrame(
                    {"keep": True}, index=keep
                )
                night_position_df.dropna(inplace=True)
                night_position_df.drop(columns="keep", inplace=True)
            else:
                # Nothing to keep, so drop all rows
                night_position_df = night_position_df.iloc[0:0]

        night_position_df = night_position_df.copy()
        # deal with pandas view/copy awkwardness
        night_position_dfs.append(night_position_df)

    if len(night_position_dfs) > 0:
        positions = pd.concat(night_position_dfs)
    else:
        positions = pd.DataFrame()
    return positions


def bit_flag(flags, n):
    """Return True iff the nth bit of an integer is set

    Parameters
    ----------
    flags : `int`
        The integer in which to look for flag bits
    n : `int`
        The bit for the flag of interest

    Returns
    -------
    flag_value : `bool`
        True iff the flag is set
    """

    value = (flags & 2 ** n) > 0
    return value


def plot_pos(night_mjd, petal_loc, device_loc, positions):
    """Plot the measured positions for a positioner on a night.

    Parameters
    ----------
    night_mjd : `int`
        The Modified Julian Date of the night to plot
    petal_loc : `int`
        The petal location
    device_loc : `int`
        The device location
    positions : `pandas.DataFrame`
        The DataFrame with the device positions
        Must include expid, obs_x, and obs_y columns

    Returns
    -------
    fig_ax : `tuple`
        Tuple with `matplotlib.Figure` and `matplotlib.Axes`

    """
    these_positions = positions.loc[(night_mjd, petal_loc, device_loc)].copy()
    night = these_positions["night"][0]
    label = f"Petal {petal_loc} device {device_loc} on {str(night)[:11]}"
    these_positions["fvc"] = (
        these_positions["expid"] + these_positions["configid"] / 100
    )
    fig, axes = plt.subplots(1, 3, figsize=(8 * 3, 5))

    ax = axes[0]
    these_positions.plot("obs_x", "obs_y", ax=ax, legend=False)
    ax.set_ylabel("obs_y")
    ax.set_aspect(1, adjustable="datalim")

    for ax, col in zip(axes[1:], ("obs_x", "obs_y")):
        these_positions.plot(
            "fvc", col, marker="+", markerfacecolor="red", ax=ax, legend=False
        )
        ax.set_ylabel(col)
        ax.set_xlabel("expid.configid")

    axes[1].set_title(label)

    return fig, axes


def plot_path_with_neighbors(
    night_mjd, petal_loc, device_loc, plot_size=20, night_positions=None
):
    """Plot the measured positions for a positioner on a night.

    Parameters
    ----------
    night_mjd : `int`
        The Modified Julian Date of the night to plot
    petal_loc : `int`
        The petal location
    device_loc : `int`
        The device location
    plot_size : `float`
        The size of the plot around the fiber, in mm
        height (and width)
    night_positions : `pandas.DataFrame`
        The DataFrame with the device positions
        Must include expid, obs_x, and obs_y columns

    Returns
    -------
    fig_ax : `tuple`
        Tuple with `matplotlib.Figure` and `matplotlib.Axes`

    """
    logging.info(f"Loading night of {night_mjd}")

    if night_positions is None:
        night_positions = read_night_positions(night_mjd)

    night = str(night_positions["night"].iloc[0])[:10]
    this_petal_positions = night_positions.loc[petal_loc]

    print(f"{night_mjd}, {petal_loc}, {device_loc}")

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 5))
    axes[1].set_title(
        f"Petal {petal_loc}, device {device_loc} on {str(night)[:11]}"
    )

    ax = axes[0]

    these_positions = this_petal_positions.loc[device_loc].copy()
    these_positions["fvc"] = (
        these_positions["expid"] + these_positions["configid"] / 100
    )
    these_positions.plot("obs_x", "obs_y", marker="o", ax=ax, label=device_loc)
    ax.set_ylabel("obs_y")
    ax.set_aspect(1, adjustable="datalim")

    max_move = these_positions.obs_diff.max()
    ax.annotate(
        f"Max move: {max_move:.2f}mm", (0.05, 0.05), xycoords="axes fraction"
    )

    for ax, col in zip(axes[1:], ("obs_x", "obs_y")):
        these_positions.plot(
            "fvc", col, marker="+", markerfacecolor="red", ax=ax, legend=False
        )
        ax.set_ylabel(col)
        ax.set_xlabel("expid.configid")

    # Go back to the first ax to add neighbors

    ax = axes[0]
    x_center = np.mean(ax.get_xlim())
    y_center = np.mean(ax.get_ylim())
    x_lim = x_center - plot_size / 2, x_center + plot_size / 2
    y_lim = y_center - plot_size / 2, y_center + plot_size / 2

    for (
        other_device_loc
    ) in this_petal_positions.reset_index().device_loc.unique():
        these_positions = this_petal_positions.loc[other_device_loc]
        if (
            len(
                these_positions.query(
                    f"(obs_x > {x_lim[0]}) and (obs_x < {x_lim[1]})"
                )
            )
            == 0
        ):
            continue

        if (
            len(
                these_positions.query(
                    f"(obs_y > {y_lim[0]}) and (obs_y < {y_lim[1]})"
                )
            )
            == 0
        ):
            continue

        these_positions = this_petal_positions.loc[other_device_loc]
        these_positions.plot(
            "obs_x", "obs_y", marker="o", ax=ax, label=other_device_loc
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.legend(loc=2)
    plt.tight_layout()

    return fig, axes
