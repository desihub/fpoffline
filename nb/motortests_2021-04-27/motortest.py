#!/usr/bin/env python
# coding: utf-8

# Imports

import os
import re
from functools import partial
from collections import namedtuple
import numbers
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

import desimeter.transform
import desimeter.transform.pos2ptl as pos2ptl

# Constants

DESI_DATABASE = "desi_dev"
DESI_DBUSER = "desi_reader"
DESI_DBHOST = "db.replicator.dev-cattle.stable.spin.nersc.org"
DESI_DBPORT = 60042
MOVE_RE = re.compile(r".*(creep|cruise) (\-?\d+\.?\d*)")
T_MOVE_BINS = np.arange(-100, 100, 5)
P_MOVE_BINS = np.arange(-42.5, 42.5, 2.5)

# Interface functions


def desidb_query(query, password):
    """Query the DESI database, and return a `pandas.DataFrame` result.

    Parameters
    ----------
    query : str
       The query to be executed.
    password : str
       The database password.

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
        password=password,
    )

    with dosdb_connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def get_exposure_limits(exposure_id, password):
    """Query time limits for positioner moves in an exposure.

    Parameters
    ----------
    exposure_id : int
       The exposure id
    password : str
       The database password

    Returns
    -------
    limits : pandas.DataFrame
       A dataframe with the time_recorded for the first and last moves
       in the exposure.
    """
    limit_query = f"""
        SELECT {exposure_id} AS exposure_id, MIN(time_recorded) AS record_time, MIN(pos_move_index) AS pos_move_index
          FROM positioner_moves
          WHERE exposure_id BETWEEN {exposure_id} AND (SELECT MIN(exposure_id) FROM positioner_moves WHERE exposure_id>{exposure_id})
          GROUP BY exposure_id
          ORDER BY record_time
    """
    these_limits = (
        desidb_query(limit_query, password)
        .assign(which=["this", "next"])
        .set_index(["which", "exposure_id"])
        .unstack("which")
    )
    return these_limits


def get_moves(exposure_id, password):
    """Query move and calibration parameters.

    Parameters
    ----------
    exposure_id : int
       The exposure id
    password : str
       The database password

    Returns
    -------
    limits : pandas.DataFrame
       A dataframe with parameters and calibration from the
       positioner_moves and positioner_calibration tables in the database.
    """
    exp_limits = get_exposure_limits(exposure_id, password)

    min_time = (
        exp_limits.loc[exposure_id, ("record_time", "this")].isoformat()[:19]
        + "Z"
    )
    max_time = (
        exp_limits.loc[exposure_id, ("record_time", "next")].isoformat()[:19]
        + "Z"
    )
    move_query = f"""
SELECT pm.petal_id,
       pm.device_loc, 
       pm.pos_id, 
       pm.pos_move_index, 
       pm.time_recorded, 
       pm.bus_id, 
       pm.pos_t,
       pm.pos_p,
       LAG(pm.pos_t, 1) OVER (
                 PARTITION BY pm.bus_id, pm.device_loc, pm.petal_id, pm.pos_id, pm.site
                 ORDER BY pm.time_recorded) AS previous_pos_t,
       LAG(pm.pos_p, 1) OVER (
                 PARTITION BY pm.bus_id, pm.device_loc, pm.petal_id, pm.pos_id, pm.site
                 ORDER BY pm.time_recorded) AS previous_pos_p,
       pm.obs_x,
       pm.obs_y,
       pm.ptl_x,
       pm.ptl_y,
       pm.ptl_z,
       pm.move_cmd, 
       pm.move_val1, 
       pm.move_val2,
       pm.total_cruise_moves_t,
       pm.total_creep_moves_t,
       pm.total_cruise_moves_p,
       pm.total_creep_moves_p,
       pm.log_note, 
       pm.exposure_iter, 
       pm.flags, 
       pc.pos_calib_index,
       pc.length_r1,
       pc.length_r2,
       pc.offset_x,
       pc.offset_y,
       pc.offset_p,
       pc.offset_t,
       pc.physical_range_p,
       pc.physical_range_t,
       pc.classified_as_retracted, 
       pc.calib_note, 
       pc.fiber_intact, 
       pc.device_classified_nonfunctional, 
       pc.calib_time_recorded, 
       pc.next_time AS next_calib_time
FROM positioner_moves AS pm
JOIN (SELECT *,
             time_recorded AS calib_time_recorded,
             -- Get the next calibration time by grouping all calibrations
             -- by positioner, ordering by time, and shifting by one
             LAG(time_recorded, -1) OVER (
                 PARTITION BY bus_id, device_loc, petal_id, pos_id, site
                 ORDER BY time_recorded) AS next_time
             FROM positioner_calibration
     ) AS pc
     ON pm.bus_id=pc.bus_id
        AND pm.device_loc=pc.device_loc
        AND pm.petal_id=pc.petal_id
        AND pm.pos_id=pc.pos_id
        AND pm.site=pc.site
        AND pm.time_recorded > pc.time_recorded -- positioner move is after calibration time
        AND ((pm.time_recorded < pc.next_time)  -- and positioner move is either before the next calibration time,
              OR (pc.next_time IS NULL))        -- or it is after the last calibration time
WHERE pm.time_recorded >= '{min_time}' -- Cannot use BETWEEN here, because it has inclusive bounds
  AND pm.time_recorded < '{max_time}'
ORDER BY pm.petal_id, pm.pos_id, pm.time_recorded"""
    moves = (
        desidb_query(move_query, password)
        .set_index(["petal_id", "pos_id", "pos_move_index"])
        .sort_index()
    )
    return moves


def extract_move(move_string):
    """Compute the total move from move_command string.

    Parameters
    ----------
    move_string : str
       The move description in the format of the move_string column in
       the positioner_moves column in the DESI database.

    Returns
    -------
    total_move : float
       The total move, in degrees.
    """
    total_move = 0
    for move_command in move_string.split(";"):
        matched_move = MOVE_RE.match(move_command.strip())
        if matched_move is None:
            return np.nan
        move_value = float(matched_move.group(2))
        total_move += move_value
    return total_move


def compute_derived_columns(
    moves,
    t_move_bins=T_MOVE_BINS,
    p_move_bins=P_MOVE_BINS,
    test_coord="t",
    inplace=False,
):
    """Compute an assortment of derived columns for moves.

    Parameters
    ----------
    moves : pandas.DataFrame
       Contains move data, as produced by get_moves().
    t_move_bins : numpy.array
       bin boundaries for binned plots of moves in theta
    p_move_bins : numpy.array
       bin boundaries for binned plots of moves in phi
    inplace : bool
       modify passed DataFrame if true, return modified copy otherwise

    Returns
    -------
    moves : pandas.DataFrame
       contains passed and derived columns.
    """
    if not inplace:
        moves = moves.copy()

    moves["calc_t"], moves["calc_p"], moves["ptl2int_flags"] = pos2ptl.ptl2int(
        moves.ptl_x,
        moves.ptl_y,
        moves.length_r1,
        moves.length_r2,
        moves.offset_t,
        moves.offset_p,
        moves.offset_x,
        moves.offset_y,
        moves.pos_t,
    )

    moves["cmd_ptl_x"], moves["cmd_ptl_y"] = pos2ptl.int2ptl(
        moves.pos_t,
        moves.pos_p,
        moves.length_r1,
        moves.length_r2,
        moves.offset_t,
        moves.offset_p,
        moves.offset_x,
        moves.offset_y,
    )

    const_coord = "p" if test_coord == "t" else "t"
    moves[f"test_{const_coord}"] = moves.groupby(["petal_id", "pos_id"])[
        f"pos_{const_coord}"
    ].median()
    moves[f"has_test_{const_coord}"] = np.isclose(
        moves[f"test_{const_coord}"], moves[f"pos_{const_coord}"]
    )
    moves["stored_new"] = (
        moves.log_note.str[:]
        == "Stored new: OBS_X OBS_Y PTL_X PTL_Y PTL_Z FLAGS"
    )
    moves["has_dTdP"] = moves.move_cmd.str.match(r"^dTdP")
    moves["working_moves"] = (
        moves[f"has_test_{const_coord}"]
        & moves["has_dTdP"]
        & ~(
            moves["classified_as_retracted"]
            | moves["device_classified_nonfunctional"]
        )
    )

    moves["move_p"] = 0.0
    moves.loc[moves["has_dTdP"], "move_p"] = moves.loc[
        moves["has_dTdP"], "move_val2"
    ].apply(extract_move)
    moves["move_t"] = 0.0
    moves.loc[moves["has_dTdP"], "move_t"] = moves.loc[
        moves["has_dTdP"], "move_val1"
    ].apply(extract_move)

    moves["prev_calc_p"] = moves.groupby(["petal_id", "pos_id"])[
        "calc_p"
    ].shift()
    moves["dcalc_p"] = moves["calc_p"] - moves["prev_calc_p"]
    moves["prev_calc_t"] = moves.groupby(["petal_id", "pos_id"])[
        "calc_t"
    ].shift()
    moves["dcalc_t"] = moves["calc_t"] - moves["prev_calc_t"]

    moves["ptl_x_diff"] = moves.cmd_ptl_x - moves.ptl_x
    moves["ptl_y_diff"] = moves.cmd_ptl_y - moves.ptl_y
    moves["ptl_dist"] = np.sqrt(moves.ptl_x_diff ** 2 + moves.ptl_y_diff ** 2)

    moves[f"binned_move_p"] = pd.cut(moves["move_p"], bins=p_move_bins)
    moves[f"binned_move_t"] = pd.cut(moves["move_t"], bins=t_move_bins)

    return moves


def plot_move_boxplot(moves, bins, coordinate="t"):
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    fig, ax = plt.subplots(1)
    moves.boxplot(
        f"dcalc_{coordinate}",
        by=f"binned_move_{coordinate}",
        whis=[5, 95],
        showfliers=False,
        ax=ax,
    )
    ax.plot(1 + np.arange(len(bins) - 1), bin_centers, color="blue")
    ax.tick_params(axis="x", labelrotation=90)
    return fig, ax


def plot_dist_hists(moves):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bins = np.arange(-0.1, 0.1, 0.001)
    moves.ptl_x_diff.hist(bins=bins, ax=axes[0, 0])
    axes[0, 0].set_title("cmd_ptl_x - ptl_x")
    moves.ptl_y_diff.hist(bins=bins, ax=axes[0, 1])
    axes[0, 1].set_title("cmd_ptl_y - ptl_y")
    moves.ptl_dist.hist(bins=bins, ax=axes[0, 2])
    axes[0, 2].set_xlim(0, np.max(bins))
    axes[0, 2].set_title("distance between cmd and FVC")

    bins = np.arange(-1, 1, 0.01)
    moves.ptl_x_diff.hist(bins=bins, log=True, ax=axes[1, 0])
    moves.ptl_y_diff.hist(bins=bins, log=True, ax=axes[1, 1])
    moves.ptl_dist.hist(bins=bins, log=True, ax=axes[1, 2])
    axes[1, 2].set_xlim(0, np.max(bins))

    axes[1, 1].set_xlabel("difference (mm)")
    axes[0, 0].set_ylabel("# moves")
    return fig, axes


def assign_distance_peaks(moves, boundary1, boundary2):
    moves = moves.copy()
    peak1_move = moves["ptl_dist"] <= boundary1
    peak2_move = np.logical_and(
        boundary1 < moves["ptl_dist"], moves["ptl_dist"] < boundary2
    )
    mismatch_move = ~(peak1_move | peak2_move)
    moves["peak1"] = peak1_move
    moves["peak2"] = peak2_move
    moves["mismatch"] = mismatch_move
    return moves


def plot_move_histogram_by_peak(
    moves, bins, coord="t", include_mismatched=True
):
    mismatched = moves.groupby(["petal_id", "pos_id"])["mismatch"].any()
    peak1 = moves.groupby(["petal_id", "pos_id"])["peak1"].all()
    peak2 = ~(mismatched | peak1)

    ncols = 3 if include_mismatched else 2
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3))
    axes_iter = iter(axes)

    column = f"dcalc_{coord}"
    if coord == "t":
        xlabel = r"measured move in $\theta$ (deg.)"
    elif coord == "p":
        xlabel = r"measured move in $\phi$ (deg.)"
    else:
        raise NotImplementedError()

    if include_mismatched:
        ax = next(axes_iter)
        moves.loc[mismatched].hist("dcalc_t", bins=bins, ax=ax)
        ax.set_title("mismatched")
        ax.set_xlabel(xlabel)

    ax = next(axes_iter)
    moves.loc[moves["peak1"]].hist("dcalc_t", bins=bins, ax=ax)
    ax.set_title("peak 1")
    ax.set_xlabel(xlabel)

    ax = next(axes_iter)
    moves.loc[moves["peak2"]].hist("dcalc_t", bins=bins, ax=ax)
    ax.set_title("peak 2")
    ax.set_xlabel(xlabel)

    return fig, axes


def fit_delta_line_polyfit(
    df, exog_col="move_t", endog_col="dcalc_t", x_col="ptl_x", y_col="ptl_y"
):
    df = df.dropna(subset=[exog_col, endog_col, x_col, y_col])
    if len(df) == 0:
        return None

    num_points = len(df)
    if len(df[exog_col]) == 1:
        coeffs = (df.iloc[0][exog_col] / df.iloc[0][exog_col], 0)
        pred_endog = np.poly1d(coeffs)(df[exog_col])
    else:
        try:
            coeffs = np.polyfit(df[exog_col], df[endog_col], 1)
            pred_endog = np.poly1d(coeffs)(df[exog_col])
        except np.linalg.LinAlgError:
            coeffs = (np.nan, np.nan)
            pred_endog = np.full_like(df[exog_col], np.nan)
            num_points = 0

    resid = df[endog_col] - pred_endog
    resid_rms = np.sqrt(np.sum(resid ** 2))
    resid_max = np.max(resid)
    resid_min = np.min(resid)

    fit_ptl_x, fit_ptl_y = pos2ptl.int2ptl(
        df.pos_t,
        df.pos_p - df.move_p + pred_endog,
        df.length_r1,
        df.length_r2,
        df.offset_t,
        df.offset_p,
        df.offset_x,
        df.offset_y,
    )
    fit_dist = np.sqrt(
        (fit_ptl_x - df[x_col]) ** 2 + (fit_ptl_y - df[y_col]) ** 2
    )
    dist_rms = np.sqrt(np.mean(fit_dist ** 2))
    dist_mean = np.mean(fit_dist)
    dist_median = np.median(fit_dist)
    dist_max = np.max(fit_dist)
    dist_min = np.min(fit_dist)

    fit = pd.Series(
        {
            "slope": coeffs[0],
            "intercept": coeffs[1],
            "resid_rms": resid_rms,
            "resid_max": resid_max,
            "resid_min": resid_min,
            "dist_rms": dist_rms,
            "dist_max": dist_max,
            "dist_min": dist_min,
            "dist_mean": dist_mean,
            "dist_median": dist_median,
            "num_points": num_points,
        }
    )
    return fit


def fit_move_lines(moves, **kwargs):
    fit_one_move_line = partial(fit_delta_line_polyfit, **kwargs)
    move_fits = moves.groupby(["petal_id", "pos_id"]).apply(fit_one_move_line)
    return move_fits


def plot_residuals_by_param(moves, move_fits, min_points=10):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    condition = f"num_points>{min_points}"

    mismatched = moves.groupby(["petal_id", "pos_id"])["mismatch"].any()
    peak1 = moves.groupby(["petal_id", "pos_id"])["peak1"].all()
    peak2 = ~(mismatched | peak1)

    move_fits.loc[peak2].query(condition).plot.scatter(
        "slope", "resid_rms", c="orange", ax=axes[0], label="peak 2"
    )
    move_fits.loc[peak1].query(condition).plot.scatter(
        "slope",
        "resid_rms",
        xlim=(0, 2),
        ylim=(0, 40),
        c="blue",
        ax=axes[0],
        label="peak 1",
    )
    move_fits.loc[mismatched].query(condition).plot.scatter(
        "slope", "resid_rms", c="red", ax=axes[0], label="mismatched"
    )

    move_fits.loc[peak2].query(condition).plot.scatter(
        "intercept", "resid_rms", c="orange", ax=axes[1], label="peak 2"
    )
    move_fits.loc[peak1].query(condition).plot.scatter(
        "intercept",
        "resid_rms",
        xlim=(-1.2, 1.2),
        ylim=(0, 40),
        ax=axes[1],
        c="blue",
        label="peak 1",
    )
    move_fits.loc[mismatched].query(condition).plot.scatter(
        "intercept", "resid_rms", c="red", ax=axes[1], label="mismatched"
    )
    axes[0].set_ylabel("RMS difference in angle for fit (deg)")
    return fig, axes


def plot_distances_by_param(moves, move_fits, min_points=10):
    condition = f"num_points>{min_points}"

    mismatched = moves.groupby(["petal_id", "pos_id"])["mismatch"].any()
    peak1 = moves.groupby(["petal_id", "pos_id"])["peak1"].all()
    peak2 = ~(mismatched | peak1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    peak1_fits = move_fits.loc[peak1].query(condition)
    mismatched_fits = move_fits.loc[mismatched].query(condition)
    peak2_fits = move_fits.loc[peak2].query(condition)
    for col, term in enumerate(["slope", "intercept"]):
        mismatched_fits.loc[mismatched].query(condition).plot.scatter(
            term, "dist_mean", c="red", ax=axes[col], label="mismatched"
        )
        peak2_fits.loc[peak2].query(condition).plot.scatter(
            term, "dist_mean", c="orange", ax=axes[col], label="peak 2"
        )
        peak1_fits.plot.scatter(
            term, "dist_max", ax=axes[col], c="blue", label="peak 1"
        )
        axes[col].set_ylim(0, 1.5)

    return fig, axes


def plot_moves(move_fits, moves, coordinate="t", ncols=3, size=3):
    idxs = move_fits.index
    ptl_dist = moves.groupby(["petal_id", "pos_id"])["ptl_dist"].mean()[idxs]
    fit_stats = pd.DataFrame(
        {
            "raw": ptl_dist,
            "fit": move_fits["dist_mean"],
            "slope": move_fits["slope"],
            "intercept": move_fits["intercept"],
        }
    )

    nrows = np.ceil(len(idxs) / ncols).astype(int)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=False,
        figsize=(size * ncols, size * nrows),
    )
    for idx, ax in zip(idxs, axes.flatten()):
        this_fit = move_fits.loc[idx]
        moves.loc[idx].plot.scatter(
            f"move_{coordinate}", f"dcalc_{coordinate}", color="blue", ax=ax
        )
        auto_xlim = ax.get_xlim()
        ref_line = np.array(auto_xlim)
        ax.plot(ref_line, ref_line, color="red")
        ax.plot(
            ref_line,
            this_fit.intercept + ref_line * this_fit.slope,
            color="blue",
            alpha=0.2,
        )
        ax.set_xlim(auto_xlim)
        ax.set_title(f"petal {idx[0]}, pos_id {idx[1]}")
        ax.set_xlabel("Commanded move (deg)")
        ax.set_ylabel("Measured move (deg)")
    plt.tight_layout()
    return fit_stats


# In[109]:


def plot_fit_improvement(fit_stats):
    ax = fit_stats.plot.scatter("raw", "fit")
    ax.set_ylabel(
        "mean distance from position computed\nusing angle inferred from fit slope"
    )
    ax.set_xlabel("mean distance from position computed using original angle")
    ax.plot([0, 1.5], [0, 1.5], c="red")


def plot_move_difference_boxplot(moves, coordinate, move_bins):
    binned_moves = pd.DataFrame(
        {
            "diff": moves[f"move_{coordinate}"] - moves[f"dcalc_{coordinate}"],
            "bin": pd.cut(moves[f"move_{coordinate}"], bins=move_bins),
        }
    )
    bin_centers = (move_bins[1:] + move_bins[:-1]) / 2.0

    fig, axes = plt.subplots(2, figsize=(10, 10))
    axes_iter = iter(axes)

    ax = next(axes_iter)
    binned_moves.boxplot(
        "diff", by="bin", whis=[5, 95], showfliers=False, ax=ax
    )

    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel("Offset from measured move (degrees)")
    ax.set_title(None)
    ax.get_figure().suptitle("")

    ax = next(axes_iter)
    moves.hist(f"move_{coordinate}", bins=move_bins, ax=ax)
    ax.set_xlim(move_bins[0], move_bins[-1])
    ax.set_xticks(bin_centers)
    ax.set_xlabel("commanded move (degrees)")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(None)
    plt.tight_layout()
    return fig, axes


def compute_radius(x, y, x0, y0):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return r


def compute_radial_distance(x, y, x0, y0):
    r = compute_radius(x, y, x0, y0)
    d = r - np.mean(r)
    return d


def best_circle(these_moves, x0=None, y0=None):
    # Follow the methed described in "Using scipy.optimize.leastsq" of
    # https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    # Clean up the data

    median_pos_t = np.median(these_moves.pos_t)
    these_moves = these_moves.query(f"pos_t == {median_pos_t}")
    these_moves = these_moves.dropna(
        subset=("ptl_y", "ptl_y", "pos_p", "length_r2")
    )

    # Try to guess the center
    x, y, p, r2 = (
        these_moves.ptl_x.values,
        these_moves.ptl_y.values,
        these_moves.pos_p.values,
        these_moves.length_r2.values,
    )

    if x0 is None:
        x0 = np.median(x - r2 * np.cos(np.radians(p)))
        y0 = np.median(y - r2 * np.sin(np.radians(p)))

    def opt_func(center_coords):
        d = compute_radial_distance(x, y, *center_coords)
        return d

    center_estimate = (x0, y0)
    fit_center, ier = scipy.optimize.leastsq(opt_func, center_estimate)
    fit_xc, fit_yc = fit_center
    r2s = compute_radius(x, y, fit_xc, fit_yc)
    fit_r = np.mean(r2s)
    rms_resid = np.sqrt(np.mean((r2s - fit_r) ** 2))
    return fit_xc, fit_yc, fit_r, rms_resid


def plot_positioner(
    petal_id,
    pos_id,
    moves,
    fit_center="expected",
    pos_color=1,
    meas_color=0,
    cmap=plt.get_cmap("Set1"),
    ax=None,
):
    meas_color = (
        cmap(meas_color)
        if isinstance(meas_color, numbers.Number)
        else meas_color
    )
    pos_color = (
        cmap(pos_color) if isinstance(pos_color, numbers.Number) else pos_color
    )

    if ax is None:
        fig, ax = plt.subplots()

    these_moves = moves.loc[(petal_id, pos_id)]
    these_moves = these_moves.dropna(
        subset=(
            "ptl_y",
            "ptl_y",
            "cmd_ptl_x",
            "cmd_ptl_y",
            "pos_p",
            "pos_t",
            "length_r1",
            "length_r2",
        )
    )

    move0 = these_moves.iloc[0]

    meas_axis1_x, meas_axis1_y = pos2ptl.int2ptl(
        move0.calc_t,
        move0.calc_p,
        0,
        0,
        move0.offset_t,
        move0.offset_p,
        move0.offset_x,
        move0.offset_y,
    )
    meas_axis2_x, meas_axis2_y = pos2ptl.int2ptl(
        move0.calc_t,
        move0.calc_p,
        move0.length_r1,
        0,
        move0.offset_t,
        move0.offset_p,
        move0.offset_x,
        move0.offset_y,
    )
    meas_fib_x, meas_fib_y = pos2ptl.int2ptl(
        move0.calc_t,
        move0.calc_p,
        move0.length_r1,
        move0.length_r2,
        move0.offset_t,
        move0.offset_p,
        move0.offset_x,
        move0.offset_y,
    )
    ax.plot(
        [meas_axis1_x, meas_axis2_x, meas_fib_x],
        [meas_axis1_y, meas_axis2_y, meas_fib_y],
        color=meas_color,
    )

    meas_axis2_xs, meas_axis2_ys = pos2ptl.int2ptl(
        these_moves.calc_t,
        these_moves.calc_p,
        these_moves.length_r1,
        0,
        these_moves.offset_t,
        these_moves.offset_p,
        these_moves.offset_x,
        these_moves.offset_y,
    )

    pos_axis1_x, pos_axis1_y = pos2ptl.int2ptl(
        move0.pos_t,
        move0.pos_p,
        0,
        0,
        move0.offset_t,
        move0.offset_p,
        move0.offset_x,
        move0.offset_y,
    )
    pos_axis2_x, pos_axis2_y = pos2ptl.int2ptl(
        move0.pos_t,
        move0.pos_p,
        move0.length_r1,
        0,
        move0.offset_t,
        move0.offset_p,
        move0.offset_x,
        move0.offset_y,
    )
    pos_fib_x, pos_fib_y = pos2ptl.int2ptl(
        move0.pos_t,
        move0.pos_p,
        move0.length_r1,
        move0.length_r2,
        move0.offset_t,
        move0.offset_p,
        move0.offset_x,
        move0.offset_y,
    )
    ax.plot(
        [pos_axis1_x, pos_axis2_x, pos_fib_x],
        [pos_axis1_y, pos_axis2_y, pos_fib_y],
        color=pos_color,
    )
    ax.scatter(
        pos_axis1_x,
        pos_axis1_y,
        edgecolors=pos_color,
        facecolors="none",
        marker="o",
        s=100,
        label="central axis",
    )
    ax.scatter(
        pos_axis2_x,
        pos_axis2_y,
        edgecolors=pos_color,
        facecolors="none",
        marker="o",
        s=30,
        label="cmd axis 2",
    )

    these_moves.plot.scatter(
        "cmd_ptl_x",
        "cmd_ptl_y",
        color=pos_color,
        marker="+",
        s=60,
        ax=ax,
        label="cmd position",
    )

    these_moves.plot.scatter(
        "ptl_x", "ptl_y", color=meas_color, ax=ax, label="meas. position"
    )

    if fit_center == "expected":
        xc, yc, r2, rms_resid = best_circle(
            these_moves, pos_axis1_x, pos_axis1_y
        )
    elif fit_center == "measured":
        xc, yc, r2, s_resid = best_circle(
            these_moves, meas_axis1_x, meas_axis1_y
        )
    elif fit_center == "median":
        xc, yc, r2, rms_resid = best_circle(these_moves)
    else:
        xc, yc, r2, rms_resid = None, None, None, None

    if xc is not None:
        ax.scatter(
            xc,
            yc,
            color=meas_color,
            marker="x",
            s=100,
            label="fit circle center",
        )

    xlim = ax.get_xlim()
    x_width = max(xlim) - min(xlim)
    ylim = ax.get_ylim()
    y_width = max(ylim) - min(ylim)
    if x_width > y_width:
        ax.set_ylim(ylim[0], ylim[0] + x_width)
    else:
        ax.set_xlim(xlim[0], xlim[0] + y_width)

    ax.set_aspect("equal")
    ax.legend()
    return ax


def plot_measured_vs_commanded_move(
    petal_id,
    pos_id,
    moves,
    move_fits,
    coordinate="t",
    fit_color=0,
    data_color=2,
    match_color=1,
    cmap=plt.get_cmap("Set1"),
    ax=None,
):
    fit_color = (
        cmap(fit_color) if isinstance(fit_color, numbers.Number) else fit_color
    )
    data_color = (
        cmap(data_color)
        if isinstance(data_color, numbers.Number)
        else data_color
    )
    match_color = (
        cmap(match_color)
        if isinstance(match_color, numbers.Number)
        else match_color
    )

    if ax is None:
        fig, ax = plt.subplots()

    these_moves = moves.loc[(petal_id, pos_id)]
    these_moves = these_moves.dropna(
        subset=(
            "ptl_y",
            "ptl_y",
            "cmd_ptl_x",
            "cmd_ptl_y",
            "pos_p",
            "pos_t",
            "length_r1",
            "length_r2",
        )
    )

    these_fits = move_fits.loc[(petal_id, pos_id)]
    these_moves.plot.scatter(
        f"move_{coordinate}",
        f"dcalc_{coordinate}",
        s=20,
        color=data_color,
        ax=ax,
    )

    auto_xlim = ax.get_xlim()
    ref_line = np.array(auto_xlim)
    ax.plot(
        ref_line, ref_line, color=match_color, label="commanded = measured"
    )
    this_fit = move_fits.loc[(petal_id, pos_id)]
    ax.plot(
        ref_line,
        this_fit.intercept + ref_line * this_fit.slope,
        color=fit_color,
        alpha=1,
        label="linear fit",
    )
    ax.set_xlim(auto_xlim)
    ax.set_xlabel("commanded move (deg)")
    ax.set_ylabel("measured move (deg)")
    ax.legend()
    return ax


def plot_coord_against_time(
    petal_id,
    pos_id,
    moves,
    meas_coord="calc_t",
    cmd_coord="pos_t",
    pos_color=1,
    meas_color=0,
    cmap=plt.get_cmap("Set1"),
    ax=None,
):
    meas_color = (
        cmap(meas_color)
        if isinstance(meas_color, numbers.Number)
        else meas_color
    )
    pos_color = (
        cmap(pos_color) if isinstance(pos_color, numbers.Number) else pos_color
    )

    if ax is None:
        fig, ax = plt.subplots()

    these_moves = moves.loc[(petal_id, pos_id)]
    these_moves = these_moves.reset_index("pos_move_index", drop=False)

    these_moves.plot(
        "pos_move_index",
        cmd_coord,
        color=pos_color,
        marker="o",
        ax=ax,
        label="commanded",
    )
    these_moves.plot(
        "pos_move_index",
        meas_coord,
        color=meas_color,
        marker="o",
        ax=ax,
        label="measured",
    )
    ax.set_ylabel(cmd_coord)
    return ax


def multi_plot_pos(
    petal_id, pos_id, moves, move_fits, fit_center="expected", **kwargs
):
    fig, axes = plt.subplots(2, 2, **kwargs)

    axes_iter = iter(axes.T.flatten())

    ax = next(axes_iter)
    plot_positioner(
        petal_id, pos_id, moves=moves, fit_center=fit_center, ax=ax
    )

    ax = next(axes_iter)
    plot_measured_vs_commanded_move(petal_id, pos_id, moves, move_fits, ax=ax)

    ax = next(axes_iter)
    plot_coord_against_time(petal_id, pos_id, moves, ax=ax)

    ax = next(axes_iter)
    plot_coord_against_time(
        petal_id, pos_id, moves, meas_coord="calc_t", cmd_coord="pos_t", ax=ax
    )

    plt.tight_layout()
    fig.suptitle(f"Petal {petal_id}, positioner {pos_id}", va="bottom")

    return fig, axes


def plot_move_fit_params_by_mag(moves, coordinate, cuts=[-15, 15]):

    move_fit_kwargs = {
        "exog_col": f"move_{coordinate}",
        "endog_col": f"dcalc_{coordinate}",
        "x_col": "ptl_x",
        "y_col": "ptl_y",
    }

    pos_move_fits = fit_move_lines(
        moves.query(f"move_{coordinate}>{cuts[1]}"), **move_fit_kwargs
    )
    neg_move_fits = fit_move_lines(
        moves.query(f"move_{coordinate}<{cuts[0]}"), **move_fit_kwargs
    )
    small_move_fits = fit_move_lines(
        moves.query(
            f"(move_{coordinate}<{cuts[1]}) and (move_{coordinate}>{cuts[0]})"
        ),
        **move_fit_kwargs,
    )

    fig, axes = plt.subplots(2, figsize=(5, 8))

    colors = plt.get_cmap("Dark2")([2, 0, 1])

    ax = axes[0]
    _ = ax.hist(
        [
            neg_move_fits.slope,
            small_move_fits.slope,
            pos_move_fits.slope,
        ],
        color=colors,
        stacked=True,
        bins=np.arange(0.9, 1.1, 0.001),
        label=["move_t<-15", "|move_t|<=15", "move_t>15"],
    )
    ax.legend()
    ax.set_xlabel("slope (measured move / commanded move)")

    ax = axes[1]
    _ = ax.hist(
        [
            neg_move_fits.intercept,
            small_move_fits.intercept,
            pos_move_fits.intercept,
        ],
        color=colors,
        stacked=True,
        bins=np.arange(-2.5, 2.5, 0.02),
    )
    ax.set_xlabel("intercept (degrees)")

    plt.tight_layout()
    return fig, axes
