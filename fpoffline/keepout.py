# See https://github.com/desihub/fpoffline/blob/main/nb/Keepouts/Keepouts.ipynb for details

from ast import literal_eval as safe_eval

import numpy as np


def expanded_radially(poly, dR):
    x, y = np.array(poly)
    R = np.hypot(x, y)
    ratio = (R + dR) / R
    x *= ratio
    y *= ratio
    return x, y


def expanded_angularly(poly, dA):
    dA = np.deg2rad(dA)
    x, y = poly
    angle = np.arctan2(y, x)
    angle += np.sign(angle) * dA
    R = np.hypot(x, y)
    xnew = R * np.cos(angle)
    ynew = R * np.sin(angle)
    return xnew, ynew


def translated(poly, dx, dy):
    return poly + np.array((dx,dy)).reshape(2, 1)


def expanded_x(poly, left_shift, right_shift):
    x, y = np.array(poly)
    xpos = x > 0
    xneg = x < 0
    x[xpos] += right_shift
    x[xneg] -= left_shift
    return x, y


nominal_T = np.array(
    [[ 0.814,  2.083,  2.613,  4.154,  4.695,  5.094,  5.094,  4.831,
       4.235,  3.432,  2.28 , -1.902, -2.007, -1.139, -0.17 ,  0.814],
     [-3.236, -2.707, -2.665, -2.759, -1.68 , -0.54 ,  0.54 ,  1.291,
       1.933,  2.283,  2.283, -0.935, -2.665, -3.137, -3.332, -3.236]])

nominal_P = np.array(
    [[ 3.967,  3.918,  3.269, -1.172, -1.172,  3.269,  3.918,  3.967],
     [ 0.   ,  1.014,  1.583,  1.037, -1.037, -1.583, -1.014,  0.   ]])


def adjusted_keepouts(dR_T, dA_T, dR_P, dA_P, R1, R2,
                      is_linphi=False, linphi_dA=5, R1_nom=3, R2_nom=3):
    """Get adjusted keepouts given positioner calibration parameters.

    Parameters
    ----------
    dR_T : float
        Radial expansion of theta keepout in mm
    dA_T : float
        Angular expansion of theta keepout in degrees
    dR_P : float
        Radial expansion of phi keepout in mm
    dA_P : float
        Angular expansion of phi keepout in degrees
    R1 : float
        Length of R1 in mm
    R2 : float
        Length of R2 in mm
    is_linphi : bool
        True if this is a recovered linear phi positioner. These are
        treated specially in the petal code, and use keepouts where
        dA_P is always set to at least linphi_dA, independent of
        the calibration value.
    linphi_dA : float
        Minimum angular expansion of phi keepout in degrees
    R1_nom : float
        Nominal length of R1 in mm
    R2_nom : float
        Nominal length of R2 in mm

    Returns
    -------
    tuple (poly_T, poly_P) where poly_T and poly_P are the adjusted keepouts
    for theta and phi respectively. Each is a tuple of x and y coordinates.
    The coordinates are in mm measured in the petal coordinate system, and
    are relative to the corresponding arm's pivot point.
    """
    poly_T = expanded_radially(nominal_T, dR_T)
    poly_T = expanded_angularly(poly_T, dA_T)

    poly_P = expanded_radially(nominal_P, dR_P)
    if is_linphi:
        # Expand phi angularly by at least linphi_dA
        dA_P = max(dA_P, linphi_dA)
    poly_P = expanded_angularly(poly_P, dA_P)

    # true R1 err desired since it is kinematically real
    R1_error = R1 - R1_nom
    # only expand phi (not contract) it, since this is just distance to fiber,
    # and contraction might not represent true mechanical shape
    R2_error = max(R2 - R2_nom, 0)

    poly_P = translated(poly_P, dx=R1_error, dy=0)
    poly_P = expanded_x(poly_P, left_shift=R1_error, right_shift=R2_error)

    return poly_T, poly_P


def adjusted_keepouts_from_calib(calib, pos_id):
    """Get adjusted keepouts from the a dump of the calibration DB for a given positioner.
    The calib dump must be indexed by pos_id, e.g. obtained with

      calib = fpoffline.scripts.endofnight.get_calib(DB)
      calib.set_index('pos_id', verify_integrity=True, inplace=True)

    This should give the same answer as adjusted_keepouts_from_snapshot().
    """
    if pos_id not in calib.index:
        raise ValueError(f"POS_ID {pos_id} not found in calib.index")
    dev = calib.loc[pos_id]

    dR_T = dev.keepout_expansion_theta_radial
    dA_T = dev.keepout_expansion_theta_angular
    dR_P = dev.keepout_expansion_phi_radial
    dA_P = dev.keepout_expansion_phi_angular
    R1 = dev.length_r1
    R2 = dev.length_r2
    is_linphi = dev.zeno_motor_p == True
    return adjusted_keepouts(dR_T, dA_T, dR_P, dA_P, R1, R2, is_linphi)


def adjusted_keepouts_from_snapshot(snap, pos_id):
    """Get adjusted keepouts from a snapshot of the calibration DB for a given positioner.
    To obtain a snapshot, use e.g.

        snap = fpoffline.io.get_snapshot()

    This should give the same answer as adjusted_keepouts_from_calib().
    """
    sel = snap['POS_ID'] == pos_id
    if not np.any(sel):
        raise ValueError(f"POS_ID {pos_id} not found in snapshot")
    dev = snap[sel]
    if len(dev) != 1:
        raise ValueError(f"POS_ID {pos_id} not unique in snapshot")
    dev = dev[0]

    canonical = lambda k: np.round(safe_eval(k), 5)
    return canonical(dev['KEEPOUT_T']), canonical(dev['KEEPOUT_P'])
