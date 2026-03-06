import numpy as np


def bilinear_channel_estimation(hp_ls):
    """
    Bilinear interpolation of LS channel estimates on a regular pilot grid.

    Pilots sit on even subcarriers (0, 2, …, 118). Pilot OFDM symbols can be
    at any subset of symbol indices (e.g. [2], [2, 3], or [2, 7, 11]).

    1) Linear interpolation across subcarriers (frequency).
    2) Piecewise linear interpolation / extrapolation across symbols (time).
    """
    hp_ls = hp_ls.clone() if hasattr(hp_ls, 'clone') else hp_ls.copy()
    pilot_syms = np.where(hp_ls[0, :].real != 0)[0]

    # Frequency: average even-indexed neighbours to fill odd subcarrier rows
    hp_ls[1:-1:2, pilot_syms] = (hp_ls[:-2:2, pilot_syms] + hp_ls[2::2, pilot_syms]) / 2
    hp_ls[-1, pilot_syms] = hp_ls[-2, pilot_syms]

    # Time: nearest-neighbour or piecewise linear interpolation / extrapolation
    if len(pilot_syms) == 1:
        hp_ls[:] = hp_ls[:, pilot_syms[0]:pilot_syms[0] + 1]
    elif len(pilot_syms) == 2:
        p0, p1 = int(pilot_syms[0]), int(pilot_syms[1])
        slope = (hp_ls[:, p1] - hp_ls[:, p0]) / (p1 - p0)
        for i in range(hp_ls.shape[1]):
            if i not in (p0, p1):
                hp_ls[:, i] = hp_ls[:, p0] + slope * (i - p0)
    elif len(pilot_syms) == 3:
        p0, p1, p2 = int(pilot_syms[0]), int(pilot_syms[1]), int(pilot_syms[2])
        slope_1 = (hp_ls[:, p1] - hp_ls[:, p0]) / (p1 - p0)
        slope_2 = (hp_ls[:, p2] - hp_ls[:, p1]) / (p2 - p1)
        for i in range(hp_ls.shape[1]):
            if i < p0:
                hp_ls[:, i] = hp_ls[:, p0] + slope_1 * (i - p0)
            elif p0 < i < p1:
                hp_ls[:, i] = hp_ls[:, p0] + slope_1 * (i - p0)
            elif p1 < i < p2:
                hp_ls[:, i] = hp_ls[:, p1] + slope_2 * (i - p1)
            elif i > p2:
                hp_ls[:, i] = hp_ls[:, p2] + slope_2 * (i - p2)
    return hp_ls