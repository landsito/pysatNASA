# -*- coding: utf-8 -*-
"""Supports OMNI Combined, Definitive, IMF and Plasma Data, and Energetic
Proton Fluxes, Time-Shifted to the Nose of the Earth's Bow Shock, plus Solar
and Magnetic Indices. Downloads data from the NASA Coordinated Data Analysis
Web (CDAWeb). Supports both 5 and 1 minute files.

Properties
----------
platform
    'omni'
name
    'hro'
tag
    Select time between samples, one of {'1min', '5min'}
inst_id
    None supported

Note
----
Files are stored by the first day of each month. When downloading use
omni.download(start, stop, freq='MS') to only download days that could possibly
have data.  'MS' gives a monthly start frequency.

This material is based upon work supported by the
National Science Foundation under Grant Number 1259508.

Any opinions, findings, and conclusions or recommendations expressed in this
material are those of the author(s) and do not necessarily reflect the views
of the National Science Foundation.

Warnings
--------
- Currently no cleaning routine. Though the CDAWEB description indicates that
  these level-2 products are expected to be ok.
- Module not written by OMNI team.


Custom Functions
----------------
time_shift_to_magnetic_poles
    Shift time from bowshock to intersection with one of the magnetic poles
calculate_clock_angle
    Calculate the clock angle and IMF mag in the YZ plane
calculate_imf_steadiness
    Calculate the IMF steadiness using clock angle and magnitude in the YZ plane
calculate_dayside_reconnection
    Calculate the dayside reconnection rate

"""

import datetime as dt
import functools
import numpy as np
import pandas as pds
import scipy.stats as stats
import warnings

from pysat.instruments.methods import general as mm_gen
from pysat import logger

from pysatNASA.instruments.methods import cdaweb as cdw

# ----------------------------------------------------------------------------
# Instrument attributes

platform = 'omni'
name = 'hro'
tags = {'1min': '1-minute time averaged data',
        '5min': '5-minute time averaged data'}
inst_ids = {'': [tag for tag in tags.keys()]}

# ----------------------------------------------------------------------------
# Instrument test attributes

_test_dates = {'': {'1min': dt.datetime(2009, 1, 1),
                    '5min': dt.datetime(2009, 1, 1)}}

# ----------------------------------------------------------------------------
# Instrument methods


def init(self):
    """Initializes the Instrument object with instrument specific values.

    Runs once upon instantiation.

    """

    ackn_str = ''.join(('For full acknowledgement info, please see: ',
                        'https://omniweb.gsfc.nasa.gov/html/citing.html'))
    self.acknowledgements = ackn_str
    self.references = ' '.join(('J.H. King and N.E. Papitashvili, Solar',
                                'wind spatial scales in and comparisons',
                                'of hourly Wind and ACE plasma and',
                                'magnetic field data, J. Geophys. Res.,',
                                'Vol. 110, No. A2, A02209,',
                                '10.1029/2004JA010649.'))
    logger.info(ackn_str)
    return


def clean(self):
    """ Cleaning function for OMNI data

    Note
    ----
    'clean' - Replace default fill values with NaN
    'dusty' - Same as clean
    'dirty' - Same as clean
    'none'  - Preserve original fill values

    """
    for key in self.data.columns:
        if key != 'Epoch':
            fill = self.meta[key, self.meta.labels.fill_val][0]
            idx, = np.where(self[key] == fill)
            # Set the fill values to NaN
            self[idx, key] = np.nan

            # Replace the old fill value with NaN and add this to the notes
            fill_notes = "".join(["Replaced standard fill value with NaN. ",
                                  "Standard value was: {:}".format(
                                      self.meta[key,
                                                self.meta.labels.fill_val])])
            notes = '\n'.join([str(self.meta[key, self.meta.labels.notes]),
                               fill_notes])
            self.meta[key, self.meta.labels.notes] = notes
            self.meta[key, self.meta.labels.fill_val] = np.nan

    return


# ----------------------------------------------------------------------------
# Instrument functions
#
# Use the default CDAWeb and pysat methods

# Set the list_files routine
fname = ''.join(['omni_hro_{tag:s}_{{year:4d}}{{month:02d}}{{day:02d}}_',
                 'v{{version:02d}}.cdf'])
supported_tags = {inst_id: {tag: fname.format(tag=tag) for tag in tags.keys()}
                  for inst_id in inst_ids.keys()}
list_files = functools.partial(mm_gen.list_files,
                               supported_tags=supported_tags,
                               file_cadence=pds.DateOffset(months=1))

# Set the load routine
load = functools.partial(cdw.load, file_cadence=pds.DateOffset(months=1))

# Set the download routine
remote_dir = '/pub/data/omni/omni_cdaweb/hro_{tag:s}/{{year:4d}}/'
download_tags = {inst_id: {tag: {'remote_dir': remote_dir.format(tag=tag),
                                 'fname': supported_tags[inst_id][tag]}
                           for tag in inst_ids[inst_id]}
                 for inst_id in inst_ids.keys()}
download = functools.partial(cdw.download, supported_tags=download_tags)

# Set the list_remote_files routine
list_remote_files = functools.partial(cdw.list_remote_files,
                                      supported_tags=download_tags)

# ----------------------------------------------------------------------------
# Local functions


def time_shift_to_magnetic_poles(inst):
    """ OMNI data is time-shifted to bow shock. Time shifted again
    to intersections with magnetic pole.

    Parameters
    ----------
    inst : Instrument class object
        Instrument with OMNI HRO data

    Note
    ----
    Time shift calculated using distance to bow shock nose (BSN)
    and velocity of solar wind along x-direction.

    Warnings
    --------
    Use at own risk.

    """

    # need to fill in Vx to get an estimate of what is going on
    inst['Vx'] = inst['Vx'].interpolate('nearest')
    inst['Vx'] = inst['Vx'].fillna(method='backfill')
    inst['Vx'] = inst['Vx'].fillna(method='pad')

    inst['BSN_x'] = inst['BSN_x'].interpolate('nearest')
    inst['BSN_x'] = inst['BSN_x'].fillna(method='backfill')
    inst['BSN_x'] = inst['BSN_x'].fillna(method='pad')

    # make sure there are no gaps larger than a minute
    inst.data = inst.data.resample('1T').interpolate('time')

    time_x = inst['BSN_x'] * 6371.2 / -inst['Vx']
    idx, = np.where(np.isnan(time_x))
    if len(idx) > 0:
        logger.info(time_x[idx])
        logger.info(time_x)
    time_x_offset = [pds.DateOffset(seconds=time)
                     for time in time_x.astype(int)]
    new_index = []
    for i, time in enumerate(time_x_offset):
        new_index.append(inst.data.index[i] + time)
    inst.data.index = new_index
    inst.data = inst.data.sort_index()

    return


def calculate_clock_angle(inst):
    """ Calculate IMF clock angle and magnitude of IMF in GSM Y-Z plane

    Parameters
    -----------
    inst : pysat.Instrument
        Instrument with OMNI HRO data

    """

    # Calculate clock angle in degrees
    clock_angle = np.degrees(np.arctan2(inst['BY_GSM'], inst['BZ_GSM']))
    clock_angle[clock_angle < 0.0] += 360.0
    inst['clock_angle'] = pds.Series(clock_angle, index=inst.data.index)

    # Calculate magnitude of IMF in Y-Z plane
    inst['BYZ_GSM'] = pds.Series(np.sqrt(inst['BY_GSM']**2
                                         + inst['BZ_GSM']**2),
                                 index=inst.data.index)

    return


def calculate_imf_steadiness(inst, steady_window=15, min_window_frac=0.75,
                             max_clock_angle_std=(90.0 / np.pi),
                             max_bmag_cv=0.5):
    """ Calculate IMF steadiness using clock angle standard deviation and
    the coefficient of variation of the IMF magnitude in the GSM Y-Z plane

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument with OMNI HRO data
    steady_window : int
        Window for calculating running statistical moments in min (default=15)
    min_window_frac : float
        Minimum fraction of points in a window for steadiness to be calculated
        (default=0.75)
    max_clock_angle_std : float
        Maximum standard deviation of the clock angle in degrees (default=22.5)
    max_bmag_cv : float
        Maximum coefficient of variation of the IMF magnitude in the GSM
        Y-Z plane (default=0.5)

    """

    # We are not going to interpolate through missing values
    rates = {'': 1, '1min': 1, '5min': 5}
    sample_rate = int(rates[inst.tag])
    max_wnum = np.floor(steady_window / sample_rate)
    if max_wnum != steady_window / sample_rate:
        steady_window = max_wnum * sample_rate
        logger.warning("sample rate is not a factor of the statistical window")
        logger.warning("new statistical window is {:.1f}".format(steady_window))

    min_wnum = int(np.ceil(max_wnum * min_window_frac))

    # Calculate the running coefficient of variation of the BYZ magnitude
    byz_mean = inst['BYZ_GSM'].rolling(min_periods=min_wnum, center=True,
                                       window=steady_window).mean()
    byz_std = inst['BYZ_GSM'].rolling(min_periods=min_wnum, center=True,
                                      window=steady_window).std()
    inst['BYZ_CV'] = pds.Series(byz_std / byz_mean, index=inst.data.index)

    # Calculate the running circular standard deviation of the clock angle
    circ_kwargs = {'high': 360.0, 'low': 0.0, 'nan_policy': 'omit'}
    try:
        ca_std = \
            inst['clock_angle'].rolling(min_periods=min_wnum,
                                        window=steady_window,
                                        center=True).apply(stats.circstd,
                                                           kwargs=circ_kwargs,
                                                           raw=True)
    except TypeError:
        warnings.warn(' '.join(['To automatically remove NaNs from the',
                                'calculation, please upgrade to scipy 1.4 or',
                                'newer']))
        circ_kwargs.pop('nan_policy')
        ca_std = \
            inst['clock_angle'].rolling(min_periods=min_wnum,
                                        window=steady_window,
                                        center=True).apply(stats.circstd,
                                                           kwargs=circ_kwargs,
                                                           raw=True)
    inst['clock_angle_std'] = pds.Series(ca_std, index=inst.data.index)

    # Determine how long the clock angle and IMF magnitude are steady
    imf_steady = np.zeros(shape=inst.data.index.shape)

    steady = False
    for i, cv in enumerate(inst.data['BYZ_CV']):
        if steady:
            del_min = int((inst.data.index[i]
                           - inst.data.index[i - 1]).total_seconds() / 60.0)
            if np.isnan(cv) or np.isnan(ca_std[i]) or del_min > sample_rate:
                # Reset the steadiness flag if fill values are encountered, or
                # if an entry is missing
                steady = False

        if cv <= max_bmag_cv and ca_std[i] <= max_clock_angle_std:
            # Steadiness conditions have been met
            if steady:
                imf_steady[i] = imf_steady[i - 1]

            imf_steady[i] += sample_rate
            steady = True

    inst['IMF_Steady'] = pds.Series(imf_steady, index=inst.data.index)
    return


def calculate_dayside_reconnection(inst):
    """ Calculate the dayside reconnection rate (Milan et al. 2014)

    Parameters
    ----------
    inst : pysat.Instrument
        Instrument with OMNI HRO data, requires BYZ_GSM and clock_angle

    Note
    ----
    recon_day = 3.8 Re (Vx / 4e5 m/s)^1/3 Vx B_yz (sin(theta/2))^9/2

    """

    rearth = 6371008.8
    sin_htheta = np.power(np.sin(np.radians(0.5 * inst['clock_angle'])), 4.5)
    byz = inst['BYZ_GSM'] * 1.0e-9
    vx = inst['flow_speed'] * 1000.0

    recon_day = 3.8 * rearth * vx * byz * sin_htheta * np.power((vx / 4.0e5),
                                                                1.0 / 3.0)
    inst['recon_day'] = pds.Series(recon_day, index=inst.data.index)
    return
