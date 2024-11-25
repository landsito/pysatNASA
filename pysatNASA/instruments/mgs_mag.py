#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# TODO(#XXX): Update NRL review before next version release
# ---------------------------------------------------------
"""Module for the MGS mag instrument.

Supports the Magnetometer (MAG) onboard the Mars Global
Surveyor (MGS) satellite.
Accesses local data in netCDF format.
Downloads from CDAWeb.

Properties
----------
platform
    'mgs'
name
    'mag'
tag
    None supported
inst_id
    ['low', 'high']

Warnings
--------

- None.


Examples
--------
::

    import pysat

    mag = pysat.Instrument('mgs', 'mag', inst_id='low')
    mag.download(dt.datetime(1998, 1, 1), dt.datetime(1998, 1, 31))
    mag.load(1998, 9)

"""

import datetime as dt
import functools

from pysat.instruments.methods import general as mm_gen
from pysat.utils.io import load_netcdf
from pysatNASA.instruments.methods import cdaweb as cdw
from pysatNASA.instruments.methods import general as mm_nasa
from pysatNASA.instruments.methods import mgs as mm_mgs

# ----------------------------------------------------------------------------
# Instrument attributes

platform = 'mgs'
name = 'mag'
tags = {'': 'Level 2 magnetometer data'}
inst_ids = {'high': [''],
            'low': ['']}

pandas_format = True
# ----------------------------------------------------------------------------
# Instrument test attributes

_test_dates = {iid: {'': dt.datetime(1998, 1, 9)} for iid in inst_ids.keys()}

# ----------------------------------------------------------------------------
# Instrument methods

# Use standard init routine
init = functools.partial(mm_nasa.init, module=mm_mgs, name=name)


# Use default clean
clean = mm_nasa.clean


# ----------------------------------------------------------------------------
# Instrument functions
#
# Use the CDAWeb and pysat methods

# Set the list_files routine
datestr = '{year:04d}{month:02d}{day:02d}_v{version:02d}'
fname1 = 'mgs_mag_high_{year:04d}{month:02d}{day:02d}_v{version:02d}.nc'
fname2 = 'mgs_mag_low_{year:04d}{month:02d}{day:02d}_v{version:02d}.nc'
supported_tags = {'high': {'': fname1},
                  'low': {'': fname2}}
list_files = functools.partial(mm_gen.list_files,
                               supported_tags=supported_tags)
# Set the download routine
download_tags = {'low': {'': 'MGS_MAG_LOW'},
                 'high': {'': 'MGS_MAG_HIGH'}}
download = functools.partial(cdw.cdas_download, supported_tags=download_tags)

# Set the list_remote_files routine
list_remote_files = functools.partial(cdw.cdas_list_remote_files,
                                      supported_tags=download_tags)

# Set the load routine


def load(fnames, tag=None, inst_id=None):
    """Load MGS MAG data into `pandas.DataFrame` and `pysat.Meta` objects.

    This routine is called as needed by pysat. It is not intended
    for direct user interaction.

    Parameters
    ----------
    fnames : array-like
        iterable of filename strings, full path, to data files to be loaded.
        This input is nominally provided by pysat itself.
    tag : str
        tag name used to identify particular data set to be loaded.
        This input is nominally provided by pysat itself.
    inst_id : str
        Satellite ID used to identify particular data set to be loaded.
        This input is nominally provided by pysat itself.

    Returns
    -------
    data : pds.DataFrame
        A pandas DataFrame with data prepared for the pysat.Instrument
    meta : pysat.Meta
        Metadata formatted for a pysat.Instrument object.

    Note
    ----
    Any additional keyword arguments passed to pysat.Instrument
    upon instantiation are passed along to this routine.

    Examples
    --------
    ::

        inst = pysat.Instrument('reach', 'dosimeter', inst_id='101', tag='')
        inst.load(2020, 1)

    """

    # Use standard netcdf interface
    labels = {'units': ('UNITS', str), 'name': ('LONG_NAME', str),
              'notes': ('VAR_NOTES', str), 'desc': ('CATDESC', str),
              'min_val': ('VALIDMIN', (int, float)),
              'max_val': ('VALIDMAX', (int, float)),
              'fill_val': ('_FillValue', (int, float))}
    data, meta = load_netcdf(fnames, epoch_name='unix_time',
                             epoch_unit='s',
                             meta_kwargs={'labels': labels})
    return data, meta
