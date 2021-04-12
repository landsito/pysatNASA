# -*- coding: utf-8 -*-
"""Provides default routines for integrating NASA CDAWeb instruments into
pysat. Adding new CDAWeb datasets should only require mininal user
intervention.

"""

import datetime as dt
import numpy as np
import os
import pandas as pds
import re
import requests
import sys

from bs4 import BeautifulSoup
import cdflib

import pysat
from pysat import logger
from pysat.utils import files as futils
from pysat.instruments.methods import general


def convert_ndimensional(data, index=None, columns=None):
    """Converts high-dimensional data to a Dataframe

    Parameters
    ----------
    data : numpy.ndarray or List object
        Data from cdf to be converted
    index : pandas.Index or array-like or NoneType
        Index to use for resulting frame. (default=None)
    columns : pandas.Index or array-like or NoneType
        Column labels to use for resulting frame. (default=None)

    Returns
    -------
    pandas.DataFrame

    """
    if columns is None:
        columns = [range(i) for i in data.shape[1:]]
        columns = pds.MultiIndex.from_product(columns)

    return pds.DataFrame(data.T.reshape(data.shape[0], -1),
                         columns=columns, index=index)


class CDF():
    """cdflib wrapper for loading time series data

    Loading routines borrow heavily from pyspedas's cdf_to_tplot function

    Parameters
    ----------
    filename : string
        full filename to location of .cdf
    varformat : string
        format of variables for regex parsing. (default='*')
    var_types : list of strings
        list of the different variable types in cdf
        (default=['data', 'support_data'])
    center_measurement : bool
        indicates center measurement. (default=False)
    raise_errors : bool
        turn error raising on or off. (default=False)
    regnames : dict or NoneType
        dictionary to map Registration names to kamodo-compatible names
        (default=None)
    datetime : bool
        True if epoch dependency data should be in datetime format
        (default=True)

    """

    def __init__(self, filename,
                 varformat='*',  # Regular expressions
                 var_types=['data', 'support_data'],
                 center_measurement=False,
                 raise_errors=False,
                 regnames=None,
                 datetime=True,
                 **kwargs):
        self._raise_errors = raise_errors
        self._filename = filename
        self._varformat = varformat
        self._var_types = var_types
        self._datetime = datetime
        self._center_measurement = center_measurement

        # Registration names map from file params to kamodo-compatible names
        if regnames is None:
            regnames = {}
        self._regnames = regnames

        self._cdf_file = cdflib.CDF(self._filename)
        self._cdf_info = self._cdf_file.cdf_info()
        self.data = {}
        self.meta = {}
        self._dependencies = {}

        self._variable_names = (self._cdf_info['rVariables']
                                + self._cdf_info['zVariables'])

        self.load_variables()

    def __enter__(self):
        """Context manager protocol
        enters runtime context related to this object.
        The with statement will bind this method’s return value
        to the target(s) specified in the as clause of the statement
        """
        return self

    def __exit__(self, type, value, tb):
        """Context manager protocol
        If any exceptions occur while attempting to execute the block
        of code nested after the with statement, Python will pass
        information about the exception into this method

        Parameters
        ----------
        type : str
            type of exception
        value : str
            value of exception
        tb : traceback
            traceback object generated by exception
        """
        pass

    def get_dependency(self, x_axis_var):
        """Retrieves variable dependency unique to filename

        Parameter
        ---------
        x_axis_var : string
            name of variable
        """

        return self._dependencies.get(self._filename + x_axis_var)

    def set_dependency(self, x_axis_var, x_axis_data):
        """Sets variable dependency unique to filename

        Parameters
        ----------
        x_axis_var : str
            name of variable
        x_axis_data : array-like
            epoch dependency data
        """

        self._dependencies[self._filename + x_axis_var] = x_axis_data

    def set_epoch(self, x_axis_var):
        """Stores epoch dependency

        Parameters
        ----------
        x_axis_var : str
            name of variable

        """

        data_type_description \
            = self._cdf_file.varinq(x_axis_var)['Data_Type_Description']

        center_measurement = self._center_measurement
        cdf_file = self._cdf_file
        if self.get_dependency(x_axis_var) is None:
            delta_plus_var = 0.0
            delta_minus_var = 0.0
            has_plus_minus = [False, False]

            xdata = cdf_file.varget(x_axis_var)
            epoch_var_atts = cdf_file.varattsget(x_axis_var)

            # Check for DELTA_PLUS_VAR/DELTA_MINUS_VAR attributes
            if center_measurement:
                if 'DELTA_PLUS_VAR' in epoch_var_atts:
                    delta_plus_var = cdf_file.varget(
                        epoch_var_atts['DELTA_PLUS_VAR'])
                    delta_plus_var_att = cdf_file.varattsget(
                        epoch_var_atts['DELTA_PLUS_VAR'])
                    has_plus_minus[0] = True

                    # Check if a conversion to seconds is required
                    if 'SI_CONVERSION' in delta_plus_var_att:
                        si_conv = delta_plus_var_att['SI_CONVERSION']
                        delta_plus_var = delta_plus_var.astype(float) \
                            * np.float(si_conv.split('>')[0])
                    elif 'SI_CONV' in delta_plus_var_att:
                        si_conv = delta_plus_var_att['SI_CONV']
                        delta_plus_var = delta_plus_var.astype(float) \
                            * np.float(si_conv.split('>')[0])

                if 'DELTA_MINUS_VAR' in epoch_var_atts:
                    delta_minus_var = cdf_file.varget(
                        epoch_var_atts['DELTA_MINUS_VAR'])
                    delta_minus_var_att = cdf_file.varattsget(
                        epoch_var_atts['DELTA_MINUS_VAR'])
                    has_plus_minus[1] = True

                    # Check if a conversion to seconds is required
                    if 'SI_CONVERSION' in delta_minus_var_att:
                        si_conv = delta_minus_var_att['SI_CONVERSION']
                        delta_minus_var = \
                            delta_minus_var.astype(float) \
                            * np.float(si_conv.split('>')[0])
                    elif 'SI_CONV' in delta_minus_var_att:
                        si_conv = delta_minus_var_att['SI_CONV']
                        delta_minus_var = \
                            delta_minus_var.astype(float) \
                            * np.float(si_conv.split('>')[0])

            if ('CDF_TIME' in data_type_description) or \
                    ('CDF_EPOCH' in data_type_description):
                if self._datetime:
                    # Convert xdata to datetime
                    try:
                        new_xdata = cdflib.cdfepoch.to_datetime(xdata)
                    except TypeError as terr:
                        estr = ("Invalid data file(s). Please contact CDAWeb "
                                "for assistance: {:}".format(str(terr)))
                        logger.warning(estr)
                        new_xdata = []

                    # Add delta to time, if both plus and minus are defined
                    if np.all(has_plus_minus):
                        # This defines delta_time in seconds supplied
                        delta_time = np.asarray((delta_plus_var
                                                 - delta_minus_var) / 2.0)

                        # delta_time may be a single value or an array
                        xdata = [xx + dt.timedelta(seconds=int(delta_time))
                                 if delta_time.shape == ()
                                 else xx + dt.timedelta(seconds=delta_time[i])
                                 for i, xx in enumerate(new_xdata)]
                    else:
                        xdata = new_xdata

                self.set_dependency(x_axis_var, xdata)

    def get_index(self, variable_name):
        """Return index of variable

        Parameters
        ----------
        variable_name : str
            String to access desired data variable

        Returns
        -------
        index_ : pandas.DataFrame or pandas.MultiIndex
            Dependency data used to index and select desired data

        """

        var_atts = self._cdf_file.varattsget(variable_name)

        if "DEPEND_TIME" in var_atts:
            x_axis_var = var_atts["DEPEND_TIME"]
            self.set_epoch(x_axis_var)
        elif "DEPEND_0" in var_atts:
            x_axis_var = var_atts["DEPEND_0"]
            self.set_epoch(x_axis_var)

        dependencies = []
        for suffix in ['TIME'] + list('0123'):
            dependency = "DEPEND_{}".format(suffix)
            dependency_name = var_atts.get(dependency)
            if dependency_name is not None:
                dependency_data = self.get_dependency(dependency_name)
                if dependency_data is None:
                    dependency_data = self._cdf_file.varget(dependency_name)
                    # Get first unique row
                    dependency_data = pds.DataFrame(dependency_data)
                    dependency_data = dependency_data.drop_duplicates()
                    self.set_dependency(dependency_name,
                                        dependency_data.values[0])
                dependencies.append(dependency_data)

        index_ = None
        if len(dependencies) == 0:
            pass
        elif len(dependencies) == 1:
            index_ = dependencies[0]
        else:
            index_ = pds.MultiIndex.from_product(dependencies)

        return index_

    def load_variables(self):
        """Loads cdf variables based on varformat
        """
        varformat = self._varformat
        if varformat is None:
            varformat = ".*"

        varformat = varformat.replace("*", ".*")
        var_regex = re.compile(varformat)

        for variable_name in self._variable_names:
            if not re.match(var_regex, variable_name):
                # Skip this variable
                continue
            var_atts = self._cdf_file.varattsget(variable_name, to_np=True)
            for k in var_atts:
                var_atts[k] = var_atts[k]  # [0]

            if 'VAR_TYPE' not in var_atts:
                continue

            if var_atts['VAR_TYPE'] not in self._var_types:
                continue

            var_properties = self._cdf_file.varinq(variable_name)

            try:
                ydata = self._cdf_file.varget(variable_name)
            except (TypeError):
                continue

            if ydata is None:
                continue

            if "FILLVAL" in var_atts:
                if (var_properties['Data_Type_Description'] == 'CDF_FLOAT'
                    or var_properties['Data_Type_Description']
                    == 'CDF_REAL4'
                    or var_properties['Data_Type_Description']
                    == 'CDF_DOUBLE'
                    or var_properties['Data_Type_Description']
                        == 'CDF_REAL8'):

                    if ydata[ydata == var_atts["FILLVAL"]].size != 0:
                        ydata[ydata == var_atts["FILLVAL"]] = np.nan

            index = self.get_index(variable_name)

            try:
                if isinstance(index, pds.MultiIndex):
                    self.data[variable_name] = pds.DataFrame(ydata.ravel(),
                                                             index=index)
                else:
                    if len(ydata.shape) == 1:
                        self.data[variable_name] = pds.Series(ydata,
                                                              index=index)
                    elif len(ydata.shape) == 2:
                        self.data[variable_name] = pds.DataFrame(ydata,
                                                                 index=index)
                    elif len(ydata.shape) > 2:
                        tmp_var = convert_ndimensional(ydata, index=index)
                        self.data[variable_name] = tmp_var
                    else:
                        raise NotImplementedError('Cannot handle {} with shape'
                                                  ' {}'.format(variable_name,
                                                               ydata.shape))
            except (ValueError, NotImplementedError):
                self.data[variable_name] = {'ydata': ydata, 'index': index}
                if self._raise_errors:
                    raise

            self.meta[variable_name] = var_atts

    def to_pysat(self, flatten_twod=True,
                 labels={'units': ('Units', str), 'name': ('Long_Name', str),
                         'notes': ('Var_Notes', str), 'desc': ('CatDesc', str),
                         'min_val': ('ValidMin', float),
                         'max_val': ('ValidMax', float),
                         'fill_val': ('FillVal', float)}):
        """
        Exports loaded CDF data into data, meta for pysat module

        Parameters
        ----------
        flatten_twod : bool
            If True, then two dimensional data is flattened across
            columns. Name mangling is used to group data, first column
            is 'name', last column is 'name_end'. In between numbers are
            appended 'name_1', 'name_2', etc. All data for a given 2D array
            may be accessed via, data.ix[:,'item':'item_end']
            If False, then 2D data is stored as a series of DataFrames,
            indexed by Epoch. data.ix[0, 'item']  (default=True)

        labels : dict
            Dict where keys are the label attribute names and the values
            are tuples that have the label values and value types in
            that order.
            (default={'units': ('units', str), 'name': ('long_name', str),
                      'notes': ('notes', str), 'desc': ('desc', str),
                      'min_val': ('value_min', float),
                      'max_val': ('value_max', float)
                      'fill_val': ('fill', float)})

        Returns
        -------
        pandas.DataFrame, pysat.Meta
            Data and Metadata suitable for attachment to a pysat.Instrument
            object.

        Note
        ----
        The *_labels should be set to the values in the file, if present.
        Note that once the meta object returned from this function is attached
        to a pysat.Instrument object then the *_labels on the Instrument
        are assigned to the newly attached Meta object.

        The pysat Meta object will use data with labels that match the patterns
        in *_labels even if the case does not match.

        """
        # Create pysat.Meta object using data above
        # and utilizing the attribute labels provided by the user
        meta = pysat.Meta(pds.DataFrame.from_dict(self.meta, orient='index'),
                          labels=labels)

        cdata = self.data.copy()
        lower_names = [name.lower() for name in meta.keys()]
        for name, true_name in zip(lower_names, meta.keys()):
            if name == 'epoch':
                meta.data.rename(index={true_name: 'epoch'}, inplace=True)
                epoch = cdata.pop(true_name)
                cdata['Epoch'] = epoch

        data = dict()
        index = None
        for varname, df in cdata.items():
            if varname not in ('Epoch', 'DATE'):
                if type(df) == pds.Series:
                    data[varname] = df

                    # CDF data Series are saved using a mix of Range and
                    # Datetime Indexes. This requires that the user specify
                    # the desired index when creating a DataFrame
                    if type(df.index) == pds.DatetimeIndex and index is None:
                        index = df.index

        if index is None:
            raise ValueError(''.join(['cdflib did not load a DatetimeIndex, ',
                                      'not pysat compatible']))

        try:
            data = pds.DataFrame(data, index=index)
        except pds.core.indexes.base.InvalidIndexError as ierr:
            estr = "Invalid times in data file(s): {:}".format(str(ierr))
            logger.warning(estr)
            data = pds.DataFrame(None)

        return data, meta


def load(fnames, tag=None, inst_id=None, file_cadence=dt.timedelta(days=1),
         flatten_twod=True):
    """Load NASA CDAWeb CDF files.

    Parameters
    ----------
    fnames : pandas.Series
        Series of filenames
    tag : str or NoneType
        tag or None (default=None)
    inst_id : str or NoneType
        satellite id or None (default=None)
    file_cadence : dt.timedelta or pds.DateOffset
        pysat assumes a daily file cadence, but some instrument data files
        contain longer periods of time.  This parameter allows the specification
        of regular file cadences greater than or equal to a day (e.g., weekly,
        monthly, or yearly). (default=dt.timedelta(days=1))
    flatted_twod : bool
        Flattens 2D data into different columns of root DataFrame rather
        than produce a Series of DataFrames. (default=True)

    Returns
    -------
    data : pandas.DataFrame
        Object containing satellite data
    meta : pysat.Meta
        Object containing metadata such as column names and units

    Note
    ----
    This routine is intended to be used by pysat instrument modules supporting
    a particular NASA CDAWeb dataset.

    Examples
    --------
    ::

        # within the new instrument module, at the top level define
        # a new variable named load, and set it equal to this load method
        # code below taken from cnofs_ivm.py.

        # support load routine
        # use the default CDAWeb method
        load = cdw.load

    """

    # Load data from any files provided
    if len(fnames) <= 0:
        return pds.DataFrame(None), None
    else:
        # Using cdflib wrapper to load the CDF and format data and
        # metadata for pysat using some assumptions. Depending upon your needs
        # the resulting pandas DataFrame may need modification
        ldata = []
        for lfname in fnames:
            if not general.is_daily_file_cadence(file_cadence):
                # Parse out date from filename
                fname = lfname[0:-11]
                date = dt.datetime.strptime(lfname[-10:], '%Y-%m-%d')

                with CDF(fname) as cdf:
                    # Convert data to pysat format
                    try:
                        temp_data, meta = cdf.to_pysat(
                            flatten_twod=flatten_twod)

                        # Select data from multi-day down to daily
                        temp_data = temp_data.loc[
                            date:date + dt.timedelta(days=1)
                            - dt.timedelta(microseconds=1), :]
                        ldata.append(temp_data)
                    except ValueError as verr:
                        logger.warn("unable to load {:}: {:}".format(fname,
                                                                     str(verr)))
            else:
                # Basic data return
                with CDF(lfname) as cdf:
                    try:
                        temp_data, meta = cdf.to_pysat(
                            flatten_twod=flatten_twod)
                        ldata.append(temp_data)
                    except ValueError as verr:
                        logger.warn("unable to load {:}: {:}".format(lfname,
                                                                     str(verr)))

        # Combine individual files together
        if len(ldata) > 0:
            data = pds.concat(ldata)

        return data, meta


def download(date_array, tag=None, inst_id=None, supported_tags=None,
             remote_url='https://cdaweb.gsfc.nasa.gov', data_path=None):
    """Routine to download NASA CDAWeb CDF data.

    This routine is intended to be used by pysat instrument modules supporting
    a particular NASA CDAWeb dataset.

    Parameters
    ----------
    date_array : array_like
        Array of datetimes to download data for. Provided by pysat.
    tag : str or NoneType
        tag or None (default=None)
    inst_id : str or NoneType
        satellite id or None (default=None)
    supported_tags : dict
        dict of dicts. Keys are supported tag names for download. Value is
        a dict with 'remote_dir', 'fname'. Inteded to be pre-set with
        functools.partial then assigned to new instrument code.
        (default=None)
    remote_url : string or NoneType
        Remote site to download data from
        (default='https://cdaweb.gsfc.nasa.gov')
    data_path : string or NoneType
        Path to data directory.  If None is specified, the value previously
        set in Instrument.files.data_path is used.  (default=None)

    Examples
    --------
    ::

        # download support added to cnofs_vefi.py using code below
        fn = 'cnofs_vefi_bfield_1sec_{year:4d}{month:02d}{day:02d}_v05.cdf'
        dc_b_tag = {'remote_dir': ''.join(('/pub/data/cnofs/vefi/bfield_1sec',
                                            '/{year:4d}/')),
                    'fname': fn}
        supported_tags = {'dc_b': dc_b_tag}

        download = functools.partial(nasa_cdaweb.download,
                                     supported_tags=supported_tags)

    """

    if tag is None:
        tag = ''
    if inst_id is None:
        inst_id = ''
    try:
        inst_dict = supported_tags[inst_id][tag]
    except KeyError:
        raise ValueError('inst_id / tag combo unknown.')

    # Naming scheme for files on the CDAWeb server
    remote_dir = inst_dict['remote_dir']

    # Get list of files from server
    remote_files = list_remote_files(tag=tag, inst_id=inst_id,
                                     remote_url=remote_url,
                                     supported_tags=supported_tags,
                                     start=date_array[0],
                                     stop=date_array[-1])
    # Download only requested files that exist remotely
    for date, fname in remote_files.iteritems():
        # Format files for specific dates and download location
        formatted_remote_dir = remote_dir.format(year=date.year,
                                                 month=date.month,
                                                 day=date.day,
                                                 hour=date.hour,
                                                 min=date.minute,
                                                 sec=date.second)
        saved_local_fname = os.path.join(data_path, fname)

        # Perform download
        try:
            logger.info(' '.join(('Attempting to download file for',
                                  date.strftime('%d %B %Y'))))
            sys.stdout.flush()
            remote_path = '/'.join((remote_url.strip('/'),
                                    formatted_remote_dir.strip('/'),
                                    fname))
            req = requests.get(remote_path)
            if req.status_code != 404:
                with open(saved_local_fname, 'wb') as open_f:
                    open_f.write(req.content)
                logger.info('Finished.')
            else:
                logger.info(' '.join(('File not available for',
                                      date.strftime('%d %B %Y'))))
        except requests.exceptions.RequestException as exception:
            logger.info(' '.join((exception, '- File not available for',
                                  date.strftime('%d %B %Y'))))
    return


def list_remote_files(tag=None, inst_id=None, start=None, stop=None,
                      remote_url='https://cdaweb.gsfc.nasa.gov',
                      supported_tags=None, two_digit_year_break=None,
                      delimiter=None):
    """Return a Pandas Series of every file for chosen remote data.

    This routine is intended to be used by pysat instrument modules supporting
    a particular NASA CDAWeb dataset.

    Parameters
    ----------
    tag : string or NoneType
        Denotes type of file to load.  Accepted types are <tag strings>.
        (default=None)
    inst_id : string or NoneType
        Specifies the satellite ID for a constellation.
        (default=None)
    start : dt.datetime or NoneType
        Starting time for file list. A None value will start with the first
        file found.
        (default=None)
    stop : dt.datetime or NoneType
        Ending time for the file list.  A None value will stop with the last
        file found.
        (default=None)
    remote_url : string or NoneType
        Remote site to download data from
        (default='https://cdaweb.gsfc.nasa.gov')
    supported_tags : dict
        dict of dicts. Keys are supported tag names for download. Value is
        a dict with 'remote_dir', 'fname'. Inteded to be
        pre-set with functools.partial then assigned to new instrument code.
        (default=None)
    two_digit_year_break : int or NoneType
        If filenames only store two digits for the year, then
        '1900' will be added for years >= two_digit_year_break
        and '2000' will be added for years < two_digit_year_break.
        (default=None)
    delimiter : string or NoneType
        If filename is delimited, then provide delimiter alone e.g. '_'
        (default=None)

    Returns
    -------
    pysat.Files.from_os : (pysat._files.Files)
        A class containing the verified available files

    Examples
    --------
    ::

        fname = 'cnofs_vefi_bfield_1sec_{year:04d}{month:02d}{day:02d}_v05.cdf'
        supported_tags = {'dc_b': fname}
        list_remote_files = \
            functools.partial(nasa_cdaweb.list_remote_files,
                              supported_tags=supported_tags)

        fname = 'cnofs_cindi_ivm_500ms_{year:4d}{month:02d}{day:02d}_v01.cdf'
        supported_tags = {'': fname}
        list_remote_files = \
            functools.partial(cdw.list_remote_files,
                              supported_tags=supported_tags)

    """

    if tag is None:
        tag = ''
    if inst_id is None:
        inst_id = ''
    try:
        inst_dict = supported_tags[inst_id][tag]
    except KeyError:
        raise ValueError('inst_id / tag combo unknown.')

    # Naming scheme for files on the CDAWeb server
    format_str = '/'.join((inst_dict['remote_dir'].strip('/'),
                           inst_dict['fname']))

    # Break string format into path and filename
    dir_split = os.path.split(format_str)

    # Parse the path to find the number of levels to search
    format_dir = dir_split[0]
    search_dir = futils.construct_searchstring_from_format(format_dir)
    n_layers = len(search_dir['keys'])

    # Only keep file portion of format
    format_str = dir_split[-1]
    # Generate list of targets to identify files
    search_dict = futils.construct_searchstring_from_format(format_str)
    targets = [x.strip('?') for x in search_dict['string_blocks'] if len(x) > 0]

    # Remove any additional '?' characters that the user may have supplied
    new_targets = []
    for target in targets:
        tstrs = target.split('?')
        for tstr in tstrs:
            if tstr != '':
                new_targets.append(tstr)
    targets = new_targets

    remote_dirs = []
    for level in range(n_layers + 1):
        remote_dirs.append([])
    remote_dirs[0] = ['']

    # Build a list of files using each filename target as a goal
    full_files = []

    if start is None and stop is None:
        # Use the topmost directory without variables
        url_list = ['/'.join((remote_url,
                              search_dir['string_blocks'][0]))]
    elif start is not None:
        stop = dt.datetime.now() if (stop is None) else stop

        if 'year' in search_dir['keys']:
            if 'month' in search_dir['keys']:
                search_times = pds.date_range(start,
                                              stop + pds.DateOffset(months=1),
                                              freq='M')
            else:
                search_times = pds.date_range(start,
                                              stop + pds.DateOffset(years=1),
                                              freq='Y')
            url_list = []
            for time in search_times:
                subdir = format_dir.format(year=time.year, month=time.month)
                url_list.append('/'.join((remote_url, subdir)))
    try:
        for top_url in url_list:
            for level in range(n_layers + 1):
                for directory in remote_dirs[level]:
                    temp_url = '/'.join((top_url.strip('/'), directory))
                    soup = BeautifulSoup(requests.get(temp_url).content,
                                         "lxml")
                    links = soup.find_all('a', href=True)
                    for link in links:
                        # If there is room to go down, look for directories
                        if link['href'].count('/') == 1:
                            remote_dirs[level + 1].append(link['href'])
                        else:
                            # If at the endpoint, add matching files to list
                            add_file = True
                            for target in targets:
                                if link['href'].count(target) == 0:
                                    add_file = False
                            if add_file:
                                full_files.append(link['href'])
    except requests.exceptions.ConnectionError as merr:
        raise type(merr)(' '.join((str(merr), 'pysat -> Request potentially',
                                   'exceeds the server limit. Please try',
                                   'again using a smaller data range.')))

    # Parse remote filenames to get date information
    if delimiter is None:
        stored = futils.parse_fixed_width_filenames(full_files, format_str)
    else:
        stored = futils.parse_delimited_filenames(full_files, format_str,
                                                  delimiter)
    # Process the parsed filenames and return a properly formatted Series
    stored_list = futils.process_parsed_filenames(stored, two_digit_year_break)

    # Downselect to user-specified dates, if needed
    if start is not None:
        mask = (stored_list.index >= start)
        if stop is not None:
            stop_point = (stop + pds.DateOffset(days=1)
                          - pds.DateOffset(microseconds=1))
            mask = mask & (stored_list.index <= stop_point)
        stored_list = stored_list[mask]

    return stored_list
