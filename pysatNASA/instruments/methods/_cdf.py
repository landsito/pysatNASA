# -*- coding: utf-8 -*-
"""Provides CDF class to parse cdaweb CDF files."""

import datetime as dt
import numpy as np
import pandas as pds
import re

import cdflib
import pysat
from pysat import logger


def convert_ndimensional(data, index=None, columns=None):
    """Convert high-dimensional data to a Dataframe.

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


class CDF(object):
    """Wraps cdflib for loading time series data.

    Loading routines borrow heavily from pyspedas's cdf_to_tplot function

    Parameters
    ----------
    filename : str
        full filename to location of .cdf
    varformat : str
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
        """Initialize the CDF class."""

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
        """Enter runtime context related to this object.

        The with statement will bind this method’s return value
        to the target(s) specified in the as clause of the statement
        """
        return self

    def __exit__(self, type, value, tb):
        """Exit runtime context related to this object.

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
        """Retrieve variable dependency unique to filename.

        Parameter
        ---------
        x_axis_var : str
            name of variable
        """

        return self._dependencies.get(self._filename + x_axis_var)

    def set_dependency(self, x_axis_var, x_axis_data):
        """Set variable dependency unique to filename.

        Parameters
        ----------
        x_axis_var : str
            name of variable
        x_axis_data : array-like
            epoch dependency data
        """

        self._dependencies[self._filename + x_axis_var] = x_axis_data

    def set_epoch(self, x_axis_var):
        """Store epoch dependency.

        Parameters
        ----------
        x_axis_var : str
            name of variable

        """

        data_type_description = self._cdf_file.varinq(
            x_axis_var)['Data_Type_Description']

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

            if ('CDF_TIME' in data_type_description) \
                    or ('CDF_EPOCH' in data_type_description):
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
        """Retrieve index of variable.

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
        """Load cdf variables based on varformat."""

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
        """Export loaded CDF data into data, meta for pysat module.

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
