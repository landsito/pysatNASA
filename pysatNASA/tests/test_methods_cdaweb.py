import datetime as dt
import requests

import pytest

import pysat
import pysatNASA
from pysatNASA.instruments.methods import cdaweb as cdw


class TestCDAWeb():

    def setup(self):
        """Runs before every method to create a clean testing setup."""
        self.download_tags = pysatNASA.instruments.cnofs_plp.download_tags
        self.kwargs = {'tag': None, 'inst_id': None}

    def teardown(self):
        """Runs after every method to clean up previous testing."""
        del self.download_tags, self.kwargs

    def test_remote_file_list_connection_error_append(self):
        """pysat should append suggested help to ConnectionError"""
        with pytest.raises(Exception) as excinfo:
            # Giving a bad remote_site address yields similar ConnectionError
            cdw.list_remote_files(tag='', inst_id='',
                                  supported_tags=self.download_tags,
                                  remote_url='https://bad/path')

        assert excinfo.type is requests.exceptions.ConnectionError
        # Check that pysat appends the message
        assert str(excinfo.value).find('pysat -> Request potentially') > 0

    def test_load_with_empty_file_list(self):
        """pysat should return empty data if no files are requested"""
        data, meta = cdw.load(fnames=[])
        assert len(data) == 0
        assert meta is None

    @pytest.mark.parametrize("bad_key,bad_val,err_msg",
                             [("tag", "badval", "inst_id / tag combo unknown."),
                              ("inst_id", "badval",
                               "inst_id / tag combo unknown.")])
    def test_bad_kwarg_download(self, bad_key, bad_val, err_msg):
        date_array = [dt.datetime(2019, 1, 1)]
        self.kwargs[bad_key] = bad_val
        with pytest.raises(ValueError) as excinfo:
            cdw.download(supported_tags=self.download_tags,
                         date_array=date_array,
                         tag=self.kwargs['tag'],
                         inst_id=self.kwargs['inst_id'])
        assert str(excinfo.value).find(err_msg) >= 0

    @pytest.mark.parametrize("bad_key,bad_val,err_msg",
                             [("tag", "badval", "inst_id / tag combo unknown."),
                              ("inst_id", "badval",
                               "inst_id / tag combo unknown.")])
    def test_bad_kwarg_list_remote_files(self, bad_key, bad_val, err_msg):
        self.kwargs[bad_key] = bad_val
        with pytest.raises(ValueError) as excinfo:
            cdw.list_remote_files(supported_tags=self.download_tags,
                                  tag=self.kwargs['tag'],
                                  inst_id=self.kwargs['inst_id'])
        assert str(excinfo.value).find(err_msg) >= 0

    def test_remote_file_list_all(self):
        """Test that remote_file_list works if start/stop dates unspecified"""
        self.module = pysatNASA.instruments.cnofs_plp
        self.test_inst = pysat.Instrument(inst_module=self.module)
        files = self.test_inst.remote_file_list()
        assert len(files) > 0
