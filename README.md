# pysatNASA: pysat support for NASA instruments
[![PyPI Package latest release](https://img.shields.io/pypi/v/pysatNASA.svg)](https://pypi.python.org/pypi/pysatNASA)
[![Build Status](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)](https://github.com/github/docs/actions/workflows/main.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/pysat/pysatNASA/badge.svg?branch=main)](https://coveralls.io/github/pysat/pysatNASA?branch=main)

[![Documentation Status](https://readthedocs.org/projects/pysatnasa/badge/?version=latest)](https://pysatnasa.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/287387638.svg)](https://zenodo.org/badge/latestdoi/287387638)

# Installation

Currently, the main way to get pysatNASA is through github.

```
git clone https://github.com/pysat/pysatNASA.git
```

Change directories into the repository folder and run the setup.py file.  For
a local install use the "--user" flag after "install".

```
cd pysatNASA/
python setup.py install
```

Note: pre-1.0.0 version
------------------
pysatNASA is currently in an initial development phase.  Much of the API is being built off of pysat 3.0.0 software in order to streamline the usage and test coverage. 

# Using with pysat

The instrument modules are portable and designed to be run like any pysat instrument.

```
import pysat
from pysatNASA.instruments import icon_ivm

ivm = pysat.Instrument(inst_module=icon_ivm, inst_id='a')
```
Another way to use the instruments in an external repository is to register the instruments.  This only needs to be done the first time you load an instrument.  Afterward, pysat will identify them using the `platform` and `name` keywords.

```
import pysat

pysat.utils.registry.register('pysatNASA.instruments.icon_ivm')
ivm = pysat.Instrument('icon', 'ivm', inst_id='a')
```
