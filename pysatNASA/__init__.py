"""Core library for pysatNASA.

This is a library of `pysat` instrument modules and methods designed to support
NASA instruments and missions archived at the Community Data Analysis Web
portal.

"""

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

from pysatNASA import constellations  # noqa F401
from pysatNASA import instruments  # noqa F401

__version__ = metadata.version('pysatNASA')
