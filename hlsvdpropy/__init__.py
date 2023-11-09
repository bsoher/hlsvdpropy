"""This is a 'pure Python' implementation of the 'hlsvdpro' algorithm found in 
the HLSVDPRO package version 2.x. 

Modules:
    hlsvd.py - contains methods to convert complex signal data into a fitted 
        model of a 'sum of lorentzians'. All methods use the hlsvd() call, but 
        the hlsvd_v1() method allow for users to be compatible with the 
        HLSVDPRO version 1.x API. 

Installation:
    $ pip install hlsvdpropy
    
Dependencies:
    Numpy, Scipy, (optional) matplotlib
    
"""      
try:
    import importlib.metadata
    version_method = 'importlib'
except:
    # 3rd party imports
    import pkg_resources
    version_method = 'pkg_resources'

from importlib.metadata import version

# 3rd party imports
import pkg_resources

# This removes a useless layer of naming indirection.
from hlsvdpropy.hlsvd import *

if version_method == 'importlib':
    __version__ = version('hlsvdpropy')
else:
    __version__ = pkg_resources.get_distribution('hlsvdpropy').version

