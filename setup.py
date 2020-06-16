# Python modules
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil

# 3rd party imports
import setuptools

VERSION = open('VERSION').read().strip()

NAME = "hlsvdpropy"
DESCRIPTION = """This is a 'pure Python' implementation of the HLSVDPRO version 
2.x package. This function fits time domain data to a model function that is a 
set of lorentzian decaying sinusoids using the state space approach. We are using 
the scipy.linalg SVD libraries, rather than the PROPACK Fortran libraries used
in the HLSVDPRO package."""
LONG_DESCRIPTION = \
"""This is a 'pure Python' implementation of the HLSVDPRO version 2.x package. 

The HLSVDPROPY package computes a 'sum of lorentzians' model for the complex 
'signals' data passed in via a 'black box' state space approach. We are using 
the scipy.linalg SVD libraries, rather than the PROPACK Fortran libraries used
in the HLSVDPRO package, but the algorithm is otherwise similarly based on the 
algorithm in:

  Laudadio T, et.al. "Improved Lanczos algorithms for blackbox MRS data 
  quantitation", Journal of Magnetic Resonance, Volume 157, p.292-297, 2002

This sort of algorithm is often used by the clinical MR spectroscopy community  
for residual water removal from MRS signals in the time domain. 

"""

# Note that Python's distutils writes a PKG-INFO file that replaces the author metadata with
# the maintainer metadata. As a result, it's impossible (AFAICT) to get correct author metadata
# to appear on PyPI. https://bugs.python.org/issue16108
AUTHOR = "Brian J. Soher"
AUTHOR_EMAIL = "bsoher ~at~ briansoher ~dot com~"
MAINTAINER = "Brian J. Soher"
MAINTAINER_EMAIL = "bsoher ~at~ briansoher ~dot com~"
URL = "http://scion.duhs.duke.edu/projects/hlsvdpropy"
# http://pypi.python.org/pypi?:action=list_classifiers
CLASSIFIERS = ['Development Status :: 5 - Production/Stable',
               'Intended Audience :: Science/Research',
               "License :: OSI Approved :: BSD License",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: POSIX :: Linux",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               ]
LICENSE = "http://creativecommons.org/licenses/BSD/"
KEYWORDS = "svd, hlsvd, hlsvdpro, time domain, fitting"
PLATFORMS = 'Linux, OS X, Windows, POSIX'

   

setuptools.setup(name=NAME,
                 version=VERSION,
                 packages=["hlsvdpropy"],
                 zip_safe=False,
                 url=URL,
                 author=AUTHOR,
                 author_email=AUTHOR_EMAIL,
                 maintainer=MAINTAINER,
                 maintainer_email=MAINTAINER_EMAIL,
                 classifiers=CLASSIFIERS,
                 license=LICENSE,
                 keywords=KEYWORDS,
                 description=DESCRIPTION,
                 long_description=LONG_DESCRIPTION,
                 platforms=PLATFORMS,
                 # setuptools should be installed along with hlsvdpropy; the latter requires the
                 # former to run. (hlsvdpropy uses setuptools' pkg_resources in __init__.py to
                 # get the package version.) Since hlsvdpropy is distributed as a wheel which can
                 # only be installed by pip, and pip installs setuptools, this 'install_requires'
                 # is probably superfluous and just serves as documentation.
                 install_requires=['setuptools'],
                 )
