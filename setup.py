# Python modules
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil

# 3rd party imports
import setuptools

"""setuptools should be installed along with hlsvdpro; the latter requires the
former to run. (hlsvdpro uses setuptools' pkg_resources in __init__.py to
get the package version.) Since hlsvdpro is distributed as a wheel which can
only be installed by pip, and pip installs setuptools, this 'install_requires'
is probably superfluous and just serves as documentation.
"""


with open("README.md", "r") as fh:
    long_description = fh.read()
    
NAME = "hlsvdpropy"
VERSION = open('VERSION').read().strip()

DESCRIPTION = \
"""This is a 'pure Python' implementation of the HLSVDPRO package version 2.0.0. It fits time domain data to a model function that is a set of lorentzian decaying sinusoids using the state space approach."""


# Note that Python's distutils writes a PKG-INFO file that replaces the author 
# metadata with the maintainer metadata. As a result, it's impossible (AFAICT) 
# to get correct author metadata to appear on PyPI.  
# https://bugs.python.org/issue16108

AUTHOR = "Brian J. Soher"
AUTHOR_EMAIL = "bsoher@briansoher.com"
MAINTAINER = "Brian J. Soher"
MAINTAINER_EMAIL = "bsoher@briansoher.com"
URL = "https://github.com/bsoher/hlsvdpropy"

# http://pypi.python.org/pypi?:action=list_classifiers
CLASSIFIERS = ['Development Status :: 5 - Production/Stable',
               'Intended Audience :: Science/Research',
               "License :: OSI Approved :: BSD License",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: 3.7",
               "Programming Language :: Python :: 3.8",
               "Programming Language :: Python :: 3.9",
               "Programming Language :: Python :: 3.10",
               "Programming Language :: Python :: 3.11",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: POSIX :: Linux",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               ]
LICENSE = "https://opensource.org/licenses/BSD-3-Clause"
KEYWORDS = "svd, hlsvd, hlsvdpro, propack, time domain, fitting"
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
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 platforms=PLATFORMS,
                 install_requires=['setuptools'],
                 python_requires='>=3.7',
                 )
