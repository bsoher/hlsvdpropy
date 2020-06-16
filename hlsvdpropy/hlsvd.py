"""This module contains a 'pure Python' implementation of the version 2.x 
HLSVDPRO package. 

The HLSVDPROPY package computes a 'sum of lorentzians' model for the complex 
'signals' data passed in via a 'black box' state space approach. We are using 
the scipy.linalg SVD libraries, rather than the PROPACK Fortran libraries used
in the HLSVDPRO package, but the algorithm is otherwise similarly based on the 
algorithm in:

Laudadio T, et.al. "Improved Lanczos algorithms for blackbox MRS data 
quantitation", Journal of Magnetic Resonance, Volume 157, p.292-297, 2002

Functions:
    hlsvd(data, nsv_sought, dwell_time, sparse=False) -> 6-tuple
    hlsvdpro(data, nsv_sought, m=None, sparse=True) -> 8-tuple
    convert_hlsvd_result(result, dwell)
    create_hlsvd_fids(result, npts, dwell, sum_results=False, convert=True)
    
Example:
    $ python hlvsd.py
    
    Running this module from the site-packages directory will run the internal 
    example. It requires matplotlib be already installed.

"""      

# Python modules
from __future__ import division
import math

# 3rd party modules
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.linalg.lapack as lapack

# Our modules

RADIANS_TO_DEGREES = 180 / 3.1415926



def hlsvd(data, nsv_sought, dwell_time):
    """
    This calls HLSVDPRO version 2.x code, but simulates the hlsvd.hlsvd()
    call from HLSVDPRO version 1.0.x to maintain the API. See doc string
    below for hlsvdpro() method

    Args:
        data (ndarray): an iterable of complex numbers.

        nsv_sought (int): The number of singular values sought. The function
            will return a maximum of this many singular values.

        dwell_time (float): Dwell time in milliseconds.

    Returns:
        tuple: a 6-tuple containing -
            (int), number of singular values found (nsv_found <= nsv_sought)
            (ndarray, floats) the singular values
            (ndarray, floats) the frequencies (in kilohertz)
            (ndarray, floats) the damping factors (in milliseconds?)
            (ndarray, floats) the amplitudes (in arbitrary units)
            (ndarray, floats) the phases (in degrees)

            Each list's length == nsv_found. The five lists are correlated
            (element N of one list is associated with element N in the other
            four lists) and are sorted by singular value with the largest
            (strongest signal) first. The frequencies and damping factors
            HAVE been adjusted by the dwell time, and phases converted to
            degrees for compatibility with HLSVDPRO version 1.x API.

    """
    m = len(data) // 2
    r = hlsvdpro(data, nsv_sought, m=m)
    r = convert_hlsvd_result(r, dwell_time)

    nsv_found, singular_values, frequencies, damping_factors, amplitudes, phases = r[0:6]

    return (nsv_found, singular_values, frequencies, damping_factors, amplitudes, phases)


    
def hlsvdpro(data, nsv_sought, m=None, sparse=True):
    """A pure Python implementation of the HLSVDPRO version 2.x package.
    
    Computes a 'sum of lorentzians' model for the complex 'data' passed in 
    using the scipy.linalg SVD libraries and based on the algorithm in:
      
    Laudadio T, et.al. "Improved Lanczos algorithms for blackbox MRS data 
    quantitation", Journal of Magnetic Resonance, Volume 157, p.292-297, 2002

    Args:
        data (list, ndarray): an iterable of complex numbers.

        nsv_sought (int): The number of singular values sought. The function 
            will return a maximum of this many singular values.

        m (int): (optional) default=len(data)/2, Use to set the size of
            the Hankel matrix used to compute the singular values. Hankel 
            matrix shape is (L+1,L) where L = len(data)-m-1

        sparse (bool): (optional) default True. If set to True, the  
            scipy.sparse.linalg.svds() is used to calculate singular values and 
            nsv_sought is passed in as a parameter. If False, scipy.linalg.svd()
            is used to calculate the singular values, and nsv_sought is used to 
            truncate the results returned.

    Returns:
        tuple: an 8-tuple containing -
            (int), number of singular values found (nsv_found <= nsv_sought)
            (ndarray, floats) the singular values
            (ndarray, floats) the frequencies (in kilohertz)
            (ndarray, floats) the damping factors 
            (ndarray, floats) the amplitudes (in arbitrary units)
            (ndarray, floats) the phases (in radians)
            (ndarray) The top nsv_found left singular vectors, shape=(L+1,3)
            (ndarray) The top nsv_found right singular vectors, shape=(3,L)

            For the first five ndarrays, length == nsv_found and element N of 
            one list is associated with element N in the other four lists. They
            are sorted by singular value with the largest (strongest signal) 
            first.  The frequencies and damping factors have NOT been adjusted 
            by the dwell time, nor the phases converted to degrees.

    """
    mode = 'f' # or 'b'
    xx = data
    k = nsv_sought
    n = len(xx)
    m = int(n/2) if m is None else m
    
    l = n - m - 1

    if mode == "f":
        x = scipy.linalg.hankel(xx[:l + 1], xx[l:])
    else:
        # for backward LP we need to make the hankel matrix:
        #    x_N-1 x_N-2 ... x_N-M-1
        #    x_N-2 x_N-3 ... x_N-M-2
        #      ...
        #    x_M   x_M-1 ... x_0

        x = scipy.linalg.hankel(xx[:m - 1:-1], xx[m::-1])    # SVD of data matrix and truncation of U to form Uk
    
    if sparse:
        u, s, vh = scipy.sparse.linalg.svds(x, k=k)
    else:
        u, s, vh = scipy.linalg.svd(x, full_matrices=False)

    k = min(k,len(s))               # number of singular values found
        
    uk = np.mat(u[:, :k])           # trucated U matrix of rank K
    ub = uk[:-1]                    # Uk with bottom row removed
    ut = uk[1:]                     # Uk with top row removed

    zp, resid, rank, ss = scipy.linalg.lstsq(ub, ut)

    #----------------------------------------------------------------
    # Diagonalize Z' (=hx), yields 'signal'poles' aka 'roots'
    #   Eigenvalues are returned unordered. I sort them here to make
    #   it easier to compare them with output from the Fortran code.

    roots = scipy.linalg.eigvals(zp)
    roots = np.array(sorted(roots, reverse=True))

    #----------------------------------------------------------------
    #  Calculate dampings (damp) and frequencies (freq) from roots

    dampings = np.log(np.abs(roots))
    frequencies = np.arctan2(roots.imag, roots.real) / (math.pi * 2)

    #----------------------------------------------------------------
    #  Calculate complex-valued amplitudes, using the pseudoinverse
    #    of the Lrow*kfit Vandermonde matrix zeta.
    #
    # FIXME this code, like the Fortran, ignores possible errors
    #    reported in the "info" return value.

    zeta = np.vander(roots, N=len(data), increasing=True).T
    v1, x1, s1, rank, _, info = lapack.zgelss(zeta, data)

    #----------------------------------------------------------------
    # Discard uneeded values of x
    x1 = x1[:k]
    amplitudes = np.abs(x1)
    phases = np.arctan2(x1.imag, x1.real)

    return k, s[::-1], frequencies, dampings, amplitudes, phases, u, vh



def hlsvd_v1(data, nsv_sought, dwell_time, sparse=False):
    """Replicates the API for hlsvd.hlsvd() method in HLSVDPRO version 1.x

    See documentation for hlsvdpro() method for more information on algorithm.

    Args:
        data (list, ndarray): an iterable of complex numbers.

        nsv_sought (int): The number of singular values sought. The function 
            will return a maximum of this many singular values.

        dwell_time (float): Dwell time in milliseconds.
        
        sparse (bool): (optional) default False. Flag sent to the hlsvdpro() 
            method. If set to True, the scipy.sparse.linalg.svds() is used to 
            calculate singular values and nsv_sought is passed in as a 
            parameter. If False, scipy.linalg.svd() is used to calculate the 
            singular values, and nsv_sought is used to truncate the results 
            returned.

    Returns:
        tuple: a 6-tuple containing -
            (int), number of singular values found (nsv_found <= nsv_sought)
            (ndarray, floats) the singular values
            (ndarray, floats) the frequencies (in kilohertz)
            (ndarray, floats) the damping factors (in milliseconds?)
            (ndarray, floats) the amplitudes (in arbitrary units)
            (ndarray, floats) the phases (in degrees)

            Each list's length == nsv_found. The five lists are correlated 
            (element N of one list is associated with element N in the other 
            four lists) and are sorted by singular value with the largest 
            (strongest signal) first. The frequencies and damping factors 
            HAVE been adjusted by the dwell time, and phases converted to
            degrees for compatibility with HLSVDPRO version 1.x API.
    
    """
    m = len(data) // 2
    r = hlsvd(data, nsv_sought, m=m, sparse=sparse)
    r = convert_hlsvd_result(r, dwell_time)

    nsv_found, singular_values, frequencies, damping_factors, amplitudes, phases = r[0:6]

    return (nsv_found, singular_values, frequencies, damping_factors, amplitudes, phases)


def convert_hlsvd_result(result, dwell):
    """
    Use dwell time to convert output from hlsvdpro() method to standard units.
    Frequencies convert to [kHz] and damping factors to [ms]. Phases convert
    to [degrees]. Singular values and row and column matrices are maintained
    at their same values and output tuple locations.

    """
    nsv_found, sing_vals, frequencies, damping, ampls, phases = result[0:6]

    damping = np.array([1.0 / df for df in damping])
    damping = np.array([df * dwell for df in damping])
    frequencies = np.array([frequency / dwell for frequency in frequencies])
    phases = np.array([phase * RADIANS_TO_DEGREES for phase in phases])

    return (nsv_found, sing_vals, frequencies, damping, ampls, phases, result[6], result[7])



def create_hlsvd_fids(result, npts, dwell, sum_results=False, convert=True):
    """ 
    This is a convenience function for creating time domain FID arrays from
    the results of the hlsvd() method. You can either send results in 
    directly from hlsvd() (the assumed condition) or run the conversion
    routine convert_hlsvd_results() manually yourself and send them in, but
    in the latter case set convert=False.
    
    See the self-test example at the bottom of this module for how to
    process the fids into spectra and compare in matplotlib.

    Args:
        result (tuple): output from hlsvd() method, an 8-tuple.

        npts (int): The number of points in the created fids.

        dwell (float): Dwell time in milliseconds for created fids.

        sum_results (bool): (optional) default False. If True, all fid lines
            will be summed up to one fid before being returned. If false, a
            (len(result[2]),npts) ndarray is returned

        convert (bool): (optional) default True. If True, the results parameter
            will be run through the convert_hlsvd_results() method before
            being used. If False, it will assume that the user has already
            converted the freq, damp and phase values.

    Returns:
        ndarray: either a (npts,) ndarray (if sum_results=True) or a
            (len(result[2]),npts) ndarray (if sum_results=False) with time
            domain fids.

    """
    if convert: result = convert_hlsvd_result(result, dwell)

    freqs, damps, areas, phase = result[2:6]

    fids = np.zeros((len(freqs), npts), dtype=np.complex)
    t = np.arange(npts) * dwell
    k = 1j * 2 * np.pi

    for i, damp in enumerate(damps):
        if damp:
            # hack for very small exp() values, temp disable numpy's error
            # report, it silently generate NaNs, then change those NaNs to 0.0
            old_settings = np.seterr(all='ignore')
            line = areas[i] * np.exp((t / damp) + k * (freqs[i] * t + phase[i] / 360.0))
            zeros = np.zeros_like(line)
            fids[i,:] = np.where(np.isnan(line), zeros, line)
            np.seterr(**old_settings)
        else:
            fids[i,:] = fids[i,:] * 0

    if sum_results: result = np.sum(fids, axis=0)

    return result


#------------------------------------------------------------------------------
# test and helper functions below

def _example():
    """
    This code can be run at the command line to test the internal methods for
    a real MRS SVS data set by running the following command:

        $ python hlsvd.py

    """
    import io
    import zlib
    import base64

    input = TESTDATA
    dwell = input['step_size']
    data = input['data']
    data = np.load(io.BytesIO(zlib.decompress(base64.b64decode(data))))
    sing0 = np.array(input['sing0'])
    freq0 = np.array(input['freq0'])
    ampl0 = np.array(input['ampl0'])
    damp0 = np.array(input['damp0'])
    phas0 = np.array(input['phas0'])
    k = input['n_singular_values']

    r = hlsvdpro(data, k, m=None, sparse=True)
    c = convert_hlsvd_result(r, dwell)

    nsv_found, sigma, freq, damp, ampl, phas, u, v = c

    udot = abs(np.dot(u.conjugate().T, u))
    vdot = abs(np.dot(v, v.conjugate().T))
    udot[udot < 1e-6] = 0.0
    vdot[vdot < 1e-3] = 0.0

    # print the results -------------------------------------------------------

    np.set_printoptions(suppress=True, precision=6)

    print('Singular Values hlsvdpro  = ', sigma)
    print('Singular Values actual    = ', sing0[:k])
    print('')
    print('SingVal Diffs (hlsvdpro vs actual ) = ', sigma - sing0[:k])
    print('Freqs   Diffs (hlsvdpro vs actual ) = ', freq - freq0[:k])
    print('Damps   Diffs (hlsvdpro vs actual ) = ', damp - damp0[:k])
    print('Ampls   Diffs (hlsvdpro vs actual ) = ', ampl - ampl0[:k])
    print('Phase   Diffs (hlsvdpro vs actual ) = ', phas - phas0[:k])
    print('')
    print('  np.clip(abs(np.dot(u.conj().T, u))) = \n',udot,'\n')
    print('  np.clip(abs(np.dot(v, v.conj().T))) = \n',vdot,'\n')

    # plot the results --------------------------------------------------------

    import matplotlib.pyplot as plt

    lines = create_hlsvd_fids(r, len(data), input['step_size'], sum_results=True)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    phase = np.exp(1j * (np.pi/180.0) * input['phase'])  # empirical value for display
    datt = data.flatten()
    fitt = lines.flatten()
    axes[0].plot(datt.real, color='r')
    axes[0].plot(fitt.real, color='b')
    axes[0].plot((datt-fitt).real, color='g')
    axes[1].plot(np.fft.fft(_chop((datt*phase))).real, color='r')
    axes[1].plot(np.fft.fft(_chop((fitt*phase))).real, color='b')
    axes[1].plot(np.fft.fft(_chop(((datt-fitt)*phase))).real, color='g')
    plt.show()


def _chop(data):
    return data * ((((np.arange(len(data)) + 1) % 2) * 2) - 1)

TESTDATA = {"step_size": 0.256,
            "phase": -115,
            "data": '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',
            "n_singular_values": 20,
            "sing0": [93297.38131152, 27982.48510524, 25885.79287496, 14870.44956187, 13262.2743428, 11315.46381824, 7954.73081632, 7864.72967152, 5709.91492496, 3975.58105714, 3705.77055356, 3540.11927371, 2981.25162343, 2886.42312726, 2630.57463738, 2295.34194136, 2212.52039032, 2209.27627207, 2192.7444973, 2080.21864471],
            "freq0": [ 4.76136514e-06, 3.82767848e-02, 1.86034106e-03,  9.33885257e-04, 4.81741754e-02,  6.14865777e-02, 7.24885016e-02, 9.38660379e-02, -9.97383420e-03, 1.05790402e-01, 1.30972147e-01, 1.41888111e-01, 1.67031668e-01, 1.71169281e-01, 6.48584501e-02, 2.18820360e-01, 1.54509917e-01, 2.45653729e-01, -6.48697872e-01, 9.78783204e-01],
            "ampl0": [5.86887542e+00, 8.68084399e+00, 1.47479303e+03, 2.42805265e+03, 4.17925234e+01, 4.32654342e+00, 3.31532514e+01, 8.66250561e+01, 2.55382599e+02, 8.64488651e+01, 3.69462322e+01, 5.91182740e+00, 1.54992703e+01, 1.26347451e+02, 6.57057401e+02, 1.44725802e+01, 5.07422712e+02, 1.22175799e+02, 1.36463358e+00, 6.16075064e-01],
            "damp0": [-7.75265930e+03, -1.60257296e+02, -6.96852744e+01, -5.19982959e+01, -9.38436859e+01, -1.65632951e+02, -1.32189167e+02, -8.74949438e+01, -1.49498542e+01, -9.07437716e+01, -6.56605647e+01, -2.31128042e+02, -1.11419785e+02, -9.96009334e+01, -6.85079634e+00, -1.03269678e+02, -7.24604519e+00, -1.76640059e+01, -5.74333252e+02, 2.01856151e+03],
            "phas0": [-11.4319313, -179.08898198, -86.2675685, 101.57672159, 135.98672553, -1.85247859, 120.9919392, 126.92047746, 134.64739231, 126.75651104, 114.78938575, 104.07279722, -79.49029511, 102.0643998, 104.37157872, 58.94575542, 149.03438696,  88.51004286, -39.23428509, 6.26519408],
            }


if __name__ == "__main__":
    _example()

