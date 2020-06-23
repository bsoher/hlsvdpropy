HLSVDPROPY
======

Overview - Black box fitting of Time Domain Signals
------

This is a 'pure Python' implementation of the algorithm provided by the Fortran
based HLSVDPRO package, version 2.0.0. 

The HLSVDPROPY package provides code to fit a model function (sum of lorentzians) 
to time-domain data via a 'black box' state space approach (see references below). 
One frequent use for this is by the clinical MRS community for residual water 
removal from MRS signals in the time domain.    

Internally, we use the scipy.linalg SVD libraries for computing the singular 
value decomposition (SVD) of large and sparse matrices. The calculated singular 
values and column vectors are subsequently processed into lists of parameters 
that describe the sum of lorentzians that approximate the data based on the 
paper by Laudadio (see below). Parameters are numpy arrays of: frequencies, 
damping factors, amplitudes, and phases. 


**Example:**

```
import hlsvdpropy
import numpy as np
import matplotlib.pyplot as plt

data = hlsvdpropy.get_testdata()
npts = len(data)
indat = hlsvdpropy.TESTDATA   		# this is a built-in dict with test data 
dwell = float(indat['step_size'])
nsv_sought = indat['n_singular_values']

result = hlsvdpropy.hlsvd(data, nsv_sought, dwell)

nsv_found, singvals, freq, damp, ampl, phas = result

print("np.allclose(freq, indat['freq0']) = ", np.allclose(freq, np.array(indat['freq0'])) )

fid = hlsvdpropy.create_hlsvd_fids(result, npts, dwell, sum_results=True, convert=False)

chop = ((((np.arange(len(fid)) + 1) % 2) * 2) - 1)
dat = data * chop
fit = fid * chop
dat[0] *= 0.5
fit[0] *= 0.5

plt.plot(np.fft.fft(dat).real, color='r') 
plt.plot(np.fft.fft(fit).real, color='b') 
plt.plot(np.fft.fft(dat-fit).real, color='g')
plt.show()

```

HLSVDPROPY Methods
------

- `hlsvdpropy.hlsvdpro(data, nsv_sought, m=None, sparse=True)` - the main method 
for running the hlsvdpro algorithm. It does not require the dwell time of the 
time domain data, but it also does not convert the results to standard units. It
does allow the user to specify the dimensions of the Hankel matrix, and whether
a sparse SVD is performed or not.

- `hlsvdpropy.hlsvd(data, nsv_sought, dwell_time)` - provides backwards 
compatibility to the API for HLSVDPRO version 1.x. It calls the hlsvdpro() method
with default values corresponding to the algorithm used in version 1.x. See 
docstring for more information on the default values used.

HLSVDPROPY Utility Methods
------
- `hlsvdpropy.create_hlsvd_fids(result, npts, dwell, sum_results=False, convert=True)` - 
can be used to create FIDs from the results tuple from either the `hlsvd()` 
or the `hlsvdpro()` methods. It can return either individual FIDs or a sum of 
all FIDs as a result.  

- `hlsvdpropy.convert_hlsvd_result(result, dwell)` - uses the dwell time to 
convert the `hlsvdpro()` result tuple to more standard units. Frequencies 
convert to [kHz], and damping factors to [ms]. Phases convert to [degrees]. 
Singular values, amplitudes and row and column matrices are maintained at 
their same values and output tuple locations. Note - the `hlsvd()` method 
automatically calls this internally, so you don't have to convert values
if you use that method.

- `hlsvdpropy.get_testdata()` - returns a numpy array of 1024 complex data 
points that represents a real world short TE single voxel PRESS data set.
This function converts the base64 encoded string saved in the TESTDATA dict
into a numpy array for you. Additional information about the data and the 
known values for fitting it via the hlsvd() method can be retrieved from 
the TESTDATA dict.  See 'Example' for more usage information.

Technical Overview and References
------

For complete copyright and license information, see the LICENSE file. The 
references are for HLSVDPRO and are provided for completeness.

The state space approach is described in S.Y. Kung, K.S. Arun and D.V. Bhaskar
Rao, J. Opt. Soc. Am. 73, 1799 (1983).

HLSVDPRO version 1.0.x was implemented based on the paper by W.W.F. Pijnappel, 
A. van den Boogaart, R. de Beer, D. van Ormondt, J. Magn. Reson. 97, 122 (1992)
and made use of code from PROPACK version 1.x.

HLSVDPRO version 2.x was adaptated to use PROPACK library version 2.1 to 
implement the HLSVDPRO algorithm as described in T. Laudadio, N. Mastronardi
L. Vanhamme, P. Van Hecke and S. Van Huffel, "Improved Lanczos algorithms for 
blackbox MRS data quantitation", Journal of Magnetic Resonance, Volume 157, 
pages 292-297, 2002. 

