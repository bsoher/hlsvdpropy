#!/usr/bin/env python
"""
This performs a simple test of the HLSVDPROPY package to see that it's working
and producing sane results. It uses stored SVS FID data as input and pre-
calculated singular values, and frequencies, dampings, amplitudes and phases
data as comparisons.

"""

#
#  (C) Brian J Soher, 2020
#

# Python modules
from __future__ import division, print_function

import os
import cProfile
import pprint
pp=pprint.pprint

# 3rd party modules
import numpy as np

# Our modules
import hlsvdpropy


DTOR = np.pi / 180.0





def test(plot_flag=True):
    """
    This code can be run at the command line to test the internal methods for
    a real MRS SVS data set by running the following command. You may need to
    move it out of the hlsvdpropy install directory for proper functioning:

        $ python simple_test.py

    This should print out results that show that the current hlsvd calculated
    singular values are the same as 'known' values previously calculated and
    stored in this module. It should also plot out for the time domain and
    spectral domain 1) raw data, 2) fitted model and 3) raw - fit plots.

    """
    indat = hlsvdpropy.TESTDATA
    datt = hlsvdpropy.get_testdata()

    dwell = float(indat["step_size"])
    nsv_sought  = indat['n_singular_values']

    reps = 1 if plot_flag else 20

    result1 = []
    result2 = [nsv_sought,indat['sing0'],indat['freq0'],indat['damp0'],indat['ampl0'],indat['phas0']]

    for i in range(reps):
        result1 = hlsvdpropy.hlsvd(datt, nsv_sought, dwell)


    if result1: pp_result_compare(result1, result2, 'hlsvdpropy', 'actual')

    if plot_flag:

        fitts = []
        for indx,res in enumerate([result1, result2]):
    
            if res:
                fitt = hlsvdpropy.create_hlsvd_fids(res, len(datt), dwell, sum_results=True, convert=False)
                fitt[0] *= 0.5
                fitts.append(fitt)
                dat = np.real(np.array(datt))   # only concerned with the real data
                fit = np.real(fitt)
                rmse = np.sqrt(np.sum((dat - fit) ** 2) / len(dat))
                nrmse = rmse / max(dat) - min(dat)
                print(str(indx)+" - RMSE = %.2f, normalized RMSE = %.2f%%" % (rmse, nrmse / 100))

        import matplotlib.pyplot as plt

        datt[0] *= 0.5
        datf = np.fft.fft(_chop(datt))
        labels = ['HLSVDProPy vs Data Time', 'HLSVDProPy vs Data Freq',
                  'Actual Values vs Data Time', 'Actual Values vs Data Freq']

        for i,fit in enumerate(fitts):
            
            fitt = np.array(fit)
            fitf = np.fft.fft(_chop(fitt))

            # The raw data values are always noisier, so plotting them first allows
            # the cleaner estimation to be displayed on top. 

            plt.subplot(3,2,i*2+1)
            plt.plot(datt.real, color='b')
            plt.plot(fitt.real, color='r')
            plt.plot((datt-fitt).real, color='g')
            plt.title(labels[i*2+0])
    
            plt.subplot(3,2,i*2+2)
            plt.plot(datf.real, color='b')
            plt.plot(fitf.real, color='r')
            plt.plot((datf-fitf).real, color='g')
            plt.title(labels[i*2+1])

        if len(fitts) >=2:
            fita, fitb = fitts
            fitt = fita - fitb
            fitaf = np.fft.fft(_chop(fita))
            fitbf = np.fft.fft(_chop(fitb))
            fitf = fitaf - fitbf

            plt.subplot(3,2,5)
            plt.plot(fita.real, color='b')
            plt.plot(fitb.real, color='r')
            plt.plot(fitt.real, color='g')
            plt.title('Both Fits Compared Time')

            plt.subplot(3,2,6)
            plt.plot(fitaf.real, color='b')
            plt.plot(fitbf.real, color='r')
            plt.plot(fitf.real,  color='g')
            plt.title('Both Fits Compared Freq')

        plt.show()


def _chop(data):
    return data * ((((np.arange(len(data)) + 1) % 2) * 2) - 1)


def pp_result_compare(res1, res2, id1, id2):
    
    labl = ['nsv_found', 'singular_values', 'frequencies', 'damping_terms', 'amplitudes', 'phases']

    # NB. The following comparisons come out in different order, thus the need to 
    #     do the np.argsort() gymnastics.
    
    _, _, frequencies1, _, _, _ = res1
    _, _, frequencies2, _, _, _ = res2
    
    indices1 = np.argsort(frequencies1)
    indices2 = np.argsort(frequencies2)
    
    for labl, item1, item2 in zip(labl, res1, res2):
        
        if labl == 'nsv_found':
            print('')
            print(labl, id1+' = '+str(item1), id2+' = '+str(item2))
        elif labl == 'singular_values':
            print('')
            pp(labl+' - '+id1)
            pp(np.array(item1))
            pp(labl+' - '+id2)
            pp(np.array(item2))
            pp(labl+ ' delta'+'('+id1+' - '+id2+')')
            pp(np.array([val1-val2 for val1,val2 in zip(np.array(item1),np.array(item2))]))    
        else:
            print('')


#------------------------------------------------------------------------------


if __name__ == "__main__":

    if os.path.exists("profile.data"):
        os.remove("profile.data")

    cProfile.run('test(plot_flag=False)', 'profile.data')
    import pstats as ps
    p = ps.Stats('profile.data')
    p.strip_dirs().sort_stats('cumulative').print_stats()

    test(plot_flag=True)


