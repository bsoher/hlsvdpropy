
This is a 'pure Python' implementation of the algorithm provided by the 
HLSVDPRO version 2.x package. 

The HLSVDPROPY package computes a 'sum of lorentzians' model for the complex 
'signals' data passed in via a 'black box' state space approach. We are using 
the scipy.linalg SVD libraries, rather than the PROPACK Fortran libraries used
in the HLSVDPRO package, but the algorithm is otherwise similarly based on the 
algorithm in:

      Laudadio T, et.al. "Improved Lanczos algorithms for blackbox MRS data 
      quantitation", Journal of Magnetic Resonance, Volume 157, p.292-297, 2002

This sort of algorithm is most often used by the clinical MRS community for 
residual water removal from MRS signals in the time domain. 

For complete copyright and license information, see the LICENSE file.


=== Technical Overview - copied from HLSVDPRO package ===

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

