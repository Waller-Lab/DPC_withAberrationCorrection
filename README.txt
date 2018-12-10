Matlab code implements a joint estimiation algorithm for complex-field and pupil aberration reconstruction using a DPC dataset.
This requires data with a combination of images under DPC or partially spatially coherent illuminations and spatially coherent illuminations.
Minimum data of 3 DPC images and 1 on-axis spatially coherent image are provided as an example in the code. In each iteration, the code solves
two sub optimization problems, including conventional DPC reconstruction with regularization and pupil aberation recovery. For the pupil aberration recovery,
we adopt the L-BFGS method implemented in the open source optimization package, minFunc.

1. Download the minFunc package (https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html).

2. Run main_dpc_aberration_estimation.m with the example DPC dataset.

Please cite as:
M. Chen, Z. F. Phillips, and L. Waller, Quantitative differential phase contrast (DPC) microscopy with computational aberration correction, Opt. Express 26(25), 32888-32899 (2018).