DPC Quantitative Phase Microscopy:
Matlab and Python codes implement a Tikhonov deconvolution based phase reconstruction algorithm from multi-axis DPC data. Images should be captured with LED array illumination or other switchable light sources using four halves of the brightfield circle on in each image. The transfer functions are calculated according to the Weak Object Transfer Function and the absorption and phase are solved with a least squares algorithm.

**Run the "main_dpc.m" under matlab_code folder and set the variable "aberration_correction" to "false", or open the "main_dpc.ipynb" jupyter notebook under python_code folder.

DPC with Aberration Correction:
Matlab code implements a joint estimiation algorithm for complex-field and pupil aberration reconstruction using a DPC dataset. This requires data with a combination of images under DPC or partially spatially coherent illuminations and spatially coherent illuminations. Minimum data of 3 DPC images and 1 on-axis spatially coherent image are provided as an example in the code. In each iteration, the code solves two sub optimization problems, including conventional DPC reconstruction with regularization and pupil aberration recovery. For the pupil aberration recovery, we adopt the L-BFGS method implemented in the open source optimization package, minFunc.

1. Download the minFunc package (https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html).
2. Run the "main_dpc.m" under matlab_code folder and set the variable "aberration_correction" to "true".

Please cite as:
[1] L. Tian and L. Waller, Quantitative differential phase contrast imaging in an LED array microscope, Opt. Express 23, 11394-11403 (2015).
[2] Z. F. Phillips, M. Chen, and L. Waller, Single-shot quantitative phase microscopy with color-multiplexed differential phase contrast (cDPC). PLOS ONE 12(2): e0171228 (2017).
[3] M. Chen, Z. F. Phillips, and L. Waller, Quantitative differential phase contrast (DPC) microscopy with computational aberration correction, Opt. Express 26(25), 32888-32899 (2018).
