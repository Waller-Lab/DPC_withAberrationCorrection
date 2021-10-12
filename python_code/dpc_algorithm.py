"""
BSD 3-Clause License

Copyright (c) 2018, Waller Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from scipy.ndimage import uniform_filter
pi    = np.pi
naxis = np.newaxis
F     = lambda x: np.fft.fft2(x)
IF    = lambda x: np.fft.ifft2(x)

def pupilGen(fxlin, fylin, wavelength, na, na_in=0.0):
    '''
    pupilGen create a circular pupil function in Fourier space.
    Inputs:
            fxlin     : 1D spatial frequency coordinate in horizontal direction
            fylin     : 1D spatial frequency coordinate in vertical direction
            wavelength: wavelength of incident light
            na        : numerical aperture of the imaging system
            na_in     : put a non-zero number smaller than na to generate an annular function
    Output:
            pupil     : pupil function
    '''
    pupil = np.array(fxlin[naxis, :]**2+fylin[:, naxis]**2 <= (na/wavelength)**2)
    if na_in != 0.0:
        pupil[fxlin[naxis, :]**2+fylin[:, naxis]**2 < (na_in/wavelength)**2] = 0.0
    return pupil

def _genGrid(size, dx):
    '''
    _genGrid create a 1D coordinate vector.
    Inputs:
            size : length of the coordinate vector
            dx   : step size of the 1D coordinate
    Output:
            grid : 1D coordinate vector
    '''
    xlin = np.arange(size, dtype='complex64')
    grid = (xlin-size//2)*dx
    return grid

def _softThreshold(x, threshold):
    '''
    _softThreshold solves a LASSO problem with isotropic total variation.
    Inputs:
            x           : input of the LASSO problem
            threshold   : a threshold value for soft-thresholding
    Output:
            x_threshold : optimized soft-thresholded result
    '''
    magnitude   = ((x.conj()*x).sum(axis=0))**0.5
    ratio       = np.maximum(magnitude-threshold, 0.0)/magnitude
    x_threshold = x*ratio[np.newaxis,:] 

    return x_threshold

class DPCSolver:
    '''
    DPCSolver class provides methods to preprocess 2D DPC measurements and solves DPC phase retrieval problems with Tikhonov or TV regularziation.
    '''
    def __init__(self, dpc_imgs, wavelength, na, na_in, pixel_size, rotation, dpc_num=4):
        '''
        Initialize system parameters and functions for DPC phase microscopy.
        '''
        self.wavelength = wavelength
        self.na         = na
        self.na_in      = na_in
        self.pixel_size = pixel_size
        self.dpc_num    = dpc_num
        self.rotation   = rotation
        self.fxlin      = np.fft.ifftshift(_genGrid(dpc_imgs.shape[-1], 1.0/dpc_imgs.shape[-1]/self.pixel_size))
        self.fylin      = np.fft.ifftshift(_genGrid(dpc_imgs.shape[-2], 1.0/dpc_imgs.shape[-2]/self.pixel_size))
        self.dpc_imgs   = dpc_imgs.astype('float32')
        self.normalization()
        self.pupil      = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na)
        self.sourceGen()
        self.WOTFGen()
        
    def setRegularizationParameters(self, reg_u = 1e-6, reg_p = 1e-6, tau_u = 1e-5, tau_p = 1e-5, rho = 1e-5):
        '''
        Set regularization parameters.
        '''
        # Tikhonov regularization parameters
        self.reg_u      = reg_u
        self.reg_p      = reg_p
        # TV regularization parameters
        self.tau_u      = tau_u
        self.tau_p      = tau_p
        # ADMM penalty parameter
        self.rho        = rho
        
    def normalization(self):
        '''
        Normalize the raw DPC measurements by dividing and subtracting out the mean intensity.
        '''
        for img in self.dpc_imgs:
            img          /= uniform_filter(img, size=img.shape[0]//2)
            meanIntensity = img.mean()
            img          /= meanIntensity        # normalize intensity with DC term
            img          -= 1.0                  # subtract the DC term
        
    def sourceGen(self):
        '''
        Generate DPC source patterns.
        '''
        self.source = []
        pupil       = pupilGen(self.fxlin, self.fylin, self.wavelength, self.na, na_in=self.na_in)
        for rotIdx in range(self.dpc_num):
            self.source.append(np.zeros((self.dpc_imgs.shape[-2:]), dtype='float32'))
            rotdegree = self.rotation[rotIdx]
            if rotdegree < 180:
                self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15>=
                                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = 1.0
                self.source[-1] *= pupil
            else:
                self.source[-1][self.fylin[:, naxis]*np.cos(np.deg2rad(rotdegree))+1e-15<
                                self.fxlin[naxis, :]*np.sin(np.deg2rad(rotdegree))] = -1.0
                self.source[-1] *= pupil
                self.source[-1] += pupil
        self.source = np.asarray(self.source)
        
    def WOTFGen(self):
        '''
        Generate transfer functions for each DPC source pattern.
        '''
        self.Hu = []
        self.Hp = []
        for rotIdx in range(self.source.shape[0]):
            FSP_cFP  = F(self.source[rotIdx]*self.pupil)*F(self.pupil).conj()
            I0       = (self.source[rotIdx]*self.pupil*self.pupil.conj()).sum()
            self.Hu.append(2.0*IF(FSP_cFP.real)/I0)
            self.Hp.append(2.0j*IF(1j*FSP_cFP.imag)/I0)
        self.Hu = np.asarray(self.Hu)
        self.Hp = np.asarray(self.Hp)

    def deconvTikhonov(self, AHA, determinant, fIntensity):
        '''
        Solve the DPC absorption and phase deconvolution with Tikhonov regularization.
        Inputs:
                AHA, determinant: auxiliary functions
                fIntensity      : Fourier spectra of DPC intensities
        Output:
                The optimal absorption and phase given the input DPC intensities and regularization parameters
        '''
        AHy        = np.asarray([(self.Hu.conj()*fIntensity).sum(axis=0), (self.Hp.conj()*fIntensity).sum(axis=0)])
        absorption = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
        phase      = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real

        return absorption+1.0j*phase

    def deconvTV(self,AHA,determinant,fIntensity,fDx,fDy,tv_order,tv_max_iter):
        '''
        Solve the DPC absorption and phase deconvolution with TV regularization using ADMM algorithm.
        Inputs:
                AHA, determinant: auxiliary functions
                fIntensity      : Fourier spectra of DPC intensities
                fDx, fDy        : TV filters in Fourier space
                tv_order        : number of times applying fDx and fDy on the signals for each filtering operation
                tv_max_iter     : number of ADMM iterations for solving this deconvolution problem
        Output:
                The optimal absorption and phase given the input DPC intensities and regularization parameters
        '''
        z_k     = np.zeros((4,)+self.dpc_imgs.shape[1:], dtype='complex64')
        u_k     = np.zeros((4,)+self.dpc_imgs.shape[1:], dtype='complex64')
        D_k     = np.zeros((4,)+self.dpc_imgs.shape[1:], dtype='complex64')
        for iteration in range(tv_max_iter):
            y_k        = [F(z_k[index] - u_k[index]) for index in range(4)]
            AHy        = np.asarray([(self.Hu.conj()*fIntensity).sum(axis=0)+self.rho*(fDx.conj()*y_k[0]+fDy.conj()*y_k[1]),\
                                     (self.Hp.conj()*fIntensity).sum(axis=0)+self.rho*(fDx.conj()*y_k[2]+fDy.conj()*y_k[3])])
            absorption = IF((AHA[3]*AHy[0]-AHA[1]*AHy[1])/determinant).real
            phase      = IF((AHA[0]*AHy[1]-AHA[2]*AHy[0])/determinant).real
            if iteration < tv_max_iter-1:
                if tv_order==1:
                    D_k[0] = absorption - np.roll(absorption,-1,axis=1)
                    D_k[1] = absorption - np.roll(absorption,-1,axis=0)
                    D_k[2] = phase - np.roll(phase,-1,axis=1)
                    D_k[3] = phase - np.roll(phase,-1,axis=0)
                elif tv_order==2:
                    D_k[0] = 2*absorption - np.roll(absorption,-1,axis=1) - np.roll(absorption,1,axis=1)
                    D_k[1] = 2*absorption - np.roll(absorption,-1,axis=0) - np.roll(absorption,1,axis=0)
                    D_k[2] = 2*phase - np.roll(phase,-1,axis=1) - np.roll(phase,1,axis=1)
                    D_k[3] = 2*phase - np.roll(phase,-1,axis=0) - np.roll(phase,1,axis=0)
                elif tv_order==3:
                    D_k[0] = 3*absorption - np.roll(absorption,-1,axis=1) - 3*np.roll(absorption,1,axis=1) + np.roll(absorption,2,axis=1)
                    D_k[1] = 3*absorption - np.roll(absorption,-1,axis=0) - 3*np.roll(absorption,1,axis=0) + np.roll(absorption,2,axis=0)
                    D_k[2] = 3*phase - np.roll(phase,-1,axis=1) - 3*np.roll(phase,1,axis=1) + np.roll(phase,2,axis=1)
                    D_k[3] = 3*phase - np.roll(phase,-1,axis=0) - 3*np.roll(phase,1,axis=0) + np.roll(phase,2,axis=0)
                z_k         = D_k + u_k
                z_k[:2,:,:] = _softThreshold(z_k[:2,:,:], self.tau_u/self.rho)
                z_k[2:,:,:] = _softThreshold(z_k[2:,:,:], self.tau_p/self.rho)
                u_k        += D_k - z_k
            print("DPC deconvolution with TV regularization, iteration:{:d}/{:d}".format(iteration+1, tv_max_iter), end="\r")

        return absorption+1.0j*phase

    def solve(self, method="Tikhonov", tv_order=1, tv_max_iter=20):
        '''
        Compute auxiliary functions and output multi-frame absortion and phase results.
        '''
        dpc_result  = []
        AHA         = [(self.Hu.conj()*self.Hu).sum(axis=0)+self.reg_u,            (self.Hu.conj()*self.Hp).sum(axis=0),\
                       (self.Hp.conj()*self.Hu).sum(axis=0)           , (self.Hp.conj()*self.Hp).sum(axis=0)+self.reg_p]
        if method == "Tikhonov":
            determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]
            for frame_index in range(self.dpc_imgs.shape[0]//self.dpc_num):
                fIntensity = np.asarray([F(self.dpc_imgs[frame_index*self.dpc_num+image_index]) for image_index in range(self.dpc_num)])
                dpc_result.append(self.deconvTikhonov(AHA, determinant, fIntensity))
        elif method == "TV":
            fDx = np.zeros(self.dpc_imgs.shape[1:], dtype='complex64')
            fDy = np.zeros(self.dpc_imgs.shape[1:], dtype='complex64')
            if tv_order==1:
                fDx[0,0] = 1.0; fDx[0,-1] = -1.0; fDx = F(fDx);
                fDy[0,0] = 1.0; fDy[-1,0] = -1.0; fDy = F(fDy);
            elif tv_order==2:
                fDx[0,0] = 2.0; fDx[0,-1] = -1.0; fDx[0,1] = -1.0; fDx = F(fDx);
                fDy[0,0] = 2.0; fDy[-1,0] = -1.0; fDy[1,0] = -1.0; fDy = F(fDy);
            elif tv_order==3:
                fDx[0,0] = 3.0; fDx[0,-1] = -1.0; fDx[0,1] = -3.0; fDx[0,2] = 1.0; fDx = F(fDx);
                fDy[0,0] = 3.0; fDy[-1,0] = -1.0; fDy[1,0] = -3.0; fDy[2,0] = 1.0; fDy = F(fDy);
            else:
                print('TVDeconv does not support order higher than 3!')
                raise

            dpc_result  = []
            regTerm     = self.rho*(fDx*fDx.conj()+fDy*fDy.conj())
            AHA[0]     += regTerm
            AHA[3]     += regTerm
            determinant = AHA[0]*AHA[3]-AHA[1]*AHA[2]
            for frame_index in range(self.dpc_imgs.shape[0]//self.dpc_num):
                fIntensity = np.asarray([F(self.dpc_imgs[frame_index*self.dpc_num+image_index]) for image_index in range(self.dpc_num)])
                dpc_result.append(self.deconvTV(AHA,determinant,fIntensity,fDx,fDy,tv_order,tv_max_iter))
            
        return np.asarray(dpc_result)
