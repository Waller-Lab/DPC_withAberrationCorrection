%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gradientPupil calculates the gradient vector of the loss function with  %
% respect to the Zernike coefficients                                     %
%                                                                         %
% Inputs:                                                                 %
%        zernike_est : current estimatino of Zernike coefficients         %
%        source      : illumination patterns                              %
%        pupil       : pupil of the imaging system                        %
%        fIDPC       : Fourier spectrum of measurements                   %
%        f_amplitude : Fourier spectrum of current estimate of amplitude  %
%        f_phase     : Fourier spectrum of current estimate of phase      %
%        use_gpu     : true: use GPU, false use CPU                       %
% Outputs:                                                                %
%        f           : loss function value                                %
%        g           : gradient                                           %
%                                                                         %
% Copyright (C) 2018 Michael Chen                                         %
%                                                                         %
% This program is free software: you can redistribute it and/or modify    %
% it under the terms of the GNU General Public License as published by    %
% the Free Software Foundation, either version 3 of the License, or       %
% (at your option) any later version.                                     %
%                                                                         %
% This program is distributed in the hope that it will be useful,         %
% but WITHOUT ANY WARRANTY; without even the implied warranty of          %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           %
% GNU General Public License for more details.                            %
%                                                                         % 
% You should have received a copy of the GNU General Public License       %
% along with this program.  If not, see <http://www.gnu.org/licenses/>.   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, g] = gradientPupil(zernike_est)
    global pupil source fIDPC f_amplitude f_phase use_gpu;
    F           = @(x) fft2(x);
    IF          = @(x) ifft2(x);
    if use_gpu
        zernike_est = gpuArray(zernike_est);
    end
    f           = 0;
    g           = 0;
    pupil_phase = aberrationGeneration(zernike_est);
    pupil_est   = pupil.*exp(1i*pupil_phase);
    
    for source_index = 1:size(source, 3)
        % forward model
        source_f      = rot90(padarray(fftshift(source(:, :, source_index)), [1, 1], 'post'), 2);
        source_f      = ifftshift(source_f(1:end-1, 1:end-1));
        DC            = sum(sum(source_f.*abs(pupil_est).^2));
        f_sp          = F(source_f.*pupil_est);
        f_p           = F(pupil_est);
        H_first_half  = conj(f_sp).*f_p;
        H_second_half = conj(f_p).*f_sp;
        
        % compute cost function value
        residual      = fIDPC(:,:,source_index) -...
                        (IF(H_first_half+H_second_half).*f_amplitude +...
                        1i*IF(H_first_half-H_second_half).*f_phase)/DC;
        f             = f + 0.5*norm(residual(:))^2;
        
        % compute gradient
        backprop_1    = F(conj(f_amplitude).*residual);
        backprop_2    = F(-1i*conj(f_phase).*residual);
        grad_pupil    = (IF(f_sp.*(backprop_1+backprop_2)) +...
                        source_f.*IF(f_p.*(backprop_1-backprop_2)))/DC;
        g             = g - aberrationDecomposition(-1i*conj(pupil_est).*grad_pupil);
    end
    f = gather(f);
    g = gather(real(g)); 
end