%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DPC_L2 solves the least squared solutions for phase and amplitude       %
% with L2 regularization                                                  %
%                                                                         %
% Problem: min ||fIDPC-Ax||+alpha||a||+beta||p||      |M11 M12|       |a| %
%           x                                     A = |M21 M22| , x = |p| %
% a: F{amplitude}, p: F{phase}                                            %
%                                                                         %
% Inputs:                                                                 %
%        zernike_coeff: Zernike coefficients for the aberration function  %
%        reg_L2       : regularization for amplitude and phase            %
%        use_gpu      : true: use GPU, false use CPU                      %
%        source       : illumination patterns                             %
%        pupil        : circular support in Fourier space                 %
%        dim          : size of images                                    %
%        fIDPC        : Fourier spectrum of measurements                  %
% Outputs:                                                                %
%        amplitude    : absorption of the complex field                   %
%        phase        : phase of the complex field                        %
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
function [amplitude, phase] = DPC_L2(zernike_coeff, reg_L2)
    global pupil dim fIDPC source use_gpu
    IF              = @(x) ifft2(x);
    Hi              = zeros(dim(1), dim(2), size(source, 3));
    Hr              = zeros(dim(1), dim(2), size(source, 3));    
    if use_gpu
        Hi = gpuArray(Hi);
        Hr = gpuArray(Hr);
    end
    pupilphase      = aberrationGeneration(zernike_coeff);
    pupil_aberrated = pupil.*exp(1i*pupilphase);

    for source_index = 1:size(source, 3)
        [Hi(:, :, source_index),...
         Hr(:, :, source_index)] = genTransferFunction(source(:, :, source_index), pupil_aberrated);
    end
    
    % matrix pseudo inverse
    M11             = sum(abs(Hr).^2, 3) + reg_L2(1);
    M12             = sum(conj(Hr).*Hi, 3);
    M21             = sum(conj(Hi).*Hr, 3);
    M22             = sum(abs(Hi).^2, 3) + reg_L2(2);
    denominator     = M11.*M22 - M12.*M21;
    I1              = sum(fIDPC.*conj(Hr), 3);
    I2              = sum(fIDPC.*conj(Hi), 3);
    amplitude       = real(IF((I1.*M22-I2.*M12)./denominator));
    phase           = real(IF((I2.*M11-I1.*M21)./denominator));   
end

