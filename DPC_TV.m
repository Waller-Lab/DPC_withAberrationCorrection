%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DPC_L2 solves the least squared solutions for phase and amplitude       %
% with L2 regularization                                                  %
%                                                                         %
% Inputs:                                                                 %
%        zernike_coeff: Zernike coefficients for the aberration function  %
%        rho          : regularization for amplitude and phase            %
%        z_k          : spliting variable for 2D gradient                 %
%        u_k          : Lagrange multiplier                               %
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
function [amplitude, phase] = DPC_TV(zernike_coeff, rho, z_k, u_k)
    global pupil dim fIDPC source Dx Dy padsize use_gpu
    F               = @(x) fft2(x);
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
    
    M11         = sum(abs(Hr).^2, 3) + rho * abs(Dx).^2 + rho * abs(Dy).^2;
    M12         = sum(conj(Hr).*Hi, 3);
    M21         = sum(conj(Hi).*Hr, 3);
    M22         = sum(abs(Hi).^2, 3) + rho * abs(Dx).^2 + rho * abs(Dy).^2;
    denominator = M11.*M22-M12.*M21;
    b2          = F(padarray(z_k - u_k, [padsize/2, padsize/2,0]));
    I1          = sum(fIDPC.*conj(Hr), 3) + rho*(conj(Dx).*b2(:, :, 1) + conj(Dy).*b2(:, :, 2));
    I2          = sum(fIDPC.*conj(Hi), 3) + rho*(conj(Dx).*b2(:, :, 3) + conj(Dy).*b2(:, :, 4));
    phase       = real(IF((I2.*M11-I1.*M21)./(denominator+eps)));
    amplitude   = real(IF((I1.*M22-I2.*M12)./(denominator+eps)));
    phase       = phase(padsize/2+1:end-padsize/2, padsize/2+1:end-padsize/2);
    amplitude   = amplitude(padsize/2+1:end-padsize/2, padsize/2+1:end-padsize/2);
end

