%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% genTransferFunction calculate transfer functions for amplitude          %
% and phase for partially coherent imaging systems                        %
%                                                                         %
% Inputs:                                                                 %
%        source : source patterns in fourier space                        %
%        pupil  : pupil of the imaging system                             %
% Outputs:                                                                %
%        ImaginaryTransferFunction : phase transfer function              %
%        RealTransferFunction : amplitude transfer function               %
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
function [ImaginaryTransferFunction, RealTransferFunction] = genTransferFunction(source, pupil)
    F                         = @(x) fft2(x);
    IF                        = @(x) ifft2(x);
    source_f                  = rot90(padarray(fftshift(source), [1, 1], 'post'), 2);
    source_f                  = ifftshift(source_f(1:end-1, 1:end-1));  
    FP_cFSP                   = conj(F(source_f.*pupil)).*F(pupil);
    RealTransferFunction      = 2*IF(real(FP_cFSP));
    ImaginaryTransferFunction = 2*IF(1i*imag(FP_cFSP));
    DC                        = sum(sum(source_f.*abs(pupil).^2));
    ImaginaryTransferFunction = 1i*ImaginaryTransferFunction/DC;
    RealTransferFunction      = RealTransferFunction/DC;

end

