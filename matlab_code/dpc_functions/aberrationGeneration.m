%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% aberrationGeneration generates the aberration function given the Zernike%
% coefficients                                                            %
%                                                                         %
% Inputs:                                                                 %
%        zernike_coeff: Zernike coefficients for the aberration function  %
%        zernike_poly : Zernike polynomials                               %
%        dim          : size of images                                    %
% Outputs:                                                                %
%        aberration   : aberration of the imaging system                  %
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
function aberration = aberrationGeneration(zernike_coeff)
    global zernike_poly dim;
    aberration = zernike_poly*zernike_coeff;
    aberration = reshape(aberration, dim);
end