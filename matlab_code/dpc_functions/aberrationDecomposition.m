%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% aberrationDecomposition decomposes the aberration function into Zernike %
% coefficients                                                            %
%                                                                         %
% Inputs:                                                                 %
%        aberration   : aberration of the imaging system                  %
%        zernike_poly : Zernike polynomials                               %
% Outputs:                                                                %
%        zernike_coeff: Zernike coefficients for the aberration function  %
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
function zernike_coeff = aberrationDecomposition(aberration)
    global zernike_poly;
    zernike_coeff = zernike_poly'*aberration(:);
end