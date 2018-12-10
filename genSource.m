%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% genSource generates source patterns for partially coherent illumination %
% under Kohler illumination                                               %
%                                                                         %
% Inputs:                                                                 %
%        illu_rotation: rotation angle of the asymmetric axis             %
%        na_illum     : illumination NA                                   %
%        na_inner     : inner radious of annular illumination             %
%        lambda       : wavelength                                        %
%        Fx, Fy       : spaital frequency coordinates                     %
% Outputs:                                                                %
%        source       : source patterns                                   %
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
function source = genSource(illu_rotation, na_illum, na_inner, lambda, Fx, Fy)
    % support of the source
    S0   = (sqrt(Fx.^2+Fy.^2)*lambda<=na_illum & sqrt(Fx.^2+Fy.^2)*lambda>=na_inner*na_illum);
    mask = zeros(size(Fx));

    % asymmetric mask based on illumination angle
    if illu_rotation < 180 || illu_rotation == 270 
        mask(Fy>=(Fx*tand(illu_rotation))) = 1;
    else
        mask(Fy<=(Fx*tand(illu_rotation))) = 1;    
    end
    source = S0.*mask;
end

