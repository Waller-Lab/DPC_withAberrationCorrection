% set coordinates x and y in real space
x        = -(dim(2)-mod(dim(2), 2))/2:1:(dim(2)-mod(dim(2), 2))/2-(mod(dim(2), 2)==0);
x        = ps*x;
y        = -(dim(1)-mod(dim(1), 2))/2:1:(dim(1)-mod(dim(1), 2))/2-(mod(dim(1), 2)==0);
y        = ps*y;

% set coordinates fx and fy in Fourier space
dfx      = 1/dim(2)/ps; dfy = 1/dim(1)/ps;
fx       = -(dim(2)-mod(dim(2), 2))/2:1:(dim(2)-mod(dim(2), 2))/2-(mod(dim(2), 2)==0);
fx       = dfx*fx;
fy       = -(dim(1)-mod(dim(1), 2))/2:1:(dim(1)-mod(dim(1), 2))/2-(mod(dim(1), 2)==0);
fy       = dfy*fy;
fx       = ifftshift(fx);
fy       = ifftshift(fy);

% generate 2D grid
[Fx, Fy] = meshgrid(fx, fy);