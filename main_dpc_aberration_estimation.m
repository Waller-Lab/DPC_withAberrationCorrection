%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_dpc_aberration_estimation -                                        %
% Main file for DPC amplitude and phase recovery with aberration correction
%                                                                         %
% Copyright (C) 2018 Zack Phillips and Michael Chen                       %
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
clear; close all;
set(0, 'DefaultFigureWindowStyle', 'docked');
global zernike_poly pupil dim source fIDPC pupilphase f_amplitude f_phase use_gpu
F             = @(x) fft2(x);
IF            = @(x) ifft2(x);

%% load data
% this example dataset already remove the DC term and normalized by the
% total energy (DC term)
load('DPC_dataset.mat');

%% system parameters
dim           = [size(IDPC, 1), size(IDPC, 2)]; % image size
sigma         = 1.0;                            % partial coherence factor
na            = 0.40;                           % numerical aperture of the imaging system
na_illum      = sigma*na;                       % numerical aperture of the illumination
magnification = 20*2;                           % magnification of the imaging system
lambda        = 0.514;                          % wavelength in micron
ps            = 6.5/magnification;              % pixel size in micron
wavenumber    = 2*pi/lambda;                    % wave number
illu_rotation = [0, 180, 90];                   % orientation of the illumination           
num_rotation  = numel(illu_rotation);           % number of illumination used in DPC 
na_inner      = [0, 0, 0, 0];                   % if annular illumination is used, set the na corresponds to the inner radius     
num_Zernike   = 21;                             % highest order of Zernike coefficients used for pupil estimation
setCoordinate();

%% show measurements
figure('Name', 'normalized, background substracted DPC measurements', 'NumberTitle', 'off')
for source_index = 1:num_rotation+1
   subplot(2, 2, source_index);
   imagesc(IDPC(:, :, source_index)); axis image; axis off; colormap gray; caxis([-0.5, 0.5]);
   title(['DPC ', num2str(source_index)], 'FontSize', 24);
end
drawnow;

%% generate illumination sources
source             = zeros(dim(1), dim(2), num_rotation+1); 

for source_index = 1:num_rotation+1
    if source_index <= num_rotation
        source(:, :, source_index) = genSource(illu_rotation(source_index), na_illum, na_inner(source_index), lambda, Fx, Fy);
    else
        % additional coherent illumination
        source_temp                = zeros(dim(1), dim(2));
        source_temp(Fx==0 & Fy==0) = 1;
        source(:, :, source_index) = source_temp;
    end
end

figure('Name', 'Illumination and Phase Optical Transfer Functions', 'NumberTitle', 'off');
fig_rows    = floor(sqrt(num_rotation+1));
fig_cols    = floor((num_rotation+1)/fig_rows);
for fig_col = 1:num_rotation+1
    fig_index = fig_col;
    ax = subplot(fig_rows, fig_cols, fig_index);
    imagesc(fftshift(fx), fftshift(fy), fftshift(source(:, :, fig_col))); axis image; axis off;
    title(['Source ', num2str(fig_col)]);
    colormap(ax, 'gray'); caxis([0, 1]);
end
drawnow;

%% generate Zernike polynomials
pupil         = (Fx.^2+Fy.^2<=(na/lambda)^2);
zernike_poly  = genZernikePoly(Fx, Fy, na, lambda, num_Zernike);

figure('Name', 'generated Zernike polynomials (Defocus and Astigmatism)', 'NumberTitle', 'off');
subplot(131)
imagesc(fftshift(fx), fftshift(fy), fftshift(reshape(zernike_poly(:, 1), dim))); axis image; axis off; colormap jet;
title('aberration, Z_3', 'fontsize', 24);
subplot(132)
imagesc(fftshift(fx), fftshift(fy), fftshift(reshape(zernike_poly(:, 2), dim))); axis image; axis off; colormap jet;
title('aberration, Z_4', 'fontsize', 24);
subplot(133)
imagesc(fftshift(fx), fftshift(fy), fftshift(reshape(zernike_poly(:, 3), dim))); axis image; axis off; colormap jet;
title('aberration, Z_5', 'fontsize', 24);
drawnow;

%% joint estimation pupil function, amplitude and phase

% calculate frequency spectrum of the measurements
fIDPC               = F(IDPC);

% parameters for amplitude and phase reconstruction
zernike_coeff_k     = 0*randn(num_Zernike-3, 1);   % initalization of Zernike coefficients for pupil estimation, ignoring the first three orders
max_iter_algorithm  = 50;                          % maximum number of iteration of algorithm for DPC with pupil estimation.
reg_L2              = 1.0*[1e-1, 1e-4];            % parameters for L2 regurlarization [amplitude, phase] (can set to a very small value in noiseless case)
use_tv              = false;                       % true: use TV regularization, false: use L2 regularization
tau                 = [1e-5, 5e-3];                % parameters for total variation [amplitude, phase] (can set to a very small value in noiseless case)
verbose             = true;                        % true: show loss value and elapsed time at each iteration
show_result         = true;                        % true: show amplitude, phase and aberration images at each iteration
use_gpu             = true;                        % true: use GPU, false: use CPU

% parameters of L-BFGS algorithm for pupil estimation (default values should work)
addpath(genpath('..\minFunc\'));                   % add the path where you install the minFunc package
options.Method      = 'lbfgs';
options.maxIter     = 10;
options.PROGTOL     = 1e-30;
options.optTol      = 1e-30;
options.MAXFUNEVALS = 500;
options.corr        = 50;
options.usemex      = 0;
options.display     = false;

if use_gpu
    % place measurements and variables into GPU memory
    source = gpuArray(source);
    pupil  = gpuArray(pupil);
    fIDPC  = gpuArray(fIDPC);
end

t_start             = tic();
loss                = zeros(max_iter_algorithm, 1);
fig_results         = figure('Name', 'Reconstruction Process', 'NumberTitle', 'off');

for iter = 1:max_iter_algorithm

    if ~use_tv
        % Least Squared with L2 regularization
        [amplitude_k, phase_k]       = DPC_L2(zernike_coeff_k, reg_L2);
    else
        % ADMM algorithm with total variation regularization
        global padsize Dx Dy;
        padsize                      = 0;
        temp                         = zeros(dim);
        temp(1, 1)                   = 1;
        temp(1, end)                 = -1;
        Dx                           = F(temp);
        temp                         = zeros(dim);
        temp(1, 1)                   = 1;
        temp(end, 1)                 = -1;
        Dy                           = F(temp);
        if use_gpu
           Dx = gpuArray(Dx);
           Dy = gpuArray(Dy);
        end

        rho                          = 1;
        D_x                          = gpuArray(zeros(dim(1), dim(2), 4));
        u_k                          = gpuArray(zeros(dim(1), dim(2), 4));
        z_k                          = gpuArray(zeros(dim(1), dim(2), 4));
        for iter_ADMM = 1:20
           [amplitude_k, phase_k] = DPC_TV(zernike_coeff_k, rho, z_k, u_k);
            if iter_ADMM < 20
                D_x(:, :, 1)   = amplitude_k - circshift(amplitude_k, [0, -1]);
                D_x(:, :, 2)   = amplitude_k - circshift(amplitude_k, [-1, 0]);
                D_x(:, :, 3)   = phase_k - circshift(phase_k, [0, -1]);
                D_x(:, :, 4)   = phase_k - circshift(phase_k, [-1, 0]);
                z_k            = D_x + u_k;
                z_k(:, :, 1:2) = max(z_k(:, :, 1:2) - tau(1)/rho, 0) -...
                                 max(-z_k(:, :, 1:2) - tau(1)/rho, 0);
                z_k(:, :, 3:4) = max(z_k(:, :, 3:4) - tau(2)/rho, 0) -...
                                 max(-z_k(:, :, 3:4) - tau(2)/rho, 0);
                u_k            = u_k + (D_x-z_k);
            end
        end
            clear u_k z_k D_x;
    end
    
    f_amplitude = F(amplitude_k);
    f_phase     = F(phase_k);

    % pupil estimation
    [zernike_coeff_k, loss(iter)] = minFunc(@gradientPupil, zernike_coeff_k, options);
    pupilphase                    = aberrationGeneration(zernike_coeff_k);
    
    % print cost function value and computation time at each iteration
    if verbose
        fprintf('iteration: %04d, loss: %5.5e, elapsed time: %4.2f seconds\n', iter, loss(iter), toc(t_start));   
    end
    
    % plot recovered amplitude, phase and aberration at each iteration
    if show_result
        figure(fig_results);
        ax1 = subplot(2, 2, 1);
        imagesc(x, y, amplitude_k); axis image; axis off;
        colormap(ax1, 'gray'); caxis([-.15, 0.02]);
        title('recovered \alpha','FontSize', 24);
        ax2 = subplot(2, 2, 2);
        imagesc(x, y, phase_k); axis image; axis off;
        colormap(ax2, 'gray'); caxis([-0.5, 1.5]);
        title('recovered \phi', 'FontSize', 24);
        ax3 = subplot(2, 2, 3);
        imagesc(fftshift(fx), fftshift(fy), fftshift(pupilphase)); axis image; axis off;
        colormap(ax3, 'jet'); caxis([-1.0, 1.0]);
        title('recovered aberration', 'FontSize', 24);
        subplot(2, 2, 4);
        plot(1:iter, log10(loss(1:iter)), 'bo'); axis square;
        xlabel('iteration', 'FontSize', 20);
        ylabel('log_1_0(loss)', 'FontSize', 20)
        title('loss', 'FontSize', 24);
        linkaxes([ax1, ax2]);
        drawnow;
    end
end

% extract results and variables from GPU memory if using GPU computation
amplitude  = gather(amplitude_k); % optimized amplitude
phase      = gather(phase_k);     % optimized phase
pupilphase = gather(pupilphase);  % optimized aberration function
pupil      = gather(pupil);
source     = gather(source);
fIDPC      = gather(fIDPC);