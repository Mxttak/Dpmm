% test script for dpmmC a C++ implementation of a DPGMM code by Jacob
% Eisenstein that is available at https://github.com/jacobeisenstein/DPMM

% Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
% No warranty, no commercial use.

addpath(genpath('misc'))
addpath(genpath('NonlinearEstimationToolbox'))
addpath('dpmm')

profile on

% we use the GaussianMixture class from the Nonlinear Estimation Toolbox by
% Jannik Steinbring
pm(:,1) = [-2;0];
pCov(:,:,1) = [2^2, 1^2; 1^2, 3^2];
pw(1) = .4;
pm(:,2) = [2;-2];
pCov(:,:,2) = [1^2, -1^2; -1^2, 4^2];
pw(2) = .3;
pm(:,3) = [3;3];
pCov(:,:,3) = [1.5^2, -.2^2;-.2^2, .9^2];
pw(3) = .3;

prior = GaussianMixture(pm,pCov,pw);

% draw samples
nsamples = 1000;
niter = 100;
psamples = prior.drawRndSamples(nsamples);

% plot prior and likelihood
figure
subplot(2,2,1)
title('true')
hold on
subplot(2,2,1)
plot(psamples(1,:),psamples(2,:),'gx')
for i = 1:numel(pw)
  % this function by Gautam Vallabha is available at Mathworks' homepage
  h = plot_gaussian_ellipsoid(pm(:,i),pCov(:,:,i));
  set(h,'color','r');
end

% fit GM
tic
params = dpmmC(psamples',niter);
disp(toc)
subplot(2,2,3)
title('dpmm')
hold on
plot(psamples(1,:),psamples(2,:),'gx')
[dw,dm,dCov] = DPparamsToGM(params(end));
dpmmdens = GaussianMixture(dm,dCov,dw);
for i = 1:numel(pw)
  h = plot_gaussian_ellipsoid(pm(:,i),pCov(:,:,i));
  set(h,'color','r');
end
for i = 1:numel(dw)
  h = plot_gaussian_ellipsoid(dm(:,i),dCov(:,:,i));
  set(h,'color','b');
end


%% 3D plot
x = -10:.1:10;
y = -20:.1:20;
[X,Y] = meshgrid(x,y);

Z = exp(prior.logPdf([X(:),Y(:)]'));
dZ = exp(dpmmdens.logPdf([X(:),Y(:)]'));

Z = reshape(Z,[numel(y),numel(x)]);
dZ = reshape(dZ,[numel(y),numel(x)]);

subplot(2,2,2)
hold on
title('true')
mesh(x,y,Z)
view(10,50)
subplot(2,2,4)
hold on
title('dpmm')
mesh(x,y,dZ)
view(10,50)