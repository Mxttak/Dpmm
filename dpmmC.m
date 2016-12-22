% A wrapper for my code that initializes the parameters and calls the Gibbs
% sampler.

% Adapted from the code by Jacob Eisenstein taht is available at
% https://github.com/jacobeisenstein/DPMM 

% Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
% No warranty, no commercial use.

function params = dpmmC(data, num_its)

%some stats
allmean = mean(data,1);
allcov = cov(data);

    params(1).alpha = size(data,1) / 50; %1 / wishrnd(1,1);
    params(1).kappa = .1; %T / 1000; %a pseudo count on the mean
    params(1).nu = 6; %a pseudo-count on the covariance
    params(1).initmean = allmean;
    params(1).initcov = allcov / 10;
    params(1).num_classes = 0;
    params(1).counts = 0;
    params(1).sums = [];
    params(1).cholSSE = [];
    params(1).classes = ones(size(data,1),1);
    params(1) = addNewClass(params(1));
    params(1) = unhideObservations(params(1), 1, data);
    params(1) = ParamsConvert(params(1));

    params = ParamsConvert(dpmm_matlab(ParamsConvert(params(1)),data',num_its));

end

