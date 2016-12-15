% Extracts the parameters of the Gaussian Mixture from the DPMM parameter
% structure. See E.B. Sudderth, "Graphical Models for Visual Object
% Recognition and Tracking", 2006.

% Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
% No warranty, no commercial use.

function [w,m,C] = DPparamsToGM(params)
% convert parameters from DPMM
  z = find(params.counts == 0);
  if(~isempty(z))
    params.sums(z,:) = [];
    params.cholSSE(:,:,z) = [];
    params.counts(z) = [];
  end
  m = bsxfun(@times,params.sums',1./(params.counts+params.kappa));
  w = params.counts/(sum(params.counts));
  C = zeros(size(params.cholSSE));
  for i = 1:numel(params.counts)
    C(:,:,i) = params.cholSSE(:,:,i)'*params.cholSSE(:,:,i)/(params.counts(i)+params.nu);
  end
end % function