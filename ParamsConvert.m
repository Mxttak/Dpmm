% Converts the vectors in the parameters from row vectors as used by Jacob
% Eisenstein to column vectors. The latter are more convenient to use with
% Eigen.

% Copyright (C) 2016 Maxim Dolgov: m<dot>dolgov<at>web<dot>de
% No warranty, no commercial use.

function out = ParamsConvert(in)
  out = in;
  for i = 1:numel(out)
    out(i).sums = out(i).sums';
    out(i).initmean = out(i).initmean';
  end
end