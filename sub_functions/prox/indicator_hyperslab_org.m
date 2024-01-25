function p = indicator_hyperslab_org(x, low, high, w, dir)
%function p = project_hyperslab(x, low, high, w, dir)
%
% This procedure computes the projection onto the constraint set:
%
%                    low  <=  w'x  <=  high
% 
% When the input 'x' is an array, the computation can vary as follows:
%  - dir = 0 --> 'x' is processed as a single vector [DEFAULT]
%  - dir > 0 --> 'x' is processed block-wise along the specified direction
%
%  INPUTS
% ========
%  x    - ND array
%  low  - scalar or ND array compatible with the blocks of 'x' [OPTIONAL]
%  high - scalar or ND array compatible with the blocks of 'x' [OPTIONAL]
%  w    - ND array of the same size as 'x' [OPTIONAL]
%  dir  - integer, direction of block-wise processing [OPTIONAL]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version : 1.0 (27-04-2017)
% Author  : Giovanni Chierchia
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2017
%
% This file is part of the codes provided at http://proximity-operator.net
%
% By downloading and/or using any of these files, you implicitly agree to 
% all the terms of the license CeCill-B (available online).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% prepare the offsets
xTw = sum(w .* x, dir);
wTw = sum(w .^ 2, dir);
mu1 = max(0, low  - xTw) ./ wTw;
mu2 = min(0, high - xTw) ./ wTw;

% compute the projection
p = x + (mu1 + mu2).*w;
