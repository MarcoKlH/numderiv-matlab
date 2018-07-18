function dfdx = numderiv(f,h,order,stencil)
%NUMDERIV numerical derivative with specified order and stencil
%   dfdx = NUMDERIV(f) returns numerical gradient of vector f, with unit
%   spacing (h=1), using a 5 point stencil (stencil = -2:2).
%
%   dfdx = NUMDERIV(f,h), where h is a scalar, uses h as spacing between
%   points.
%
%   dfdx = NUMDERIV(f,h,order), where order is a scalar specifying the
%   order of the derivative (order = 1 by default, order = 2 returns second
%   derivative).
%
%   dfdx = NUMDERIV(f,h,order,stencil), where stencil is a vector
%   indicating which neighbours to use, e.g. stencil = -1:1 for three point
%   central difference, stencil = 0:1 for two point forward difference,
%   stencil = -2:2 for five-point central difference.
%
%   NUMDERIV(f,h,1,-1:1) is equivalent to gradient(f,h), while 
%   NUMDERIV(f,h,2,-2*(-1:1)) is equivalent to gradient(gradient(f,h),h). 
%   Note that the -2*(-1:1) stencil acts as a moving average filter.
%   NUMDERIV(f,h,2,-2:2) is more exact, but also preserves high frequency
%   noise.
%   
%   At the edges of the sequence, and around nan-gaps, the derivative is 
%   computed using any available elements of the specified stencil (e.g. 
%   the start will use forward difference and the end will use a backward 
%   difference). 
%   
%   This function is much slower than gradient, easily 3 times slower, so
%   if running time is more important than the differentiation scheme, use
%   GRADIENT instead. 
%
%   See also GRADIENT
%
%   Author: Marco KleinHeerenbrink
%   July 2017; Last revision 17-July-2017
%
%   Inspired by: http://web.media.mit.edu/~crtaylor/calculator.html
%
%   To improve:
%   A large part of the time this function spends in spdiags(), two thirds
%   of which are to identify realizable stencils for each data point...
%   Specifically the call S = spdiags(spdiags(...)) first constructs a
%   sparse matrix with the stencil on the diagonals and then extracts these
%   diagonals again. This can probably be done much faster using some 
%   clever indexing.


% prepare inputs
dfdx = nan*f;
m = length(f);
if nargin<3
    order = 1;
end
if nargin<4
    stencil = [-2,1,0,1,2];
end
stencil = stencil(:);

% find coefficients for all occuring stensils (edges and nan-gaps)
S = spdiags(spdiags(~isnan(f(:)).*(stencil'+0.5),stencil,m,m));% add 0.5 to distinguish from index 0 or nan
[uniqueStensils,~,idcsS] = unique(S,'rows');
C = 0*S;
for i = 1:size(uniqueStensils,1)
    iS = uniqueStensils(i,:)'-0.5;
    iC = 0*iS;
    slc_rm = ismembertol(mod(iS,1),0.5);
    iS(slc_rm) = [];
    N = length(iS);
    n = 0:(N-1);
    
    % compute stencil coefficients
    iC(~slc_rm) = (iS.^n)'\(n==order)';
    
    % if all coefficients are zero, stencil is useless
    if all(iC == 0); iC = iC*nan; end
    
    % add coefficients to the list
    slc = idcsS == i;
    C(slc,:) = repmat(iC',sum(slc),1);
end

% construct sparse differentiation matrix
spG = spdiags(C,stencil,m,m);

% numerical differentiation:
dfdx(:) = (-1)^order * spG'*f(:) * prod(1:order)/ h^order;
end