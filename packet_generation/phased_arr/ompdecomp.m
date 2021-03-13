function [Wcoeff,Watom,WatomIdx,Errnorm] = ompdecomp(Wopt,Adict,varargin)
%ompdecomp  Decomposition using orthogonal matching pursuit
%   [COEFF,DICTATOM,ATOMIDX,ERRNORM] = ompdecomp(X,DICT) computes the
%   decomposition, COEFF and DICTATOM, of the signal, X, so that
%   DICTATOM*COEFF is approximately X. X is an NxNC matrix representing the
%   data for decomposition. DICT is a collection of possible atoms that can
%   be used to construct the data. The decomposition is computed using
%   orthogonal matching pursuit (OMP) method to minimize the Frobenius norm
%   of ||X-DICTATOM*COEFF||.
%
%   DICTATOM is an NxNS matrix whose columns are atoms that form the basis
%   of the signal. All atoms are chosen from the dictionary specified in
%   DICT. NS represents the number of atoms that are chosen from the
%   dictionary and is a measure of signal sparsity. COEFF is an NSxNC
%   matrix whose rows are coefficients for the corresponding atoms in
%   DICTATOM. ATOMIDX is a length-NS row vector containing the indices of
%   chosen atoms in the original dictionary such that DICT(:,ATOMIDX) is
%   the same as DICTATOM. ERRNORM is a scalar representing the norm of the
%   decomposition error.
%
%   [...] = ompdecomp(...,'MaxSparsity',NM) specifies the maximum sparsity 
%   of the decomposition, NM, as a positive integer. The decomposition
%   stops at a sparsity of NM if the desired error cannot be achieved.
%   The default value of NM is 1 and the default error tolerance is 0.
%
%   [...] = ompdecomp(...,'NormWeight',W) uses OMP to minimize the weighted
%   Frobenius norm of ||W^(1/2)*(X-DICTATOM*COEFF)|| where W is an NxN
%   matrix. The default value of W is an identity matrix.
%
%   % Example:
%   %   Given a set of optimal full digital beamforming weights of an 
%   %   8-element ULA, decompose the weights into a combination of analog 
%   %   and digital beamforming weights assuming there are only 2 RF chains
%   %   and show the combined weights achieve similar performance as the 
%   %   optimal weights.
%   %
%   %   Note that in the context of hybrid beamforming, COEFF represents 
%   %   the digital weights; DICTATOM represents the analog weights; and
%   %   DICT is the collection of possible steering vectors that can be
%   %   used as analog weights.
%
%   Wopt = steervec((0:7)*0.5,[20 -40]);
%   stvdict = steervec((0:7)*0.5,-90:90);
%   [Wbb,Wrf,WDictIdx,Normerr] = ompdecomp(Wopt,stvdict,'MaxSparsity',2);
%
%   % Compare beam pattern from optimal weights and hybrid weights
%   plot(-90:90,abs(sum(Wopt'*stvdict)),'-',...
%       -90:90,abs(sum((Wrf*Wbb)'*stvdict)),'--','LineWidth',2);
%   xlabel('Angles (degrees)');  ylabel('Pattern');
% 	legend('Optimal','Hybrid')
%
%   See also phased, diagbfweights, omphybweights.

% Copyright 2019 The MathWorks, Inc.

% Reference
% 
% [1] Ayach, Omar El et al. Spatially Sparse Precoding in Millimeter Wave
% MIMO Systems, IEEE Trans on Wireless Communications, Vol. 13, No. 3,
% March 2014

%#ok<*EMCLS>
%#ok<*EMCA>
%#codegen

narginchk(2,inf);

validateattributes(Wopt,{'double'},{'nonempty','nonnan','finite','2d'},...
    'ompdecomp','X');
[Nelem,Nw] = size(Wopt);
validateattributes(Adict,{'double'},{'nonempty','nonnan','finite','2d'...
    'nrows',Nelem},'ompdecomp','DICT');


defaultMaxSparsity = 1;
defaultNormWeight = eye(Nelem);

if isempty(coder.target)
     p = inputParser;
     p.addParameter('MaxSparsity',defaultMaxSparsity);
     p.addParameter('NormWeight',defaultNormWeight);
     p.parse(varargin{:});
     Nsparsity = p.Results.MaxSparsity;
     W = p.Results.NormWeight;
else
     parms = struct('MaxSparsity',uint32(0), ...
                    'NormWeight',uint32(0));
     pstruct = eml_parse_parameter_inputs(parms,[],varargin{:});
     Nsparsity = eml_get_parameter_value(pstruct.MaxSparsity,defaultMaxSparsity,varargin{:});
     W = eml_get_parameter_value(pstruct.NormWeight,defaultNormWeight,varargin{:});
end

validateattributes(Nsparsity,{'double'},{'positive','integer','scalar','<=',size(Adict,2)},...
    'ompdecomp','MaxSparsity');
validateattributes(W,{'double'},{'nonempty','finite','nonnan',...
    'size',[Nelem Nelem]},'ompdecomp','NormWeight');

% Use convention in [1], convert from Comm convention
Watom_temp = complex(zeros(Nelem,Nsparsity));
Wcoeff_temp = complex(zeros(Nsparsity,Nw));
WatomIdx_temp = zeros(1,Nsparsity);
Wres = Wopt;
Errnorm = 1;
m = 1;
while m <= Nsparsity && Errnorm > eps
    Ns = m;
    Psi = Adict'*W*Wres;
    [~,k] = max(diag(Psi*Psi'));
    WatomIdx_temp(m) = k;
    Watom_temp(:,m) = Adict(:,k);
    Wcoeff_temp(1:m,:) = (Watom_temp(:,1:m)'*W*Watom_temp(:,1:m))\(Watom_temp(:,1:m)'*W*Wopt);
    temp = Wopt-Watom_temp(:,1:m)*Wcoeff_temp(1:m,:);
    Errnorm = norm(temp,'fro');
    Wres = temp/Errnorm;
    m = m+1;
end
Watom = Watom_temp(:,1:Ns);
Wcoeff = Wcoeff_temp(1:Ns,:);
WatomIdx = WatomIdx_temp(1:Ns);



