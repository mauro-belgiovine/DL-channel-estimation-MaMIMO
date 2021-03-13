function [Fbb,Frf,Wbb,Wrf] = omphybweights(Hchann_in,Ns,NtRF,At,NrRF,Ar,npower)
%omphybweights Hybrid beamforming weights using orthogonal matching pursuit
%   [WPBB,WPRF] = omphybweights(HCHAN,NS,NTRF,AT) returns the hybrid
%   precoding weights, WPBB and WPRF, for the channel matrix, HCHAN. These
%   weights together approximates the optimal full digital precoding
%   weights of HCHAN.
%
%   NS is the number of independent data streams propagated through the
%   channel defined in HCHAN. NTRF specifies the number of RF chains in the
%   transmit array.
%
%   AT is the collection of possible analog weights for WPRF.
%
%   HCHAN can be either a matrix or a 3-dimensional array. If HCHAN is a
%   matrix, HCHAN has a size of NtxNr where Nt is number of elements in the
%   transmit array and Nr is the number of elements in the receive array.
%   If HCHAN is a 3-dimensional array, its dimension is LxNtxNr where L is
%   the number of subcarriers.
%
%   If HCHAN is a matrix, WPBB is an NSxNTRF matrix and WPRF is an NTRFxNt
%   matrix. AT is a matrix with Nt rows.
%
%   If HCHAN is a 3-dimensional array, WPBB has a size of LxNSxNTRF and
%   WPRF has a size of LxNTRFxNt. Each page corresponds to a subcarrier. AT
%   also has L pages and in each page is a matrix of Nt rows.
%
%   [WPBB,WPRF,WCBB,WCRF] = omphybweights(HCHAN,NS,NTRF,AT,NRRF,AR) also
%   returns the hybrid combining weights, WCBB and WCRF, for the channel
%   matrix, HCHAN. These combining weights, together with the precoding
%   weights, diagonalize the channel into independent subchannels so that
%   the result of WPBB*WPRF*HCHAN*WCRF*WCBB is close to a diagonal matrix.
%
%   NRRF specifies the number of RF chains in the receive array. AR is the
%   collection of possible analog weights for WCRF.
%
%   If HCHAN is a matrix, WCBB is an NRRFxNS matrix, and WCRF is an NrxNRRF
%   matrix. R is a matrix with Nr rows.
%
%   If HCHAN is a 3-dimensional array, WCBB has a size of LxNRRFxNS, and
%   WCRF has a size of LxNrxNRRF. Each page corresponds to a subcarrier. AR
%   also has L pages and in each page is a matrix of Nr rows.
%
%   [...] = omphybweights(HCHAN,NS,NTRF,AT,NRRF,AR,NPOW) specifies the
%   noise power, NPOW, in each receive antenna element as a scalar. All
%   subcarriers are assumed to have the same noise power. The default value
%   of NPOW is 0.
%
%   % Example:
%   %   Assume a 8x4 MIMO system with 4 RF chains in transmit array and 2
%   %   RF chains in receive array. Show that the hybrid weights can
%   %   support transmitting two data streams simultaneously.
%
%   txpos = (0:7)*0.5;
%   rxpos = (0:3)*0.5;
%   chan = scatteringchanmtx(txpos,rxpos,10);
%   Ntrf = 4; Nrrf = 2; Ns = 2;
%   txdict = steervec(txpos,-90:90);
%   rxdict = steervec(rxpos,-90:90);
%   [Fbb,Frf,Wbb,Wrf] = omphybweights(chan,Ns,Ntrf,txdict,Nrrf,rxdict);
%
%   % Calculate effective channel matrix. A diagonal effective channel
%   % matrix indicates the capability of simultaneous transmission of
%   % multiple data streams.
%   chan_eff = Fbb*Frf*chan*Wrf*Wbb
%
%   See also phased, diagbfweights, ompdecomp.

% Copyright 2019 The MathWorks, Inc.

% Reference
% 
% [1] Ayach, Omar El et al. Spatially Sparse Precoding in Millimeter Wave
% MIMO Systems, IEEE Trans on Wireless Communications, Vol. 13, No. 3,
% March 2014

%#ok<*EMCLS>
%#ok<*EMCA>
%#codegen

narginchk(4,7);
if nargin == 4
    nargoutchk(0,2);
    isPrecodingOnly = true;
else
    narginchk(6,7);
    if nargout >=1 && nargout <= 2
        isPrecodingOnly = true;
    else
        isPrecodingOnly = false;
        if nargin < 7
            NPOW = 0;
        else
            sigdatatypes.validatePower(npower,'omphybweights','NPOW',{'scalar'});
            NPOW = npower;
        end
    end
end
validateattributes(Hchann_in,{'double'},{'nonnan','nonempty','finite','3d'},...
    'omphybweights','HCHAN');

isHmatrix = ismatrix(Hchann_in);

if isHmatrix
    [Nt,Nr] = size(Hchann_in);
    if isPrecodingOnly
        validateattributes(Ns,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',NtRF},'omphybweights','NS');
        validateattributes(NtRF,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',Nt},'omphybweights','NTRF');
        validateattributes(At,{'double'},{'nonempty','nonnan',...
            '2d','nrows',Nt},'omphybweights','AT');
        
        [Fbb,Frf] = getWeightsForSubcarrier(Hchann_in,Ns,NtRF,At);
    else
        validateattributes(Ns,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',min(NtRF,NrRF)},'omphybweights','NS');
        validateattributes(NtRF,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',Nt},'omphybweights','NTRF');
        validateattributes(At,{'double'},{'nonempty','nonnan',...
            '2d','nrows',Nt},'omphybweights','AT');
        validateattributes(NrRF,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',Nr},'omphybweights','NRRF');
        validateattributes(Ar,{'double'},{'nonempty','nonnan',...
            '2d','nrows',Nr},'omphybweights','AR');

        [Fbb,Frf,Wbb,Wrf] = getWeightsForSubcarrier(Hchann_in,Ns,NtRF,At,NrRF,Ar,NPOW);
    end
else
    [L,Nt,Nr] = size(Hchann_in);
    if isPrecodingOnly
        validateattributes(Ns,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',NtRF},'omphybweights','NS');
        validateattributes(NtRF,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',Nt},'omphybweights','NTRF');
        validateattributes(At,{'double'},{'nonempty','nonnan',...
            '3d','nrows',L,'ncols',Nt},'omphybweights','AT');

        Fbb = zeros(L,Ns,NtRF,'like',1+1i);
        Frf = zeros(L,NtRF,Nt,'like',1+1i);
        for m = 1:L
            [Fbb(m,:,:),Frf(m,:,:)] = getWeightsForSubcarrier(...
                squeeze(Hchann_in(m,:,:)),Ns,NtRF,squeeze(At(m,:,:)));
        end
    else
        validateattributes(Ns,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',min(NtRF,NrRF)},'omphybweights','NS');
        validateattributes(NtRF,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',Nt},'omphybweights','NTRF');
        validateattributes(At,{'double'},{'nonempty','nonnan',...
            '3d','nrows',L,'ncols',Nt},'omphybweights','AT');
        validateattributes(NrRF,{'double'},{'nonempty','nonnan','positive','integer',...
            '<=',Nr},'omphybweights','NRRF');
        validateattributes(Ar,{'double'},{'nonempty','nonnan',...
            '3d','nrows',L,'ncols',Nr},'omphybweights','AR');

        Fbb = zeros(L,Ns,NtRF,'like',1+1i);
        Frf = zeros(L,NtRF,Nt,'like',1+1i);
        Wbb = zeros(L,NrRF,Ns,'like',1+1i);
        Wrf = zeros(L,Nr,NrRF,'like',1+1i);
        for m = 1:L
            [Fbb(m,:,:),Frf(m,:,:),Wbb(m,:,:),Wrf(m,:,:)] = getWeightsForSubcarrier(...
                squeeze(Hchann_in(m,:,:)),Ns,NtRF,squeeze(At(m,:,:)),NrRF,squeeze(Ar(m,:,:)),NPOW);
        end
    end
end

end

function varargout = getWeightsForSubcarrier(Hin,Ns,NtRF,At,NrRF,Ar,NPOW)

    isPrecodingOnly = (nargin<7);
    % Use convention in [1], convert from Comm convention
    H = Hin.';
    [~,~,v] = svd(H);
    Fopt = v(:,1:Ns);

    [Fbb,Frf] = ompdecomp(Fopt,At,'MaxSparsity',NtRF);
    Fbb = sqrt(Ns)*Fbb/norm(Frf*Fbb,'fro');

    if ~isPrecodingOnly
        Wmmse = ((Fbb'*Frf'*(H'*H)*Frf*Fbb+NPOW*Ns*eye(Ns))\Fbb'*Frf'*H')';
        Ess = 1/Ns*eye(Ns);
        Eyy = H*Frf*Fbb*Ess*Fbb'*Frf'*H'+NPOW*eye(size(Ar,1));

        [Wbb,Wrf] = ompdecomp(Wmmse,Ar,'MaxSparsity',NrRF,'NormWeight',Eyy);
    end

    % convert back to comm convention
    % match Wbb'*Wrf'*H.'*Frf*Fbb*X.' to 
    % X*Fbb.'*Frf.'*H*conj(Wrf)*conj(Wbb) to
    % X*Fbb*Frf*H*Wrf*Wbb

    if isPrecodingOnly
        Fbb_out = Fbb.';
        Frf_out = Frf.';
        varargout = {Fbb_out,Frf_out};
    else
        Fbb_out = Fbb.';
        Frf_out = Frf.';
        Wbb_out = conj(Wbb);
        Wrf_out = conj(Wrf);
        varargout = {Fbb_out,Frf_out,Wbb_out,Wrf_out};
    end
end

