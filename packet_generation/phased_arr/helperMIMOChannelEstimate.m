function [hD,P,ltf_o,hDmmse] = helperMIMOChannelEstimate(rxData,prm,Nps,tau,SNR,isMMSE)
% Estimate channel from the preamble signal data tones
%
% Example helper utility.

% Copyright 2016-2017 The MathWorks, Inc.

[~, nltf, numRx] = size(rxData);
numSTS = prm.numSTS;
% nltf should be == numSTS

% Transmitted pilot mapping sequences
P = helperGetP(numSTS);

% Frequency subcarrier tones
ltfLeft = [1; 1;-1;-1; 1; 1;-1; 1; -1; 1; 1; 1; ...
            1; 1; 1;-1; -1; 1; 1;-1; 1;-1; 1; 1; 1; 1;];
ltfRight = [1; -1;-1; 1; 1; -1; 1;-1; 1; -1;-1;-1;-1; ...
            -1; 1; 1;-1; -1; 1;-1; 1; -1; 1; 1; 1; 1];   
ltf = [zeros(7,1); ltfLeft; 1; ltfRight;-1;-1;-1; 1; 1;-1; 1; ...
        -1; 1; 1;-1; ltfLeft; 1; ltfRight; 1;-1; 1;-1; 0; ...
        1;-1;-1; 1; ltfLeft; 1; ltfRight;-1;-1;-1; 1; 1;-1; 1; ...
        -1; 1; 1;-1; ltfLeft; 1; ltfRight; zeros(6,1)];    
Puse = P(1:numSTS,1:numSTS)'; % Extract and conjugate the P matrix 

ind = prm.CarriersLocations;
denom = nltf.*ltf(ind);

ltf_o = ltf(ind);

hD = complex(zeros(numel(denom),numSTS,numRx));
hDmmse = complex(zeros(numel(denom),numSTS,numRx));
for i = 1:numRx
    rxsym = squeeze(rxData(:,(1:nltf),i)); % Symbols per receive antenna
    for j = 1:numSTS
        hD(:,j,i) = rxsym*Puse(:,j)./denom;
        if isMMSE
            hDmmse(:,j,i) = reshape(LMMSE_ce(hD(:,j,i),length(prm.CarriersLocations),length(prm.CarriersLocations),Nps,tau,SNR(i)), [length(prm.CarriersLocations),1]);
        end
    end
end


%plot_mimo_channel(hD,numRx,41)
%plot_mimo_channel(hDmmse,numRx,42)
end

% [EOF]
