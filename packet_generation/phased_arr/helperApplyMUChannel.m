  function [lossSig,chanDelay,TAU,CH_RESPONSE] = helperApplyMUChannel(sig,prm,spLoss,Ns,pktID,varargin)
% Apply MIMO channel to input signal
%   Options include:
%       'Scattering': Phased Scattering MIMO channel
%       'MIMO': Comm MIMO channel
% 
%   INPUTS:
%       * sig - Signal to pass through the channel
%       * prm - Struct with configuration
%       * spLoss - Channel losses (MIMO channel type)
%       * Ns - Number of scatterers (Scattering channel type)
%       * pktID - Packet ID, use different seeds
%       * varargin - Extra input variables
%
%   The channel is modeled with a fixed seed so as to keep the same channel
%   realization between sounding and data transmission. In reality, the
%   channel would evolve between the two stages. This evolution is modeled
%   by prepending the preamble signal to the data signal, to prime the
%   channel to a valid state, and then ignoring the preamble portion from
%   the channel output.

narginchk(5,6);
numUsers = prm.numUsers;
numTx = prm.numTx;
numRx = prm.numRx;
if nargin>5
    % preSig, for data transmission
    preSig = varargin{1}; 
    sigPad = [preSig; zeros(prm.numPadZeros,numTx); ...
              sig; zeros(prm.numPadZeros,numTx)];
else
    % No preSig, for sounding
    preSig = []; 
    sigPad = [sig; zeros(prm.numPadZeros,numTx)];
end
numBytesPerElement = 16;

% Create independent channels per user
chan      = cell(numUsers,1);
chanDelay = zeros(numUsers,1);
lossSig   = cell(numUsers,1);

switch prm.ChanType        
    case 'Scattering'
        % phased.ScatteringMIMOChannel
        %   No motion => static channel.
        
        % Tx & Rx Arrays
        [isTxURA,expFactorTx,isRxURA,expFactorRx] = helperArrayInfo(prm);
        
        %   Specify spacing in direct units (meters)
        if isTxURA % URA 
            txarray = phased.URA([expFactorTx,prm.numSTS], ...
                [0.5 0.5]*prm.lambda,'Element', ...
                phased.IsotropicAntennaElement('BackBaffled',false));
        else % ULA
            txarray = phased.ULA('Element', ...
                phased.IsotropicAntennaElement('BackBaffled',false),...
                'NumElements',numTx,'ElementSpacing',0.5*prm.lambda);
        end         
        
        CH_RESPONSE = zeros(numUsers,numTx,numRx,Ns);
        TAU = zeros(numUsers,Ns);
        % Create independent channels per user
        for uIdx = 1:numUsers

            if isRxURA(uIdx) % URA
                rxarray = phased.URA([expFactorRx(uIdx),prm.numSTSVec(uIdx)], ...
                    [0.5 0.5]*prm.lambda,'Element', ...
                    phased.IsotropicAntennaElement);
            else % ULA
                if numRx(uIdx)>1
                    rxarray = phased.ULA('Element',phased.IsotropicAntennaElement, ...
                        'NumElements',numRx(uIdx),'ElementSpacing',0.5*prm.lambda);
                else % numRx==1 
                    error(message('comm_demos:helperApplyMUChannel:invScatConf'));
                    % only a single antenna, but ScatteringMIMOChannel doesnt accept this!
                    % rxarray = phased.IsotropicAntennaElement;
                end                    
            end

            % Place scatterers domly in a circle from the center
            % posCtr = (prm.posTx+prm.posRx(:,uIdx))/2;
 
            % Place scatterers randomly in a sphere around the Rx
            %   similar to the one-ring model
            posCtr = prm.posRx(:,uIdx);  % X,Y,Z positions
            % force the scatters to be within 10cm from the possition of
            % the receiver array
            radCtr = prm.mobileRanges(uIdx)*0.1;  
            scatBound = [posCtr(1)-radCtr posCtr(1)+radCtr; ...
                         posCtr(2)-radCtr posCtr(2)+radCtr; ...
                         posCtr(3)-radCtr posCtr(3)+radCtr];
                       
            % Channel
            chan{uIdx} = phased.ScatteringMIMOChannel(...
                'TransmitArray',txarray,...
                'ReceiveArray',rxarray,...
                'PropagationSpeed',prm.cLight,...
                'CarrierFrequency',prm.fc,...
                'SampleRate',prm.chanSRate, ...
                'SimulateDirectPath',false, ...
                'ChannelResponseOutputPort',true, ...
                'TransmitArrayPosition',prm.posTx,...
                'ReceiveArrayPosition',prm.posRx(:,uIdx),...
                'NumScatterers',Ns, ...
                'ScattererPositionBoundary',scatBound, ...
                'SeedSource','Property', ...
                'Seed',prm.seed_p{uIdx}(pktID));

            maxBytes = 1e9;
             if numTx*numRx(uIdx)*Ns*(length(sigPad)) ...
                    *numBytesPerElement > maxBytes
                % If requested sizes are too large, process symbol-wise
                fadeSig = complex(zeros(length(sigPad), numRx(uIdx)));
                symLen = prm.FFTLength+prm.CyclicPrefixLength;
                numSymb = ceil(length(sigPad)/symLen);
                for idx = 1:numSymb
                    sIdx = (idx-1)*symLen+(1:symLen).';
                    [tmp, CR, tau] = chan{uIdx}(sigPad(sIdx,:));
                    CH_RESPONSE(uIdx,:,:,:) = CR;
                    TAU(uIdx,:) = tau;
                    fadeSig(sIdx,:) = tmp;
                end
            else                       
                [fadeSig, CR, tau] = chan{uIdx}(sigPad);
                CH_RESPONSE(uIdx,:,:,:) = CR;
                TAU(uIdx,:) = tau;
            end
            
            % Compute channel Delay (Time of Flight) to the first
            % reflection (this should map the LoS path). Return samples
            chanDelay(uIdx) = floor(min(tau)*prm.chanSRate);  % in samples
            
            % Remove the preamble, if present
            if ~isempty(preSig)
                fadeSig(1:(length(preSig)+prm.numPadZeros),:) = [];
            end

            % Path loss is included in channel
            lossSig{uIdx} = fadeSig;
            
        end
        
    case 'MIMO'

        % Create independent channels per user
        for uIdx = 1:numUsers

            % Using comm.MIMOChannel, with no array information
            chan{uIdx} = comm.MIMOChannel('MaximumDopplerShift',0, ...
                'SpatialCorrelation',false, ...
                'NumTransmitAntennas',numTx, ...
                'NumReceiveAntennas',numRx(uIdx),...
                'RandomStream','mt19937ar with seed', ...
                'Seed',prm.seed_p{uIdx}(pktID), ...
                'SampleRate',prm.chanSRate);

            maxBytes = 8e9;
            if numTx*numRx*(length(sigPad))*numBytesPerElement > maxBytes
                % If requested sizes are too large, process symbol-wise
                fadeSig = complex(zeros(length(sigPad), numRx));
                symLen = prm.FFTLength+prm.CyclicPrefixLength;
                numSymb = ceil(length(sigPad)/symLen);
                for idx = 1:numSymb
                    sIdx = (idx-1)*symLen+(1:symLen).';
                    fadeSig(sIdx,:) = chan{uIdx}(sigPad(sIdx,:));
                end
            else
                fadeSig = chan{uIdx}(sigPad);
            end

            % Check derived channel parameters
            chanInfo = info(chan{uIdx});
            chanDelay(uIdx) = chanInfo.ChannelFilterDelay;        

            % Remove the preamble, if present
            if ~isempty(preSig)
                fadeSig(1:(length(preSig)+prm.numPadZeros),:) = [];
            end
            
            % Apply path loss
            lossSig{uIdx} = fadeSig/sqrt(db2pow(spLoss(uIdx)));
            
        end
    otherwise
        error(message('comm_demos:helperApplyMUChannel:invChanType'));
end

end

% [EOF]
