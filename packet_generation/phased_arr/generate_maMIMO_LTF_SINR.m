function [mean_ber] = generate_maMIMO_LTF(exp_ID, numPackets, numBSTx, numUERx, snr_dB_CS, isOnlyCSI, saveFlag, isPlotting, seed_p, isMMSE)

N_chan_taps = 100;
applyInterf = true;
I=-55; % interference (noise) power in dBm

%% Define system parameters for the example. Modify these parameters to explore their impact on the system.

% Multi-user system with single/multiple streams per user
% prm.numUsers = 4;                 % Number of users
% prm.numSTSVec = [1 1 1 1];        % Number of independent data streams per user
% prm.numSTS = sum(prm.numSTSVec);  % Must be a power of 2
% prm.numTx = prm.numSTS*8;         % Number of BS transmit antennas (power of 2)
% prm.numRx = prm.numSTSVec*4;      % Number of receive antennas, per user (any >= numSTSVec)

% THIS IS TO TEST SINGLE USER ENVIRONMENT
prm.numUsers = 1;                 % Number of users
prm.numSTSVec = [1];        % Number of independent data streams per user
prm.numSTS = sum(prm.numSTSVec);  % Must be a power of 2
prm.numTx = numBSTx;         % Number of BS transmit antennas (power of 2)
prm.numRx = prm.numSTSVec*numUERx;      % Number of receive antennas, per user (any >= numSTSVec)


% Each user has the same modulation
prm.bitsPerSubCarrier = 2;   % 2: QPSK, 4: 16QAM, 6: 64QAM, 8: 256QAM
prm.numDataSymbols = 10;     % Number of OFDM data symbols

seed_range = 10000000; % random integers will be sampled in the interval [1, seed_range]
prm.seed_p = cell(prm.numUsers,1);
if isempty(seed_p)
    s = rng(67); % after generating random seeds for channels, we want everything else to stay the same
    %rng shuffle;
    for uIdx=1:prm.numUsers
        prm.seed_p{uIdx} = randi(seed_range, numPackets, 1); % used to keep track of seeds used while generating channels
        %prm.seed_p{uIdx} = [uIdx]; % this will replicate the original example
    end
else
    for uIdx=1:prm.numUsers
        prm.seed_p{uIdx} = seed_p{uIdx}(1:numPackets,1);
    end
end
% replicate the 1 pkt experiment
% prm.seed_p{1} = 6570466;
% prm.seed_p{2} = 2962020;
% prm.seed_p{3} = 5463299;
% prm.seed_p{4} = 8475627;

display(prm.seed_p{1})

s = rng(67); % after generating random seeds for channels, we want everything else to stay the same


% MS positions: assumes BS at origin
%   Angles specified as [azimuth;elevation] degrees
%   az in range [-180 180], el in range [-90 90], e.g. [45;0]
maxRange = 1000;            % all MSs within 1000 meters of BS
prm.mobileRanges = randi([1 maxRange],1,prm.numUsers);
prm.mobileAngles = [rand(1,prm.numUsers)*360-180; ...
                    rand(1,prm.numUsers)*180-90];
disp(prm.mobileRanges)
disp(prm.mobileAngles)
                
                
prm.fc = 28e9;               % 28 GHz system
prm.chanSRate = 100e6;       % Channel sampling rate, 100 Msps
prm.ChanType = 'Scattering'; % Channel options: 'Scattering', 'MIMO'
prm.NFig = 8;                % Noise figure (increase to worsen, 5-10 dB)
prm.nRays = 500;             % Number of rays for Frf, Fbb partitioning

%% Define OFDM modulation parameters used for the system.

prm.FFTLength = 256;
prm.CyclicPrefixLength = 64;
prm.numCarriers = 234;
prm.NullCarrierIndices = [1:7 129 256-5:256]'; % Guards and DC
prm.PilotCarrierIndices = [26 54 90 118 140 168 204 232]';
nonDataIdx = [prm.NullCarrierIndices; prm.PilotCarrierIndices];
prm.CarriersLocations = setdiff((1:prm.FFTLength)', sort(nonDataIdx));

numSTS = prm.numSTS;
numTx = prm.numTx;
numRx = prm.numRx;
numSTSVec = prm.numSTSVec;
codeRate = 1/3;             % same code rate per user
numTails = 6;               % number of termination tail bits
prm.numFrmBits = numSTSVec.*(prm.numDataSymbols*prm.numCarriers* ...
                 prm.bitsPerSubCarrier*codeRate)-numTails;
prm.modMode = 2^prm.bitsPerSubCarrier; % Modulation order
% Account for channel filter delay
numPadSym = 3;          % number of symbols to zeropad
prm.numPadZeros = numPadSym*(prm.FFTLength+prm.CyclicPrefixLength);

%% Define transmit and receive arrays and positional parameters for the system.

prm.cLight = physconst('LightSpeed');
prm.lambda = prm.cLight/prm.fc;

% Get transmit and receive array information
[isTxURA,expFactorTx,isRxURA,expFactorRx] = helperArrayInfo(prm,true);

% Transmit antenna array definition
%   Array locations and angles
prm.posTx = [0;0;0];       % BS/Transmit array position, [x;y;z], meters
if isTxURA
    % Uniform Rectangular array
    txarray = phased.PartitionedArray(...
        'Array',phased.URA([expFactorTx numSTS],0.5*prm.lambda),...
        'SubarraySelection',ones(numSTS,numTx),'SubarraySteering','Custom');
else
    % Uniform Linear array
    txarray = phased.ULA(numTx, 'ElementSpacing',0.5*prm.lambda, ...
        'Element',phased.IsotropicAntennaElement('BackBaffled',false));
end
prm.posTxElem = getElementPosition(txarray)/prm.lambda;

spLoss = zeros(prm.numUsers,1);
prm.posRx = zeros(3,prm.numUsers);
for uIdx = 1:prm.numUsers

    % Receive arrays
    if isRxURA(uIdx)
        % Uniform Rectangular array
        rxarray = phased.PartitionedArray(...
            'Array',phased.URA([expFactorRx(uIdx) numSTSVec(uIdx)], ...
            0.5*prm.lambda),'SubarraySelection',ones(numSTSVec(uIdx), ...
            numRx(uIdx)),'SubarraySteering','Custom');
        prm.posRxElem = getElementPosition(rxarray)/prm.lambda;
    else
        if numRx(uIdx)>1
            % Uniform Linear array
            rxarray = phased.ULA(numRx(uIdx), ...
                'ElementSpacing',0.5*prm.lambda, ...
                'Element',phased.IsotropicAntennaElement);
            prm.posRxElem = getElementPosition(rxarray)/prm.lambda;
        else
            rxarray = phased.IsotropicAntennaElement;
            prm.posRxElem = [0; 0; 0]; % LCS
        end
    end

    % Mobile positions
    [xRx,yRx,zRx] = sph2cart(deg2rad(prm.mobileAngles(1,uIdx)), ...
                             deg2rad(prm.mobileAngles(2,uIdx)), ...
                             prm.mobileRanges(uIdx));
    prm.posRx(:,uIdx) = [xRx;yRx;zRx];
    [toRxRange,toRxAng] = rangeangle(prm.posTx,prm.posRx(:,uIdx));
    spLoss(uIdx) = fspl(toRxRange,prm.lambda);
end


%% Create output data structures
usr_data = cell(prm.numUsers,3);    % cell used to store different dataset transmissions for every user

for nUser=1:prm.numUsers
    % the following list are initialized when the first packet is processed
    usr_data{nUser,1} = []; % list of preambles (one per transmission)
    usr_data{nUser,2} = []; % list of CSI computed with LS at Channel Sounding time
    usr_data{nUser,3} = []; % list of SNR level at Channel Sounding time
    if isMMSE
        usr_data{nUser,4} = []; % list of CSI computed with LMMSE at Channel Sounding time
    end
    if ~isOnlyCSI
        usr_data{nUser,5} = zeros(numPackets, 1); % list of BER (used to compare performances)
    end
    
    
end

%% For a spatially multiplexed system, availability of channel information at the transmitter allows for precoding to be applied to maximize the signal energy in the direction and channel of interest. Under the assumption of a slowly varying channel, this is facilitated by sounding the channel first. The BS sounds the channel by using a reference transmission, that the MS receiver uses to estimate the channel. The MS transmits the channel estimate information back to the BS for calculation of the precoding needed for the subsequent data transmission.
%outDebug = cell(numPackets,1);

for p=1:numPackets
    disp(strcat('------------- Packet ',int2str(p),' -------------'));
    
    % Generate the preamble signal
    prm.numSTS = numTx;             % set to numTx to sound out all channels
    preambleSig = helperGenPreamble(prm);
    %outDebug{p}.preambleSig = preambleSig;
    if isPlotting
        close all;
    end
    
    % Transmit preamble over channel
    prm.numSTS = numSTS;            % keep same array config for channel
    [rxPreSig,chanDelay,h_tau,h_response] = helperApplyMUChannel(preambleSig,prm,spLoss,N_chan_taps,p);
    %outDebug{p}.rxPreSig = rxPreSig; outDebug{p}.chanDelay = chanDelay;
    % Channel state information feedback
    hDp = cell(prm.numUsers,1);
    hDmmse = cell(prm.numUsers,1);
    prm.numSTS = numTx;             % set to numTx to estimate all links
    
    % Compute Power offered by channel
    P_ch = zeros(numTx,numRx);
    for tx_idx = 1:numTx
        for rx_idx = 1:numRx
            t = h_response(uIdx,tx_idx,rx_idx,:);
            t = t(:);
            P_ch(tx_idx,rx_idx) = t'*t;
        end
    end
    P_ch_dB = pow2db(P_ch);
    
    %outDebug{p}.inputRXSig = cell(prm.numUsers, 1);
    %profile on

    %noise_dB = 40;
    
    for uIdx = 1:prm.numUsers
        
        
        
        %gain_dB = spLoss(uIdx); % gain used in the amplifier is equal to path loss from BS to user's position
        gain_dB =0;
        
        %sigPwr = mean(abs(rxPreSig{uIdx}).^2);   % compute the signal power in Watts
        sigPwr = rms(rxPreSig{uIdx}).^2;
        sig_dB = pow2db(sigPwr); % power in dBW
        sig_dBm = pow2db(sigPwr)+30; %power in dBm
        % snr_dB_CS = sig_dB + gain_dB - noise_dB;       % general SNR formula in dB scale
        noise_dB = sig_dB + gain_dB - snr_dB_CS ;   
        noise_dBm = sig_dBm + gain_dB - snr_dB_CS;  
        noise_dBm = mean(noise_dBm);  % average noise over receiver antennas
        snr_dB_CS = sig_dBm + gain_dB - noise_dBm;
         
        baseNoise_dBm = -85;
        
        P_TX = rms(preambleSig).^2; 
        P_TX_dBm = pow2db(P_TX) + 30;  % individial Power per sequence
        snr_dB_CS2 = P_TX_dBm + (-spLoss(uIdx) + pow2db(N_chan_taps)) - baseNoise_dBm;
        snr_dB_CS = snr_dB_CS2(1:prm.numRx);
        noise_dBm = baseNoise_dBm;
        
        
        baseNoise_W = db2pow(baseNoise_dBm-30);
        sumNoise_W = baseNoise_W + db2pow(I-30);
        sumNoise_dBm = pow2db(sumNoise_W)+30;
        
        SINR = P_TX_dBm + (-spLoss(uIdx) + pow2db(N_chan_taps)) - (sumNoise_dBm); % (Ps*ch.gain)/(N+I)
        
        
        if applyInterf
            % TODO: try the normal distribution noise sampling
            %rxPreSig{uIdx} = rxPreSig{uIdx} + wgn(size(rxPreSig{uIdx},1),size(rxPreSig{uIdx},2),I-30,'complex');
            N = size(rxPreSig{uIdx},1);
            s_n = randn(N,prm.numRx) + 1i.*randn(N,prm.numRx);  % noise
            %std = sqrt(db2pow(I-30));
            std = sqrt(sumNoise_W);
            
            rxPreSig{uIdx} = rxPreSig{uIdx} + (std .* s_n);
            
        end
        
        % Front-end amplifier gain and thermal noise
%         rxPreAmp = phased.ReceiverPreamp( ...
%             'Gain',gain_dB, ...    % account for path loss
%             'NoiseMethod', 'Noise power', ...
%             'NoisePower', db2pow(noise_dBm-30));
%             %'NoiseFigure',prm.NFig,'ReferenceTemperature',290, 'SampleRate',prm.chanSRate, ...
%         
%         rxPreSigAmp = rxPreAmp(rxPreSig{uIdx}); 
        rxPreSigAmp = rxPreSig{uIdx};

        %   scale power for used sub-carriers
        rxPreSigAmp = rxPreSigAmp * (sqrt(prm.FFTLength - ...
            length(prm.NullCarrierIndices))/prm.FFTLength);
        
        disp(strcat("CHANNEL SOUNDING - User ", num2str(uIdx), " SNR: ", num2str(snr_dB_CS)));
        disp(strcat("w/ Interference - User ", num2str(uIdx), " SINR: ", num2str(SINR)));
        
        inputRXSig = rxPreSigAmp(chanDelay(uIdx)+1: ...
            end-(prm.numPadZeros-chanDelay(uIdx)),:);

        % OFDM demodulation
        rxOFDM = ofdmdemod(inputRXSig,prm.FFTLength, ...
            prm.CyclicPrefixLength,prm.CyclicPrefixLength, ...
            prm.NullCarrierIndices,prm.PilotCarrierIndices);

        % Channel estimation from preamble
        %       numCarr, numTx, numRx
        [hDp{uIdx},P,~,hDmmse{uIdx}] = helperMIMOChannelEstimate(rxOFDM(:,1:numTx,:),prm,1,h_tau(uIdx,:),snr_dB_CS, isMMSE);
        
        % ----------------------- 
        % saving output data
        if isempty(usr_data{uIdx,1})
             usr_data{uIdx,1} = zeros(numPackets, size(inputRXSig,1),size(inputRXSig,2), 'like', inputRXSig);
        end
        usr_data{uIdx,1}(p,:,:) = inputRXSig;
        
        if isempty(usr_data{uIdx,2})
            usr_data{uIdx,2} = zeros(numPackets, size(hDp{uIdx},1),size(hDp{uIdx},2),size(hDp{uIdx},3), 'like', hDp{uIdx});
        end
        usr_data{uIdx,2}(p,:,:,:) = hDp{uIdx};
        
        if isempty(usr_data{uIdx,3})
            usr_data{uIdx,3} = zeros(numPackets, size(snr_dB_CS,1),size(snr_dB_CS,2), 'like', snr_dB_CS);
        end
        usr_data{uIdx,3}(p,:,:) = snr_dB_CS;
        
        if isMMSE
            if isempty(usr_data{uIdx,4})
                usr_data{uIdx,4} = zeros(numPackets, size(hDmmse{uIdx},1),size(hDmmse{uIdx},2),size(hDmmse{uIdx},3), 'like', hDmmse{uIdx});
            end
            usr_data{uIdx,4}(p,:,:,:) = hDmmse{uIdx};
        end
        % timing (to be done only once)
%         if p == 1
%             f1_ofdmdem = @()ofdmdemod(inputRXSig,prm.FFTLength, ...
%             prm.CyclicPrefixLength,prm.CyclicPrefixLength, ...
%             prm.NullCarrierIndices,prm.PilotCarrierIndices);
%             
%             f2_chnEst = @() helperMIMOChannelEstimate(rxOFDM(:,1:numTx,:),prm,1,h,snr_dB_CS/10);
%             
%             timing = [];
%             for i=1:1000
%                 timing(i) = timeit(f1_ofdmdem) + timeit(f2_chnEst);
%             end
%             outDebug.timing = mean(timing)
%         end
        
        % ----------------------- 
        
    end
    %profile off
    
    for estSource=1:2   % 1 = LS est.; 2 = MMSE est.;
        if estSource == 2
            disp('MMSE CSI estimation');
            hDp = hDmmse;
        else
            disp('LS CSI estimation');
        end

        if ~isOnlyCSI
            %% The example uses the orthogonal matching pursuit (OMP) algorithm [ 3 ] for a single-user system and the joint spatial division multiplexing (JSDM) technique [ 2, 4 ] for a multi-user system, to determine the digital baseband Fbb and RF analog Frf precoding weights for the selected system configuration.
            % For a single-user system, the OMP partitioning algorithm is sensitive to the array response vectors At. Ideally, these response vectors account for all the scatterers seen by the channel, but these are unknown for an actual system and channel realization, so a random set of rays within a 3-dimensional space to cover as many scatterers as possible is used. The prm.nRays parameter specifies the number of rays.
            % For a multi-user system, JSDM groups users with similar transmit channel covariance together and suppresses the inter-group interference by an analog precoder based on the block diagonalization method [ 5 ]. Here each user is assigned to be in its own group, thereby leading to no reduction in the sounding or feedback overhead.

            % Calculate the hybrid weights on the transmit side
            if prm.numUsers==1
                % Single-user OMP
                %   Spread rays in [az;el]=[-180:180;-90:90] 3D space, equal spacing
                %   txang = [-180:360/prm.nRays:180; -90:180/prm.nRays:90];
                txang = [rand(1,prm.nRays)*360-180;rand(1,prm.nRays)*180-90]; % random
                At = steervec(prm.posTxElem,txang);
                AtExp = complex(zeros(prm.numCarriers,size(At,1),size(At,2)));
                for carrIdx = 1:prm.numCarriers
                    AtExp(carrIdx,:,:) = At; % same for all sub-carriers
                end

                % Orthogonal matching pursuit hybrid weights
                [Fbb,Frf] = omphybweights(hDp{1},numSTS,numSTS,AtExp);

                v = Fbb;    % set the baseband precoder (Fbb)
                % Frf is same across subcarriers for flat channels
                mFrf = permute(mean(Frf,1),[2 3 1]);

            else
                % Multi-user Joint Spatial Division Multiplexing
                [Fbb,mFrf] = helperJSDMTransmitWeights(hDp,prm);
                % Multi-user baseband precoding
                %   Pack the per user CSI into a matrix (block diagonal)
                steeringMatrix = zeros(prm.numCarriers,sum(numSTSVec),sum(numSTSVec));
                for uIdx = 1:prm.numUsers
                    stsIdx = sum(numSTSVec(1:uIdx-1))+(1:numSTSVec(uIdx));
                    steeringMatrix(:,stsIdx,stsIdx) = Fbb{uIdx};  % Nst-by-Nsts-by-Nsts
                end
                v = permute(steeringMatrix,[1 3 2]);
                outDebug{p}.v = v;
            end

            if isPlotting
                figure(12+estSource)
                % Transmit array pattern plots
                if isTxURA
                    % URA element response for the first subcarrier
                    pattern(txarray,prm.fc,-180:180,-90:90,'Type','efield', ...
                            'ElementWeights',mFrf.'*squeeze(v(1,:,:)), ...
                            'PropagationSpeed',prm.cLight);
                else % ULA
                    % Array response for first subcarrier
                    wts = mFrf.'*squeeze(v(1,:,:));
                    pattern(txarray,prm.fc,-180:180,-90:90,'Type','efield', ...
                            'Weights',wts(:,1),'PropagationSpeed',prm.cLight);
                end
            end
            prm.numSTS = numSTS;                 % revert back for data transmission

            %% Next, we configure the system's data transmitter. This processing includes channel coding, bit mapping to complex symbols, splitting of the individual data stream to multiple transmit streams, baseband precoding of the transmit streams, OFDM modulation with pilot mapping and RF analog beamforming for all the transmit antennas employed.

            % Convolutional encoder
            encoder = comm.ConvolutionalEncoder( ...
                'TrellisStructure',poly2trellis(7,[133 171 165]), ...
                'TerminationMethod','Terminated');

            txDataBits = cell(prm.numUsers, 1);
            gridData = complex(zeros(prm.numCarriers,prm.numDataSymbols,numSTS));
            for uIdx = 1:prm.numUsers
                % Generate mapped symbols from bits per user
                txDataBits{uIdx} = randi([0,1],prm.numFrmBits(uIdx),1);
                encodedBits = encoder(txDataBits{uIdx});

                % Bits to QAM symbol mapping
                mappedSym = qammod(encodedBits,prm.modMode,'InputType','bit', ...
                'UnitAveragePower',true);

                % Map to layers: per user, per symbol, per data stream
                stsIdx = sum(numSTSVec(1:(uIdx-1)))+(1:numSTSVec(uIdx));
                gridData(:,:,stsIdx) = reshape(mappedSym,prm.numCarriers, ...
                    prm.numDataSymbols,numSTSVec(uIdx));
            end

            % Apply precoding weights to the subcarriers, assuming perfect feedback
            preData = complex(zeros(prm.numCarriers,prm.numDataSymbols,numSTS));
            for symIdx = 1:prm.numDataSymbols
                for carrIdx = 1:prm.numCarriers
                    Q = squeeze(v(carrIdx,:,:));
                    normQ = Q * sqrt(numTx)/norm(Q,'fro');
                    preData(carrIdx,symIdx,:) = squeeze(gridData(carrIdx,symIdx,:)).' ...
                        * normQ;
                end
            end

            % Multi-antenna pilots
            pilots = helperGenPilots(prm.numDataSymbols,numSTS);

            % OFDM modulation of the data
            txOFDM = ofdmmod(preData,prm.FFTLength,prm.CyclicPrefixLength,...
                             prm.NullCarrierIndices,prm.PilotCarrierIndices,pilots);
            %   scale power for used sub-carriers
            txOFDM = txOFDM * (prm.FFTLength/ ...
                sqrt((prm.FFTLength-length(prm.NullCarrierIndices))));

            % Generate preamble with the feedback weights and prepend to data
            preambleSigD = helperGenPreamble(prm,v);
            txSigSTS = [preambleSigD;txOFDM];

            % RF beamforming: Apply Frf to the digital signal
            %   Each antenna element is connected to each data stream
            txSig = txSigSTS*mFrf;

            %% The example offers an option for spatial MIMO channel and a simpler static-flat MIMO channel for validation purposes.
            %The scattering model uses a single-bounce ray tracing approximation with a parametrized number of scatterers. For this example, the number of scatterers is set to 100. The 'Scattering' option models the scatterers placed randomly within a sphere around the receiver, similar to the one-ring model [ 6 ].
            %The channel models allow path-loss modeling and both line-of-sight (LOS) and non-LOS propagation conditions. The example assumes non-LOS propagation and isotropic antenna element patterns with linear or rectangular geometry.

            % Apply a spatially defined channel to the transmit signal
            [rxSig,chanDelay,h_tau,~] = helperApplyMUChannel(txSig, prm, spLoss, N_chan_taps, p, preambleSig);

            %% Receive Amplification and Signal Recovery
            % The receiver modeled per user compensates for the path loss by amplification and adds thermal noise. Like the transmitter, the receiver used in a MIMO-OFDM system contains many stages including OFDM demodulation, MIMO equalization, QAM demapping, and channel decoding.

            if isPlotting
                hfig = figure('Name','Equalized symbol constellation per stream');
            end

            scFact = ((prm.FFTLength-length(prm.NullCarrierIndices))...
                     /prm.FFTLength^2)/numTx;
            nVar = noisepow(prm.chanSRate,prm.NFig,290)/scFact;     % this might be overwritten if NoiseMethod property of Receiver preamp is 'Noise Power'
            decoder = comm.ViterbiDecoder('InputFormat','Unquantized', ...
                'TrellisStructure',poly2trellis(7, [133 171 165]), ...
                'TerminationMethod','Terminated','OutputDataType','double');

            for uIdx = 1:prm.numUsers
                stsU = numSTSVec(uIdx);
                stsIdx = sum(numSTSVec(1:(uIdx-1)))+(1:stsU);

%                 gain_dB = spLoss(uIdx); % gain used in the amplifier is equal to path loss from BS to user's position

                %sigPwr = mean(abs(rxSig{uIdx}).^2) ;   % compute the signal power in Watts
                sigPwr = rms(rxSig{uIdx}).^2;
                sig_dBm = pow2db(sigPwr)+30;
                snr_dB_DT = sig_dBm + gain_dB - noise_dBm;
                nVar = db2pow(noise_dBm-30);
                
                
                P_TX = rms(txSig).^2; 
                P_TX_dBm = pow2db(P_TX) + 30;  % individial Power per sequence
                snr_dB_DT = P_TX_dBm + (-spLoss(uIdx) + pow2db(N_chan_taps)) - baseNoise_dBm;
                snr_dB_DT = mean(snr_dB_DT);
                
                SINR = P_TX_dBm + (-spLoss(uIdx) + pow2db(N_chan_taps)) - (sumNoise_dBm); % (Ps*ch.gain)/(N+I)
                
                
                if applyInterf
                    rxSig{uIdx} = rxSig{uIdx} + wgn(size(rxSig{uIdx},1),size(rxSig{uIdx},2),I-30,'complex');
                end
                
                % Front-end amplifier gain and thermal noise
                rxPreAmp = phased.ReceiverPreamp( ...
                    'Gain',gain_dB, ...    % account for path loss
                    'NoiseMethod', 'Noise power', ...
                    'NoisePower', db2pow(noise_dBm-30));
                    %'NoiseFigure',prm.NFig,'ReferenceTemperature',290, 'SampleRate',prm.chanSRate, ...
                rxSigAmp = rxPreAmp(rxSig{uIdx});


                % Scale power for occupied sub-carriers
                rxSigAmp = rxSigAmp*(sqrt(prm.FFTLength-length(prm.NullCarrierIndices)) ...
                    /prm.FFTLength);

    %             N = size(rxSig{uIdx},1);
    %             s_n = randn(N,prm.numRx) + 1i.*randn(N,prm.numRx);  % noise
    %             % The power of a signal is the sum of the 
    %             % absolute squares of its time-domain samples 
    %             % divided by the signal length, or, equivalently, 
    %             % the square of its RMS level
    %             P_N =rms(s_n).^2;
    %             P_CW = rms(rxSig{uIdx}).^2;
    %             %noise_var = P_CW/(P_N*db2pow(snr_dB_CS));  % reuse the
    %                                                         % variance used
    %                                                         % during channel
    %                                                         % sounding
    %             s_n_adj = sqrt(noise_var) .* s_n;
    %             sig_plus_noise = rxSig{uIdx} + s_n_adj;
    %             rxSigAmp = sig_plus_noise;
    %             nVar = noise_var;
    %             noise_db = 10*log10(rms(s_n_adj).^2);
    %             snr_dB_DT = P_CW - noise_db;

                disp(strcat("DATA TRANSM - User ", num2str(uIdx), " SNR: ", num2str(snr_dB_DT)));
                disp(strcat("w/ Interference - User ", num2str(uIdx), " SINR: ", num2str(SINR)));
                
                % OFDM demodulation
                rxOFDM = ofdmdemod(rxSigAmp(chanDelay(uIdx)+1: ...
                    end-(prm.numPadZeros-chanDelay(uIdx)),:),prm.FFTLength, ...
                    prm.CyclicPrefixLength,prm.CyclicPrefixLength, ...
                    prm.NullCarrierIndices,prm.PilotCarrierIndices);

                % Channel estimation from the mapped preamble
                [hD,~,~,~] = helperMIMOChannelEstimate(rxOFDM(:,1:numSTS,:),prm,1,h_tau(uIdx,:),snr_dB_DT,false);

                % MIMO equalization
                %   Index into streams for the user of interest
                [rxEq,CSI] = helperMIMOEqualize(rxOFDM(:,numSTS+1:end,:),hD(:,stsIdx,:));

                % Soft demodulation
                rxSymbs = rxEq(:)/sqrt(numTx);
                rxLLRBits = qamdemod(rxSymbs,prm.modMode,'UnitAveragePower',true, ...
                    'OutputType','approxllr','NoiseVariance',nVar);

                % Apply CSI prior to decoding
                rxLLRtmp = reshape(rxLLRBits,prm.bitsPerSubCarrier,[], ...
                    prm.numDataSymbols,stsU);
                csitmp = reshape(CSI,1,[],1,numSTSVec(uIdx));
                rxScaledLLR = rxLLRtmp.*csitmp;

                % Soft-input channel decoding
                rxDecoded = decoder(rxScaledLLR(:));

                % Decoded received bits
                rxBits = rxDecoded(1:prm.numFrmBits(uIdx));

                if isPlotting
                    % Plot equalized symbols for all streams per user
                    scaler = ceil(max(abs([real(rxSymbs(:)); imag(rxSymbs(:))])));
                    for i = 1:stsU
                        subplot(prm.numUsers, max(numSTSVec), (uIdx-1)*max(numSTSVec)+i);
                        plot(reshape(rxEq(:,:,i)/sqrt(numTx), [], 1), '.');
                        axis square
                        xlim(gca,[-scaler scaler]);
                        ylim(gca,[-scaler scaler]);
                        title(['U ' num2str(uIdx) ', DS ' num2str(i)]);
                        grid on;
                    end
                end

                % Compute and display the EVM
                evm = comm.EVM('Normalization','Average constellation power', ...
                    'ReferenceSignalSource','Estimated from reference constellation', ...
                    'ReferenceConstellation', ...
                    qammod((0:prm.modMode-1)',prm.modMode,'UnitAveragePower',1));
                rmsEVM = evm(rxSymbs);
                
                disp(['User ' num2str(uIdx)]);
                disp(['  RMS EVM (%) = ' num2str(rmsEVM)]);

                % Compute and display bit error rate
                ber = comm.ErrorRate;
                measures = ber(txDataBits{uIdx},rxBits);
                fprintf('  BER = %.5f; No. of Bits = %d; No. of errors = %d\n', ...
                    measures(1),measures(3),measures(2));
                % save ber 
                if estSource == 1
                    % save only LS estimation BER as baseline
                    usr_data{uIdx,5}(p) = measures(1); 
                end
            end
        end
        if isPlotting
            for i=1:prm.numUsers
    %                 figure(10+i);
    %                 imagesc(abs(hDp{i}(:,:,1)));
                plot_mimo_channel(hDp{i},prm.numRx(i),10*estSource+i);
            end
        end
        
    end
end

mean_ber = mean(usr_data{1,5});

if saveFlag
    if ~exist('packets', 'dir')
       mkdir('packets')
    end
    save(strcat('packets/SNRanalysis/','maMIMO_',num2str(numPackets),'___',exp_ID,'.mat'),'usr_data','P','prm','mean_ber','-v7.3');
end


rng(s);         % restore RNG state
end