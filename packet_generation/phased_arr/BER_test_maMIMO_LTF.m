function [metrics] = BER_test_maMIMO_LTF(txs_file, n_packets, pkt_test_ratio, real_csi_pred, imag_csi_pred, isPlotting, isSeparateFiles,snr_dB_CS)

n_test_packets = int32(floor(n_packets * pkt_test_ratio));

[usr_data,prm] = extract_pkt(txs_file,n_test_packets,false,"",true);

[numPackets, lenIn, nRXAnts] = size(usr_data{1,1});
[totPkt, nSubCar, nTXAnts, nRXAnts ] = size(usr_data{1,2});

if ~isSeparateFiles
    load(real_csi_pred);
    CSI_nn_predREAL = all_pkts_csi_nn_out.y;
    inputLTF_REAL = all_pkts_csi_nn_out.x; 
    inputLTF_REAL = inputLTF_REAL(:,1:lenIn);   % we remove the pilot sequence
    load(imag_csi_pred);
    CSI_nn_predIMAG = all_pkts_csi_nn_out.y;
    inputLTF_IMAG = all_pkts_csi_nn_out.x;
    inputLTF_IMAG = inputLTF_IMAG(:,1:lenIn);
end

isMMSE = true;
N_chan_taps = 100;

%% Define system parameters for the example. Modify these parameters to explore their impact on the system.
% NOTE: prm with following parameters will be loaded from the txs_file

% Multi-user system with single/multiple streams per user
%prm.numUsers = 4;                 % Number of users
%prm.numSTSVec = [1 1 1 1];        % Number of independent data streams per user
%prm.numSTS = sum(prm.numSTSVec);  % Must be a power of 2
%prm.numTx = prm.numSTS*8;         % Number of BS transmit antennas (power of 2)
%prm.numRx = prm.numSTSVec*4;      % Number of receive antennas, per user (any >= numSTSVec)

% Each user has the same modulation
prm.bitsPerSubCarrier = 2;   % 2: QPSK, 4: 16QAM, 6: 64QAM, 8: 256QAM
prm.numDataSymbols = 10;     % Number of OFDM data symbols

% prm.seed_p will be loaded from the input txs_file .mat file
%display(prm.seed_p{1})
s = rng(67); % we want everything else to stay the same


% MS positions: assumes BS at origin
%   Angles specified as [azimuth;elevation] degrees
%   az in range [-180 180], el in range [-90 90], e.g. [45;0]
% maxRange = 1000;            % all MSs within 1000 meters of BS
% prm.mobileRanges = randi([1 maxRange],1,prm.numUsers);
% prm.mobileAngles = [rand(1,prm.numUsers)*360-180; ...
%                     rand(1,prm.numUsers)*180-90];

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

% for nUser=1:prm.numUsers
%     %prm.seed_p{nUser} = seed_p{nUser,1}; % at TESTING TIME we load the seeds to generate channels, in order to generate the same channel instances
%     % the following list are initialized when the first packet is processed
%     %usr_data{nUser,1} = []; % list of preambles (one per transmission)
%     %usr_data{nUser,2} = []; % list of CSI (one per transmission)
%     usr_data{nUser,3} = zeros(numPackets, 1); % list of BER for LS estimation
%     usr_data{nUser,4} = zeros(numPackets, 1); % list of BER for DNN
% end

bers_LS = zeros(numPackets, 1);
EVM_rms_LS = zeros(numPackets, 1);
MSE_LS = zeros(numPackets, 1);
dtSNR_LS = zeros(numPackets,1);

bers_MMSE = zeros(numPackets, 1);
EVM_rms_MMSE = zeros(numPackets, 1);
MSE_MMSE = zeros(numPackets, 1);
dtSNR_MMSE = zeros(numPackets,1);

bers_DNN = zeros(numPackets, 1);
EVM_rms_DNN = zeros(numPackets, 1);
MSE_DNN = zeros(numPackets, 1);
dtSNR_DNN = zeros(numPackets,1);

bers_perfect = zeros(numPackets, 1);
EVM_rms_perfect = zeros(numPackets, 1);
dtSNR_perfect = zeros(numPackets, 1);

%% For a spatially multiplexed system, availability of channel information at the transmitter allows for precoding to be applied to maximize the signal energy in the direction and channel of interest. Under the assumption of a slowly varying channel, this is facilitated by sounding the channel first. The BS sounds the channel by using a reference transmission, that the MS receiver uses to estimate the channel. The MS transmits the channel estimate information back to the BS for calculation of the precoding needed for the subsequent data transmission.

for p=1:numPackets
    disp(strcat('------------- Packet ',int2str(p),' -------------'));
    
    
    
    CSI_dnn_real = zeros(nSubCar,nTXAnts,nRXAnts);
    CSI_dnn_imag = zeros(nSubCar,nTXAnts,nRXAnts);
    
    if ~isSeparateFiles
        curr_p = p;

        predicted_CSI_real = CSI_nn_predREAL(((curr_p-1)*nRXAnts*nTXAnts)+1:((curr_p-1)*nRXAnts*nTXAnts)+nRXAnts*nTXAnts, :);
        predicted_CSI_imag = CSI_nn_predIMAG(((curr_p-1)*nRXAnts*nTXAnts)+1:((curr_p-1)*nRXAnts*nTXAnts)+nRXAnts*nTXAnts, :);

        for iRX = 1:nRXAnts
            for iTX = 1:nTXAnts
                CSI_dnn_real(:,iTX,iRX) = predicted_CSI_real((iRX-1)*(nTXAnts)+iTX,:);
                CSI_dnn_imag(:,iTX,iRX) = predicted_CSI_imag((iRX-1)*(nTXAnts)+iTX,:);
            end
        end
    else
        real_file_pkt = insertBefore(real_csi_pred, ".mat",strcat("_",num2str(p)));
        imag_file_pkt = insertBefore(imag_csi_pred, ".mat",strcat("_",num2str(p)));
        load(real_file_pkt);
        CSI_nn_predREAL = all_pkts_csi_nn_out.y;
        inputLTF_REAL = all_pkts_csi_nn_out.x; 
        inputLTF_REAL = inputLTF_REAL(:,1:lenIn);   % we remove the pilot sequence
        load(imag_file_pkt);
        CSI_nn_predIMAG = all_pkts_csi_nn_out.y;
        inputLTF_IMAG = all_pkts_csi_nn_out.x;
        inputLTF_IMAG = inputLTF_IMAG(:,1:lenIn);
        
        predicted_CSI_real = CSI_nn_predREAL;
        predicted_CSI_imag = CSI_nn_predIMAG;
        
        
        for iRX = 1:nRXAnts
            for iTX = 1:nTXAnts
                CSI_dnn_real(:,iTX,iRX) = predicted_CSI_real((iRX-1)*(nTXAnts)+iTX,:);
                CSI_dnn_imag(:,iTX,iRX) = predicted_CSI_imag((iRX-1)*(nTXAnts)+iTX,:);
            end
        end
        
    end
    
    
    CSI_dnn = complex(CSI_dnn_real,CSI_dnn_imag); % NOTE prediction is only for user 1 (12 RX antennas)!!
    
    % Generate the preamble signal
    prm.numSTS = numTx;             % set to numTx to sound out all channels
    preambleSig = helperGenPreamble(prm);
    
    if isPlotting
        close all;
    end
    
    % Transmit preamble over channel
    prm.numSTS = numSTS;            % keep same array config for channel
    [rxPreSig,chanDelay,h] = helperApplyMUChannel(preambleSig,prm,spLoss,N_chan_taps,p);

    % Channel state information feedback
    hDp = cell(prm.numUsers,1);
    hDpDNN = cell(prm.numUsers,1);
    hDp_real = cell(prm.numUsers,1);
    hDp_mmse = cell(prm.numUsers,1);
    prm.numSTS = numTx;             % set to numTx to estimate all links

    %% Channel estimation
    %profile on
    for uIdx = 1:prm.numUsers
        %% compute average noise
        gain_dB = spLoss(uIdx); % gain used in the amplifier is equal to path loss from BS to user's position
        
        %sigPwr = mean(abs(rxPreSig{uIdx}));   % compute the signal power in Watts
        sigPwr = rms(rxPreSig{uIdx}).^2;
        sig_dB = 10*log10(sigPwr);
        %snr_dB_CS = sig_dB - noise_dB + gain_dB;
        noise_dB = sig_dB - snr_dB_CS + gain_dB;
        noise_dB = mean(noise_dB);  % average noise over receiver antennas
        
        snr_dB_CS = sig_dB - noise_dB + gain_dB;
        disp(strcat("CHANNEL SOUNDING - User ", num2str(uIdx), " SNR: ", num2str(snr_dB_CS)));
        disp(strcat("Sig. pow (W) = ",num2str( sigPwr )));
        disp(strcat("Sig. pow (dBm) = ",num2str( sig_dB+30 )));
        disp(strcat("noise pow (dBm) Avg. = ",num2str( noise_dB+30) ));
        %% compute real channel 
        % for "perfect" CSI estimation, we apply very low noise in the
        % preamplifier. The preamp step is necessary to obtain similar power
        % level when comparing the CSI estimation and compute MSE
        
        % Front-end amplifier gain and thermal noise
        rxPreAmp_lowNoise = phased.ReceiverPreamp( ...
            'Gain',gain_dB, ...    % account for path loss
            'NoiseMethod', 'Noise power', ...
            'NoisePower', db2pow(-100));        
        
        rxPreSigAmp_real = rxPreAmp_lowNoise(rxPreSig{uIdx}); 
        %   scale power for used sub-carriers
        rxPreSigAmp_real = rxPreSigAmp_real * (sqrt(prm.FFTLength - ...
            length(prm.NullCarrierIndices))/prm.FFTLength);
        
        inputRXSig_real = rxPreSigAmp_real(chanDelay(uIdx)+1: ...
                 end-(prm.numPadZeros-chanDelay(uIdx)),:);
         % OFDM demodulation
        rxOFDM_real = ofdmdemod(inputRXSig_real,prm.FFTLength, ...
            prm.CyclicPrefixLength,prm.CyclicPrefixLength, ...
            prm.NullCarrierIndices,prm.PilotCarrierIndices);

        % Channel estimation from preamble
        %       numCarr, numTx, numRx
        % Using classic (LS) method
        [hDp_real{uIdx},~,~,~] = helperMIMOChannelEstimate(rxOFDM_real(:,1:numTx,:),prm,1,h,snr_dB_CS,false);
        
        
         
        %% retrieve data and cnn estimation
% NOTE > WE DON'T NEED THIS PART AS WE COLLECT SIGNALS FROM GENERATION STAGE
%        AFTER THE FRONT-END AMPLIFIER. SO ITS EFFECT IS ALREADY IN THE
%        COLLECTED SIGNAL.
%         % Front-end amplifier gain and thermal noise
%         rxPreAmp = phased.ReceiverPreamp( ...
%             'Gain',spLoss(uIdx), ...    % account for path loss
%             'NoiseFigure',prm.NFig,'ReferenceTemperature',290, ...
%             'SampleRate',prm.chanSRate);
%         rxPreSigAmp = rxPreAmp(rxPreSig{uIdx});
%         %   scale power for used sub-carriers
%         rxPreSigAmp = rxPreSigAmp * (sqrt(prm.FFTLength - ...
%             length(prm.NullCarrierIndices))/prm.FFTLength);
% 
%         inputRXSig_orig = rxPreSigAmp(chanDelay(uIdx)+1: ...
%             end-(prm.numPadZeros-chanDelay(uIdx)),:);
        
        if uIdx ~= 1
            inputRXSig = squeeze(usr_data{uIdx,1}(p,:,:));
        else
            inputRXSig_real = zeros(lenIn,nRXAnts);
            inputRXSig_imag = zeros(lenIn,nRXAnts);
            for iRx = 1:nRXAnts
                inputRXSig_real(:,iRx) = inputLTF_REAL((iRx-1)*nTXAnts+1, :);
                inputRXSig_imag(:,iRx) = inputLTF_IMAG((iRx-1)*nTXAnts+1, :);
            end
            inputRXSig = complex(inputRXSig_real,inputRXSig_imag);
        end
        
        snr_dB_CS = reshape(usr_data{uIdx,3}(p,:,:), [1,nRXAnts]);

        % OFDM demodulation
        rxOFDM = ofdmdemod(inputRXSig,prm.FFTLength, ...
            prm.CyclicPrefixLength,prm.CyclicPrefixLength, ...
            prm.NullCarrierIndices,prm.PilotCarrierIndices);

        % Channel estimation from preamble
        %       numCarr, numTx, numRx
        % Using classic (LS) method
        [hDp{uIdx},~,~,hDp_mmse{uIdx}] = helperMIMOChannelEstimate(rxOFDM(:,1:numTx,:),prm,1,h,snr_dB_CS,isMMSE);
        
        if isPlotting && (uIdx == 1)
            plot_mimo_channel(hDp{uIdx},numRx,10);
            plot_mimo_channel(hDp_mmse{uIdx},numRx,11);
            plot_mimo_channel(CSI_dnn,numRx,12);
            plot_mimo_channel(hDp_real{uIdx},numRx,9);
        end
        

        if uIdx == 1
            hDpDNN{uIdx} = CSI_dnn;
        end
    end
    %profile off
    
    for estSource=1:4   % 1 = LS est.; 2 = MMSE est.; 3 = DNN est.; 4 = perfect est.
        if estSource == 3
            hDp{1} = hDpDNN{1}; % only substitute for user ID=1
        elseif estSource == 2
            hDp = hDp_mmse;
        elseif estSource == 4
            hDp = hDp_real;
        end
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
        [rxSig,chanDelay,h] = helperApplyMUChannel(txSig,prm,spLoss,N_chan_taps, p,preambleSig);

        %% Receive Amplification and Signal Recovery
        % The receiver modeled per user compensates for the path loss by amplification and adds thermal noise. Like the transmitter, the receiver used in a MIMO-OFDM system contains many stages including OFDM demodulation, MIMO equalization, QAM demapping, and channel decoding.

        if isPlotting
            hfig = figure('Name','Equalized symbol constellation per stream');
        end

        scFact = ((prm.FFTLength-length(prm.NullCarrierIndices))...
                 /prm.FFTLength^2)/numTx;
        nVar = noisepow(prm.chanSRate,prm.NFig,290)/scFact; % this might be overwritten if NoiseMethod property of Receiver preamp is 'Noise Power'
        decoder = comm.ViterbiDecoder('InputFormat','Unquantized', ...
            'TrellisStructure',poly2trellis(7, [133 171 165]), ...
            'TerminationMethod','Terminated','OutputDataType','double');

        for uIdx = 1:prm.numUsers
            stsU = numSTSVec(uIdx);
            stsIdx = sum(numSTSVec(1:(uIdx-1)))+(1:stsU);
            
            gain_dB = spLoss(uIdx); % gain used in the amplifier is equal to path loss from BS to user's position
            
%             sigPwr = mean(abs(rxSig{uIdx})) ;   % compute the signal power in Watts
%             sig_dB = 10*log10(sigPwr);
%             %snr_dB_DT = sig_dB - noise_dB + gain_dB;
%             noise_dB = sig_dB - snr_dB_DT + gain_dB;
%             noise_dB = mean(noise_dB);  % average noise over receiver antennas
%             nVar = mean( sigPwr/(10*(.1*snr_dB_DT)) ); 

            %sigPwr = mean(abs(rxSig{uIdx})) ;   % compute the signal power in Watts
            sigPwr = rms(rxSig{uIdx}).^2;
            sig_dB = 10*log10(sigPwr);
            snr_dB_DT = sig_dB - noise_dB + gain_dB;
            %nVar = mean( sigPwr/(10.^(.1*snr_dB_DT)) );  
            nVar = db2pow(noise_dB);
            
            nVar = nVar*((prm.FFTLength-length(prm.NullCarrierIndices))...
                     /prm.FFTLength^2)/numTx;
            
            disp(strcat("DATA TRANSM - User ", num2str(uIdx), " SNR: ", num2str(snr_dB_DT)));

            % Front-end amplifier gain and thermal noise
            rxPreAmp = phased.ReceiverPreamp( ...
                'Gain',gain_dB, ...    % account for path loss
                'NoiseMethod', 'Noise power', ...
                'NoisePower', db2pow(noise_dB));
                %'NoiseFigure',prm.NFig,'ReferenceTemperature',290, 'SampleRate',prm.chanSRate, ...
            rxSigAmp = rxPreAmp(rxSig{uIdx});

%             
%             % Front-end amplifier gain and thermal noise
%             rxPreAmp = phased.ReceiverPreamp( ...
%                 'Gain',spLoss(uIdx), ...        % account for path loss
%                 'NoiseFigure',prm.NFig,'ReferenceTemperature',290, ...
%                 'SampleRate',prm.chanSRate);
%             rxSigAmp = rxPreAmp(rxSig{uIdx});

            % Scale power for occupied sub-carriers
            rxSigAmp = rxSigAmp*(sqrt(prm.FFTLength-length(prm.NullCarrierIndices)) ...
                /prm.FFTLength);

            % OFDM demodulation
            rxOFDM = ofdmdemod(rxSigAmp(chanDelay(uIdx)+1: ...
                end-(prm.numPadZeros-chanDelay(uIdx)),:),prm.FFTLength, ...
                prm.CyclicPrefixLength,prm.CyclicPrefixLength, ...
                prm.NullCarrierIndices,prm.PilotCarrierIndices);

            % Channel estimation from the mapped preamble
            [hD,~,~,~] = helperMIMOChannelEstimate(rxOFDM(:,1:numSTS,:),prm,1,h,snr_dB_DT,false);

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
            
            % Compute and display bit error rate
            ber = comm.ErrorRate;
            measures = ber(txDataBits{uIdx},rxBits);
            
            % compute gain in dB for Data Transfer phase due to beamforming
            mean_SNR_gain = mean(snr_dB_DT) - mean(snr_dB_CS);
            
            
            % save ber 
            if uIdx == 1
                if estSource == 1
                    disp('LS CSI estimation');
                    disp(['  User ' num2str(uIdx)]);
                    disp(['  RMS EVM (%) = ' num2str(rmsEVM)]);

                    bers_LS(p) = measures(1);
                    EVM_rms_LS(p) = rmsEVM;
                    %MSE_LS(p) = MSE(hDp_real{uIdx}, hDp{uIdx}); 
                    MSE_LS(p) = NMSE_subk(hDp_real{uIdx}, hDp{uIdx});
                    fprintf('  MSE = %.5f;\n', MSE_LS(p));
                    dtSNR_LS(p) = mean_SNR_gain;
                    
                    
                elseif  estSource == 2
                    disp('MMSE CSI estimation');
                    disp(['  User ' num2str(uIdx)]);
                    disp(['  RMS EVM (%) = ' num2str(rmsEVM)]);
                    
                    bers_MMSE(p) = measures(1);
                    EVM_rms_MMSE(p) = rmsEVM;
                    %MSE_MMSE(p) = MSE(hDp_real{uIdx}, hDp_mmse{uIdx});
                    MSE_MMSE(p) = NMSE_subk(hDp_real{uIdx}, hDp_mmse{uIdx});
                    
                    fprintf('  MSE = %.5f;\n', MSE_MMSE(p));
                    dtSNR_MMSE(p) = mean_SNR_gain;
                    
                elseif estSource == 3
                    disp('DNN CSI estimation');
                    disp(['  User ' num2str(uIdx)]);
                    disp(['  RMS EVM (%) = ' num2str(rmsEVM)]);
                    
                    bers_DNN(p) = measures(1);
                    EVM_rms_DNN(p) = rmsEVM;
                    %MSE_DNN(p) = MSE(hDp_real{uIdx}, hDpDNN{uIdx});
                    MSE_DNN(p) = NMSE_subk(hDp_real{uIdx}, hDpDNN{uIdx});
                    fprintf('  MSE = %.5f;\n', MSE_DNN(p));
                    dtSNR_DNN(p) = mean_SNR_gain;
                    
                elseif estSource == 4
                    disp('PERFECT CSI estimation');
                    disp(['  User ' num2str(uIdx)]);
                    disp(['  RMS EVM (%) = ' num2str(rmsEVM)]);
                    
                    bers_perfect(p) = measures(1);
                    EVM_rms_perfect(p) = rmsEVM;
                    dtSNR_perfect(p) = mean_SNR_gain;
                    
                end
            end
            
            
            fprintf('  BER = %.5f; No. of Bits = %d; No. of errors = %d\n', ...
                measures(1),measures(3),measures(2));
            
                 
                
        end
    end
    
    
end

[filepath,~,~] = fileparts(real_csi_pred);
save(strcat(filepath,'/metrics.mat'), 'bers_LS', 'EVM_rms_LS', 'MSE_LS','bers_MMSE', 'EVM_rms_MMSE', 'MSE_MMSE', 'bers_DNN', 'EVM_rms_DNN', 'MSE_DNN', 'bers_perfect','EVM_rms_perfect', 'dtSNR_LS','dtSNR_MMSE','dtSNR_DNN','dtSNR_perfect', '-v7.3');
metrics.bers_LS = bers_LS;
metrics.EVM_rms_LS = EVM_rms_LS;
metrics.MSE_LS = MSE_LS;
metrics.bers_MMSE = bers_MMSE;
metrics.EVM_rms_MMSE = EVM_rms_MMSE;
metrics.MSE_MMSE = MSE_MMSE;
metrics.bers_DNN = bers_DNN;
metrics.EVM_rms_DNN = EVM_rms_DNN;
metrics.MSE_DNN = MSE_DNN;
metrics.bers_perfect = bers_perfect;
metrics.EVM_rms_perfect = EVM_rms_perfect;
metrics.dtSNR_LS = dtSNR_LS;
metrics.dtSNR_MMSE = dtSNR_MMSE;
metrics.dtSNR_DNN = dtSNR_DNN;
metrics.dtSNR_perfect = dtSNR_perfect;



rng(s);         % restore RNG state
end

function [out] = NMSE_subk(real, pred)
    diff = real - pred;
    tx = size(real,2);
    rx = size(real,3);
    subK_nmse = zeros(tx,rx);
    for t=1:tx
        for r=1:rx
            subK_nmse(t,r) = norm(squeeze(diff(:,t,r)))^2 / norm(squeeze(real(:,t,r)))^2;
        end
    end
    out = mean(subK_nmse, 'all');
end

function [out] = MSE(real, pred)
    diff = real - pred;
    out = mean(abs(diff), 'all');
end
    