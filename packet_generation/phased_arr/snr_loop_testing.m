function snr_loop_testing(testData_path)

%snr=30; %snr=[28:2:40 45:5:60]; % sameseed
%nPkts=30; 

%snr=[28:2:40];
%nPkts=300;

%snr=[-22:1:-10];
%snr=[-20:5:10];
snr=[-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-5,0,5,10];
%snr=[-10];
nPkts=500;

bers_LS_SNR = zeros(length(snr),1);
bers_LS_SNR_CIs = cell(length(snr),1); % confindence intervals
evm_LS_SNR = zeros(length(snr),1);
evm_LS_SNR_CIs = cell(length(snr),1);
MSE_LS_SNR = zeros(length(snr),1);
MSE_LS_SNR_CIs = cell(length(snr),1);

bers_MMSE_SNR = zeros(length(snr),1);
bers_MMSE_SNR_CIs = cell(length(snr),1);
evm_MMSE_SNR = zeros(length(snr),1);
evm_MMSE_SNR_CIs = cell(length(snr),1);
MSE_MMSE_SNR = zeros(length(snr),1);
MSE_MMSE_SNR_CIs = cell(length(snr),1);

bers_DNN_SNR = zeros(length(snr),1);
bers_DNN_SNR_CIs = cell(length(snr),1);
evm_DNN_SNR = zeros(length(snr),1);
evm_DNN_SNR_CIs = cell(length(snr),1);
MSE_DNN_SNR = zeros(length(snr),1);
MSE_DNN_SNR_CIs = cell(length(snr),1);

bfGain_LS_SNR = zeros(length(snr),1);
bfGain_MMSE_SNR = zeros(length(snr),1);
bfGain_DNN_SNR = zeros(length(snr),1);

c = 0;

%testData_path = "";

for s=snr
    %[bers_LS, evm_LS, bers_DNN, evm_DNN] = BER_test_maMIMO_LTF(strcat("packets/SNRanalysis/maMIMO_",num2str(nPkts),"_8dB___1Usr_BS32_SNR_sameseed",num2str(s),".mat"),nPkts,1.0,strcat("/home/mauro/Research/CSI_estimation/checkout_maMIMO_allBS/BS32_sameseed_SNR",num2str(s),"/test_csi_predictions_real.mat"),strcat("/home/mauro/Research/CSI_estimation/checkout_maMIMO_allBS/BS32_sameseed_SNR",num2str(s),"/test_csi_predictions_imag.mat"),true,true,s+10);
    %[bers_LS, evm_LS, MSE_LS, bers_MMSE, evm_MMSE, MSE_MMSE, bers_DNN, evm_DNN, MSE_DNN] = BER_test_maMIMO_LTF(strcat("packets/SNRanalysis/maMIMO_",num2str(nPkts),"_8dB___1Usr_BS32_SNR",num2str(s),".mat"),nPkts,1.0,strcat("/home/mauro/Research/CSI_estimation/checkout_maMIMO_allBS/BS32_SNR",num2str(s),"/test_csi_predictions_real.mat"),strcat("/home/mauro/Research/CSI_estimation/checkout_maMIMO_allBS/BS32_SNR",num2str(s),"/test_csi_predictions_imag.mat"),false,false,s+10);
    
    %[metrics] = BER_test_maMIMO_LTF(strcat("packets/SNRanalysis/maMIMO_",num2str(nPkts),"___1Usr_BS32_SNR",num2str(s),".mat"),nPkts,1.0,strcat("/home/mauro/Research/maMIMO_deepCSIEst/CSI_estimation/checkout_maMIMO_allBS/test_denoiseBeta/BS32_SNR",num2str(s),"/test_csi_predictions_real.mat"),strcat("/home/mauro/Research/maMIMO_deepCSIEst/CSI_estimation/checkout_maMIMO_allBS/test_denoiseBeta/BS32_SNR",num2str(s),"/test_csi_predictions_imag.mat"),true,true,s);
    %metrics = load(strcat("/home/mauro/Research/maMIMO_deepCSIEst/CSI_estimation/checkout_maMIMO_allBS/test_denoiseBeta/BS32_SNR",num2str(s),"/metrics2.mat"));
    
    %[metrics] = BER_test_maMIMO_LTF(strcat("packets/SNRanalysis/maMIMO_",num2str(nPkts),"___1Usr_BS32_SNR",num2str(s),".mat"), nPkts, 1.0, strcat(testData_path,"BS32_SNR",num2str(s),"/test_csi_predictions_real.mat"), strcat(testData_path,"BS32_SNR",num2str(s),"/test_csi_predictions_imag.mat"),false,true,s);
    metrics = load(strcat(testData_path,"BS32_SNR",num2str(s),"/metrics4.mat"));
    
    c = c + 1;
    bers_LS_SNR(c) = mean(metrics.bers_LS);
    bers_LS_SNR_CIs{c,1} = compute_CI(metrics.bers_LS);
    evm_LS_SNR(c) = mean(metrics.EVM_rms_LS);
    evm_LS_SNR_CIs{c,1} = compute_CI(metrics.EVM_rms_LS);
    MSE_LS_SNR(c) = mean(metrics.MSE_LS);
    MSE_LS_SNR_CIs{c,1} = compute_CI(metrics.MSE_LS);
    
    bers_MMSE_SNR(c) = mean(metrics.bers_MMSE);
    bers_MMSE_SNR_CIs{c,1} = compute_CI(metrics.bers_MMSE);
    evm_MMSE_SNR(c) = mean(metrics.EVM_rms_MMSE);
    evm_MMSE_SNR_CIs{c,1} = compute_CI(metrics.EVM_rms_MMSE);
    MSE_MMSE_SNR(c) = mean(metrics.MSE_MMSE);
    MSE_MMSE_SNR_CIs{c,1} = compute_CI(metrics.MSE_MMSE);
    
    bers_DNN_SNR(c) = mean(metrics.bers_DNN);
    bers_DNN_SNR_CIs{c,1} = compute_CI(metrics.bers_DNN);
    evm_DNN_SNR(c) = mean(metrics.EVM_rms_DNN);
    evm_DNN_SNR_CIs{c,1} = compute_CI(metrics.EVM_rms_DNN);
    MSE_DNN_SNR(c) = mean(metrics.MSE_DNN);
    MSE_DNN_SNR_CIs{c,1} = compute_CI(metrics.MSE_DNN);
    
    bfGain_LS_SNR(c) = mean(metrics.dtSNR_LS);
    bfGain_MMSE_SNR(c) = mean(metrics.dtSNR_MMSE);
    bfGain_DNN_SNR(c) = mean(metrics.dtSNR_DNN);
end

figure;
%f = figure('visible','off');
semilogy(snr,bers_LS_SNR,'-o',snr,bers_MMSE_SNR,'-x',snr,bers_DNN_SNR,'-*');
%semilogy(snr,bers_MMSE_SNR,'-x');
%semilogy(snr,bers_DNN_SNR,'-*');
xlabel('SNR (dB)');
ylabel('Bit error rate (BER)');
grid on;
%saveas(f,strcat(testData_path,'BER'),'fig');

figure;
%f = figure('visible','off');
hold on;
plot(snr,evm_LS_SNR,'-o');
plot(snr,evm_MMSE_SNR,'-x');
plot(snr,evm_DNN_SNR,'-*');
xlabel('SNR (dB)');
ylabel('EVM RMS (%)');
hold off;
%saveas(f,strcat(testData_path,'EVM'),'fig');

figure;
%f = figure('visible','off');
semilogy(snr,MSE_LS_SNR,'-o',snr,MSE_MMSE_SNR,'-x',snr,MSE_DNN_SNR,'-*');
%plot(snr,MSE_MMSE_SNR,'-x');
%plot(snr,MSE_DNN_SNR,'-*');
xlabel('SNR (dB)');
ylabel('MSE');
%saveas(f,strcat(testData_path,'MSE'),'fig');

figure;
%f = figure('visible','off)');
hold on;
plot(snr,pow2db(MSE_LS_SNR),'-o');
plot(snr,pow2db(MSE_MMSE_SNR),'-x');
plot(snr,pow2db(MSE_DNN_SNR),'-*');
xlabel('SNR (dB)');
ylabel('MSE');
%saveas(f,strcat(testData_path,'MSE'),'fig');


figure;
%f = figure('visible','off');
% vals = [bfGain_LS_SNR'; bfGain_MMSE_SNR'; bfGain_DNN_SNR'];
% h = bar(snr,vals);
hold on;
plot(snr,bfGain_LS_SNR,'-o');
plot(snr,bfGain_MMSE_SNR,'-x');
plot(snr,bfGain_DNN_SNR,'-*');
xlabel('SNR (dB)');
ylabel('Beamforming gain (dB)');
hold off;
%saveas(f,strcat(testData_path,'BeamformGain'),'fig');




% pkt_snr = cell(size(snr,1),1);
% c = 0;
% 
% figure;
% hold on;
% for s=snr
%     load(strcat("packets/SNRanalysis/maMIMO_",num2str(nPkts),"_8dB___1Usr_BS32_SNR_sameseed",num2str(s),".mat"));
%     c = c + 1;
%     pkt_snr{c,1} = usr_data{1,1};
%     if mod(c,2) == 1
%         plot(real(pkt_snr{c,1}(1,:,1)));
%     end
% end
% hold off;

end

function [CI] = compute_CI(x)
    SEM = std(x)/sqrt(length(x));               % Standard Error
    ts = tinv([0.025  0.975],length(x)-1);      % T-Score
    CI = mean(x) + ts*SEM;                      % Confidence Intervals
end