function snr_loop_testing(inputData_path, testData_path, nPkts, snr, Nt)

% snr=[-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-5,0,5,10];
% nPkts=500;

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

for s=snr
    % uncomment the following line in case you wanna compute the metrics directly from this script
    % [metrics] = BER_test_maMIMO_LTF(strcat(inputData_path,"/maMIMO_",num2str(nPkts),"___1UsrTest_BS",num2str(Nt),"_SNR",num2str(s),".mat"), nPkts, 1.0, strcat(testData_path,"/BS",num2str(Nt),"_SNR",num2str(s),"/test_csi_predictions_real.mat"), strcat(testData_path,"/BS",num2str(Nt),"_SNR",num2str(s),"/test_csi_predictions_imag.mat"),false,true,s);
    
    metrics = load(strcat(testData_path,"/BS",num2str(Nt),"_SNR",num2str(s),"/metrics.mat"));
    
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


f = figure('visible','off');
semilogy(snr,bers_LS_SNR,'-o',snr,bers_MMSE_SNR,'-x',snr,bers_DNN_SNR,'-*');
xlabel('SNR (dB)');
ylabel('Bit error rate (BER)');
grid on;
legend('LS','MMSE','Proposed');
saveas(f,strcat(testData_path,'BER'),'png');

f = figure('visible','off');
hold on;
grid on;
plot(snr,evm_LS_SNR,'-o');
plot(snr,evm_MMSE_SNR,'-x');
plot(snr,evm_DNN_SNR,'-*');
xlabel('SNR (dB)');
ylabel('EVM RMS (%)');
hold off;
legend('LS','MMSE','Proposed');
saveas(f,strcat(testData_path,'EVM'),'png');


f = figure('visible','off');
semilogy(snr,MSE_LS_SNR,'-o',snr,MSE_MMSE_SNR,'-x',snr,MSE_DNN_SNR,'-*');
grid on;
xlabel('SNR (dB)');
ylabel('MSE');
legend('LS','MMSE','Proposed');
saveas(f,strcat(testData_path,'MSE'),'png');


f = figure('visible','off');
hold on;
plot(snr,bfGain_LS_SNR,'-o');
plot(snr,bfGain_MMSE_SNR,'-x');
plot(snr,bfGain_DNN_SNR,'-*');
grid on;
xlabel('SNR (dB)');
ylabel('Beamforming gain (dB)');
hold off;
legend('LS','MMSE','Proposed');
saveas(f,strcat(testData_path,'BeamformGain'),'png');


end

function [CI] = compute_CI(x)
    SEM = std(x)/sqrt(length(x));               % Standard Error
    ts = tinv([0.025  0.975],length(x)-1);      % T-Score
    CI = mean(x) + ts*SEM;                      % Confidence Intervals
end