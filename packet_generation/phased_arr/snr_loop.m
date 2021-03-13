isSameConf = false; % same client configuration
isSameSeed = false; % same seed for channel instances
prm = [];
if isSameConf
    %load("packets/maMIMO_2000_8dB___1Usr_BS32.mat");
    load("packets/SNRanalysis/maMIMO_3000___1Usr_BS32_SNR120.mat","prm");
end

if ~isSameSeed
    prm.seed_p = [];    
end
snr=-15;%(-20:5:10);
snr_CS_list = [];
snr_DT_list = [];
bers = zeros(size(snr));
c = 0;
nPkts = 500;
for i=1:length(snr)
    s = snr(i)
    [mean_ber] = generate_maMIMO_LTF(strcat("1UsrTest_BS32_SNR",num2str(s)),nPkts,32,4,s,false,true,false,prm,true);
%     bers(i) = mean_ber;
end

% testData_path = "/home/mauro/Research/maMIMO_deepCSIEst/CSI_estimation/magazine_review/BS32_denoise_3k_SNR120/test_results/";
% 
% f = figure('visible','off');
% semilogy(snr,bers,'-o');
% xlabel('SNR (dB)');
% ylabel('Bit error rate (BER)');
% grid on;
% saveas(f,strcat(testData_path,'generated_BER'),'fig');
