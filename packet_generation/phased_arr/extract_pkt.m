function [usr_data,prm] = extract_pkt(txs_file, numPackets, isSave, exp_ID, isReverse)

    load(txs_file);
    usr_data_orig = usr_data;
    [numToTPackets, lenIn, nRXAnts] = size(usr_data_orig{1,1});
    % create a new cell structure
    usr_data = cell(prm.numUsers,3);    % cell used to store different dataset transmissions for every user
    
    if ~isReverse
        indexes = 1:numPackets;
    else
        indexes = (numToTPackets-numPackets)+1:numToTPackets;
    end
    
    for uIdx=1:prm.numUsers
        inputRXSig = squeeze(usr_data_orig{uIdx,1}(1,:,:)); % sample LTF to retrieve size
        hDp = squeeze(usr_data_orig{uIdx,2}(1,:,:,:)); % sample estimated channel to retrieve size
        snr_dB_CS = usr_data_orig{uIdx,3}(1,:,:);   % sample SNR for channel sounding phase for given transmission

        usr_data{uIdx,1} = zeros(numPackets, size(inputRXSig,1),size(inputRXSig,2), 'like', inputRXSig);
        usr_data{uIdx,1} = usr_data_orig{uIdx,1}(indexes, :,:);
        usr_data{uIdx,2} = zeros(numPackets,  size(hDp,1),size(hDp,2),size(hDp,3), 'like', hDp);
        usr_data{uIdx,2} = usr_data_orig{uIdx,2}(indexes,:,:,:);
        usr_data{uIdx,3} = zeros(numPackets,size(snr_dB_CS,2),size(snr_dB_CS,3), 'like', snr_dB_CS);
        usr_data{uIdx,3} = usr_data_orig{uIdx,3}(indexes,:,:);
        
        
        %adjust also seeds 
        prm.seed_p{uIdx} = prm.seed_p{uIdx}(indexes);
    end
    
    if isSave
        save(strcat('packets/','maMIMO_',num2str(numPackets),'_',num2str(prm.NFig),'dB___',exp_ID,'.mat'),'usr_data','seed_p','P','prm','-v7.3');

    end
end