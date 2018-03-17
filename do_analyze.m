function [predicted, testing] = do_analyze( filenum, is_out)
    %% Load Training data
    train_data = importdata(sprintf('hw4.train%d',filenum));
    % importdata('hw4.train2');importdata('hw4.train3')
    ANS = train_data(:,1);
    train_FEATURES = train_data(:,2:end);
    %% Train
    [coefs,scores,~] = pca(train_FEATURES); % PCA 
    % Linear Regression With 3 PCs
    rgs_2 = regress(ANS,[ones(size(ANS)) scores(:,1:3)]);
    % rgs_l = lasso(scores(:,1:3),train_ANS); 
    %% Predict
    varify_FEATURES = importdata(sprintf('hw4.test%d',filenum));
    varify_transd_FEATURES = varify_FEATURES * coefs;
    varify_pcs_3 = [ones(size(varify_FEATURES(:,1))) varify_transd_FEATURES(:,1:3)];
    varify_predict_ANS_3 = varify_pcs_3 * rgs_2;
    testing = varify_predict_ANS_3;
    predicted = [varify_predict_ANS_3 varify_FEATURES];
    if is_out 
%         fileID = fopen(sprintf('hw4.test%d',filenum),'w');
%         fprintf(fileID,predicted);
%         fclose(fileID);
        dlmwrite(sprintf('hw4.testresul%d',filenum),predicted);
    end

end

