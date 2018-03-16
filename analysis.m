% 
clear; 
close all;
number_of_train = 250;
%% Load Training data
train_data = importdata('hw4.train4');
% importdata('hw4.train2');importdata('hw4.train3')
ANS = train_data(:,1);
train_FEATURES = train_data(1:number_of_train,2:end);
train_ANS = train_data(1:number_of_train,:,1);
train_ANS = train_ANS(:,1);
%% PCA and Plot the Variances
[coefs,scores,variances] = pca(train_FEATURES);
normalized_variances = variances / variances(1);
% Drawing time
var_graph = figure('Name', 'Variances of PCA');
subplot(4,1,1)
bar(variances)
xlabel('Feature')
ylabel('variance')
title('Variances for All features')
subplot(4,1,2)
bar(normalized_variances)
xlabel('Feature')
ylabel('variance')
title('Normalized Variances for All features')
subplot(4,1,3)
bar(normalized_variances(1:4))
xlabel('Feature')
ylabel('variance')
title('Normalized Variances for the first 4 biggest ')
annotation('textbox', [.15, .05, .7, .15], 'string', 'We can see that, the first 2 principle components are so huge compare to others, which means the result is mostly dependent on those two pcs. Close the window to continue...')
% waitfor(var_graph) % close the window to continue
%%
pc1 = scores(:,1); % feature 1
pc2 = scores(:,2); % feature 2
pc3 = scores(:,3); % feature 3
pc4 = scores(:,4); % feature 3
pc5 = scores(:,5); % feature 3
pc6 = scores(:,6); % feature 3
figure('Name', 'f1 f2 f3')
%
subplot(2,2,1)
scatter3(pc1,pc2,train_ANS(:,1),'.')
xlabel('pc1')
ylabel('pc2')
zlabel('ans')
%
subplot(2,2,2)
plot(pc1,pc2,'.')
xlabel('pc1')
ylabel('pc2')
%
subplot(2,2,3)
plot(pc1,train_ANS,'.')
xlabel('pc1')
ylabel('ans')
%
subplot(2,2,4)
plot(pc2,train_ANS,'.')
xlabel('pc2')
ylabel('ans')

%% Linear Regression
% With 2 features
rgs_2 = regress(train_ANS,[ones(size(pc1)) pc1 pc2 pc3]);
rgs_6 = regress(train_ANS,[ones(size(pc1)) pc1 pc2 pc3 pc4 pc5 pc6]);

% rgs_3 = lasso(scores(:,2:10),train_ANS,'CV',10);
% With 3 features
figure('Name','Linear Regression');
scatter3(pc1,pc2,train_ANS,'filled');
hold on

f1fit = min(pc1):100:max(pc1);
f2fit = min(pc2):10:max(pc2);
[F1FIT,F2FIT] = meshgrid(f1fit,f2fit);
YFIT = rgs_2(1) + rgs_2(2)*F1FIT + rgs_2(3)*F2FIT;
mesh(F1FIT,F2FIT,YFIT);
xlabel('pc1')
ylabel('pc2')
zlabel('ans')
% view(50,10)

%% Varify Errors
varify_FEATURES = train_data(number_of_train+1:304,2:end);
varify_ANS = ANS((number_of_train+1):304);
varify_transd_FEATURES = varify_FEATURES * coefs;
varify_pc1 = varify_transd_FEATURES(:,1);
varify_pc2 = varify_transd_FEATURES(:,2);
varify_pc3 = varify_transd_FEATURES(:,3);
varify_pc4 = varify_transd_FEATURES(:,4);
varify_pc5 = varify_transd_FEATURES(:,5);
varify_pc6 = varify_transd_FEATURES(:,6);
varify_pcs_3 = [ones(size(varify_pc1)) varify_pc1 varify_pc2 varify_pc3];
varify_pcs_6 = [ones(size(varify_pc1)) varify_pc1 varify_pc2 varify_pc3 varify_pc4 varify_pc5 varify_pc6];
varify_predict_ANS_3 = varify_pcs_3 * rgs_2;
varify_predict_ANS_6 = varify_pcs_6 * rgs_6;

figure('Name','Varify Predicted Results 3D')
plot(varify_ANS, varify_predict_ANS_3,'.')
xlabel('ans')
ylabel('predict')
RMSE = sqrt(mean((varify_ANS - varify_predict_ANS_3).^2));  % Root Mean Squared Error
annotation('textbox', [.15, .05, .7, .15], 'string', sprintf('RMSE = %0.5e',RMSE))

figure('Name','Varify Predicted Results 6D')
plot(varify_ANS, varify_predict_ANS_6,'.')
xlabel('ans')
ylabel('predict')
RMSE_6 = sqrt(mean((varify_ANS - varify_predict_ANS_6).^2));  % Root Mean Squared Error
annotation('textbox', [.15, .05, .7, .15], 'string', sprintf('RMSE = %0.5e',RMSE_6))


%%
% 
% l = len(train_ANS)
% rgs_3 = regress(ANS,[ones(size(f1)) f1 f2 f3]);
