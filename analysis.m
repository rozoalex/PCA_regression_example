% 
clear; 
%% Load Training data
train_data = [importdata('hw4.train1');importdata('hw4.train2');importdata('hw4.train3')];
ANS = train_data(:,1);
FEATURES = train_data(:,2:end);
%% PCA and Plot the Variances
[coefs,scores,variances] = pca(FEATURES');
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
f1 = scores(1,:)'; % feature 1
f2 = scores(2,:)'; % feature 2
f3 = scores(3,:)'; % feature 3
figure('Name', 'f1 f2 f3')
%
subplot(2,2,1)
scatter3(f1,f2,ANS)
xlabel('f1')
ylabel('f2')
zlabel('ans')
%
subplot(2,2,2)
plot(f1,f2,'.')
xlabel('f1')
ylabel('f2')
%
subplot(2,2,3)
plot(f1,ANS,'.')
xlabel('f1')
ylabel('ans')
%
subplot(2,2,4)
plot(f2,ANS,'.')
xlabel('f2')
ylabel('ans')

%% Linear Regression
% With 2 features
rgs_2 = regress(ANS,[ones(size(f1)) f1 f2]);
% With 3 features
rgs_3 = regress(ANS,[ones(size(f1)) f1 f2 f3]);
figure('Name','Linear Regression')
scatter3(f1,f2,ANS,'filled')
hold on
f1fit = min(f1):100:max(f1);
f2fit = min(f2):10:max(f2);
[F1FIT,F2FIT] = meshgrid(f1fit,f2fit);
YFIT = rgs_2(1) + rgs_2(2)*F1FIT + rgs_2(3)*F2FIT;
mesh(F1FIT,F2FIT,YFIT)
xlabel('f1')
ylabel('f2')
zlabel('ans')
view(50,10)

%% Varify Errors
% verification_data = [importdata('hw4.train4')];
% test_data = [importdata('hw4.test1');importdata('hw4.test2');importdata('hw4.test3');importdata('hw4.test4')];


%%