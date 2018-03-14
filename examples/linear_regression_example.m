load carsmall
x1 = Weight;
x2 = Horsepower; % Contains NaN data
y = MPG;

%% Linear regression
X = [ones(size(x1)) x1 x2];
b = regress(y,X);

scatter3(x1,x2,y,'filled')
hold on
x1fit = min(x1):100:max(x1);
x2fit = min(x2):10:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
mesh(X1FIT,X2FIT,YFIT)
xlabel('Weight')
ylabel('Horsepower')
zlabel('MPG')
view(50,10)

%% Consider interaction between variables
X = [ones(size(x1)) x1 x2 x1.*x2];
b = regress(y,X);

scatter3(x1,x2,y,'filled')
hold on
x1fit = min(x1):100:max(x1);
x2fit = min(x2):10:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;
mesh(X1FIT,X2FIT,YFIT)
xlabel('Weight')
ylabel('Horsepower')
zlabel('MPG')
view(50,10)

%% Generalized linear regression

% data of bionomial distribution
x = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
y = [1 2 0 3 8 8 14 17 19 15 17 21]' ./ [48 42 31 34 31 21 23 23 21 16 17 21]';

links = {'logit', 'probit', 'log' };
linkID = 1; % right choice is 2

b = glmfit(x,y,'binomial','link', links{linkID});
yfit = glmval(b,x,'probit');
plot(x, y,'o',x,yfit,'-','LineWidth',2)

%% regularized logistic regression

%% load data
load ionosphere
Ybool = strcmp(Y,'g');
X = X(:,3:end);
% size( X )
% size( Ybool )
% plot( X(:,1), Ybool );

%% Fit the model
[B0,FitInfo0] = lassoglm(X,Ybool,'binomial','Lambda',0,'CV',10);
[B,FitInfo] = lassoglm(X,Ybool,'binomial','NumLambda',25,'CV',10);

% check FitInfo0.SE vs FitInfo.SE
% check FitInfo0.Deviance vs FitInfo.Deviance

%% examine the trace
lassoPlot(B,FitInfo,'PlotType','CV');
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');