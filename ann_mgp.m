clc
clear all
close all
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Load Database %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = readmatrix('mgp.csv');
data = [data(2:end,:)]; % to convert the datatype to double
data1 = data';
[rs, cs] = size(data);
notg = 5;
nof = 2;

% display data
% figure,
% plot(data)
% xlabel("features")
% title("USC power output")
% legend("Feature"+string(1:nof),'Location','northeastoutside');

x = data(:,1:nof);
xt=x';
y = data(:,3:notg);
yt=y';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%partition the training and test data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the ratio of training and validation data
trainRatio = 0.8; % 80% for training
valRatio = 0.2;   % 20% for validation

% Use dividerand to split data
[trainInd, valInd, testInd] = dividerand(length(data), trainRatio, valRatio, 0);

% Split the data into training and validation sets
dataTrain = data(trainInd,:);
dataTest = data(valInd,:);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Standardize data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataTrainStd = dataTrain;
dataTestStd = dataTest;
mu = zeros(1,cs);
sig = ones(1,cs);

% apply standardization to both the dataTrain and dataTest
for i=1:nof
mu(i) = mean(dataTrain(:,i));
sig(i) = std(dataTrain(:,i));
dataTrainStd(:,i) = (dataTrainStd(:,i)-mu(i))/sig(i);
dataTestStd(:,i) = (dataTestStd(:,i)-mu(i))/sig(i);
end

for i=(nof+1):cs
dataTrainStd(:,i) = log(1+dataTrain(:,i));
dataTestStd(:,i) = log(1+dataTest(:,i));
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Plot Train data and standardize Train Data %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
figure,
plot(dataTrain)
xlabel("coal in")
title("USC power output")
legend("Feature"+string(1:cs),'Location','northeastoutside');

figure,
plot(dataTrainStd)
xlabel("coal in")
title("USC power output std")
legend("feature"+string(1:cs),'Location','northeastoutside');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Separate the input and output for training %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

XTrain = dataTrainStd(:,1:nof)';
YTrain = dataTrainStd(:,(1+nof:cs))';

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Define & Train Network  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hiddenLayerSize = [8 2];
net = fitnet(hiddenLayerSize);
num_epochs = 1500;
net.trainParam.epochs = num_epochs;
[net,tr] = train(net,XTrain, YTrain);
%%
XTest = dataTestStd(:,1:nof)';
YTestPredictedn = net(XTest);
YTestPredicted = YTestPredictedn'; %taking transpose
% Undo standardization on the predicted and actual outputs
for i=1:5
    YTestP_unstd(:,i) = exp(YTestPredicted(:,i))-1;
    YTestActual(:,i) = dataTest(:,i+nof);
    rmse(i) = sqrt(mean(((YTestPredicted(i) - YTestActual(i)).^2)));
end
% YTestPredicted1 = exp(YTestPredicted(:,1))-1;
% YTestPredicted2 = exp(YTestPredicted(:,2))-1;
% YTestPredicted3 = exp(YTestPredicted(:,3))-1;
% YTestPredicted4 = exp(YTestPredicted(:,4))-1;
% YTestPredicted5 = exp(YTestPredicted(:,5))-1;
% YTestActual1 = dataTest(:,3);
% YTestActual2 = dataTest(:,4);
% YTestActual3 = dataTest(:,5);
% YTestActual4 = dataTest(:,6);
% YTestActual5 = dataTest(:,7);
% % Calculate the squared differences
% rmse1 = sqrt(mean(((YTestPredicted1 - YTestActual1).^2)));
% rmse2 = sqrt(mean(((YTestPredicted2 - YTestActual2).^2)));
% rmse3 = sqrt(mean(((YTestPredicted3 - YTestActual3).^2)));
% rmse4 = sqrt(mean(((YTestPredicted4 - YTestActual4).^2)));
% rmse5 = sqrt(mean(((YTestPredicted5 - YTestActual5).^2)));
% rmse = [rmse1 rmse2 rmse3 rmse4 rmse5]'
%%
rmse3 = zeros(10, 10);
rmse4 = zeros(10, 10);
for i=1:10
    for j=1:10
hiddenLayerSize = [i j];
net = fitnet(hiddenLayerSize);
num_epochs = 1500;
net.trainParam.epochs = num_epochs;
[net,tr] = train(net,XTrain, YTrain);
%%
XTest = dataTestStd(:,1:4)';
YTestPredictedn = net(XTest);
YTestPredicted = YTestPredictedn'; %taking transpose
% Undo standardization on the predicted and actual outputs
YTestPredicted1 = exp(YTestPredicted(:,1))-1;
YTestPredicted2 = exp(YTestPredicted(:,2))-1;
YTestActual1 = dataTest(:,5);
YTestActual2 = dataTest(:,6);
rmse3(i,j) = sqrt(mean(((YTestPredicted1 - YTestActual1).^2)));
rmse4(i,j) = sqrt(mean(((YTestPredicted2 - YTestActual2).^2)));
    end
end
