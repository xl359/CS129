clc
clear all
rng(0)
load sisre.mat
%load svn.mat
load svn_meas.mat
% implement CostTrain and CV and see that one is significantly larger than
% the other and this would not make sense for a regular fitting ML problem
% but this is data forecasting and it is unfair for the original data set
% that the net is updating each step of the way, we only need to discuss
% costCV
%% Set up Baseline Parameters:
numofData = 700;
numHiddenUnits = 500;%250
NumofIT = 500;%600;
numofLayers = 1;
Bi = 0;
LearningRate = 0.005;
PercentTrain = 0.8;%0.7;
%BastLineCost = Cost(svn_meas,numofData,numHiddenUnits,NumofIT,numofLayers,Bi, LearningRate,PercentTrain);
%% Varying the amount we can predict ahead
PercentVary = 0.65:0.05:0.85;
ind = 0;
CostCV = zeros(length(PercentVary),1);
for i = PercentVary
    ind = ind + 1;
    costCV = Cost(svn_meas,numofData,numHiddenUnits,NumofIT,numofLayers,Bi, LearningRate,i);
    CostCV(ind) = costCV;
end
figure
plot(PercentVary,CostCV,'LineWidth',2)
xlabel('Percent of Training Data','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Effect of the Percent of Training Data on Cost Value','FontSize', 14)
%% Varying Number of Data Done!
VaryingnumofData = [500 600 700 800 1000 2000 3000];
ind = 0;
CostCV = zeros(length(VaryingnumofData),1);
for i = VaryingnumofData
    ind = ind + 1;
    costCV = Cost(svn_meas,i,numHiddenUnits,NumofIT,numofLayers,Bi, LearningRate,PercentTrain);
    CostCV(ind) = costCV;
end
figure
plot(VaryingnumofData,CostCV,'LineWidth',2)
xlabel('Amount of Data','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Effect of the Amount of Data on Cost Value','FontSize', 14)
% dont worry about costtrain or just mention

%% Varying Number of Hidden Nodes Done!
VaryingnumofNodes = 100:200:900;%100:50:1000;
ind = 0;
CostCV = zeros(length(VaryingnumofNodes),1);
for i = VaryingnumofNodes
    ind = ind + 1;
    costCV = Cost(svn_meas,numofData,i,NumofIT,numofLayers,Bi, LearningRate,PercentTrain);
    CostCV(ind) = costCV;
end
spl = spline(VaryingnumofNodes,CostCV);
Smooth = smoothdata(CostCV);
figure
%plot(VaryingnumofNodes,CostCV)
xint = 100:2:750;
plot(VaryingnumofNodes(1:end-1),CostCV(1:end-1),'o',xint,ppval(spl,xint),'r-','LineWidth',2)
xlabel('Amount of Hidden Nodes','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Effect of the Amount of Hidden Nodes on Cost Value','FontSize', 14)
%% Varying Number of Layers
CostCV = zeros(3,1);
VaryNumberofLayers = 1:3;
for i = VaryNumberofLayers
    costCV = Cost(svn_meas,numofData,numHiddenUnits,NumofIT,i,Bi, LearningRate,PercentTrain);
    CostCV(i) = costCV;
end
plot(VaryNumberofLayers,CostCV)
%% Try BilstmLayer
CostCV = zeros(2,1);
for i = 0:1
    costCV = Cost(svn_meas,numofData,numHiddenUnits,NumofIT,numofLayers,i, LearningRate,PercentTrain);
    CostCV(i) = costCV;
end
plot([1,2],CostCV) % 1 is lstm layer, 2 is bilayer
% first one is lstm, second one is bi

% since all converge, there is no point in testing the learning rate and
% the number of iteration
%%
function costCV = Cost(svn_meas,numofData,numHiddenUnits,NumofIT,numofLayers,Bi, LearningRate,PercentTrain)
% If Bi is 1, use lstm. if 0, use bilstm
data = svn_meas(33,2:numofData);
%% Divide by Training , CV, and Testing, note Testing is not used here


% Divide the training set to 0.7 training, 0.2 CV and 0.1 test
% This is to prevent overit on the testing. We use the difference between
% CV and testing to tune the parameters and use test for testing. 
numTimeStepsTrain = floor(PercentTrain*numel(data));
numTimeStepsCV = floor((0.9)*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataCV = data(numTimeStepsTrain+1:numTimeStepsCV+1);
dataTest = data(numTimeStepsCV+1:end);

%% Mean Normalize the Data

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

%% Specify inputs and outputs

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
%% Mean Normalize the Data

muCV = mu;
sigCV = sig;

dataCVStandardized = (dataCV - muCV) / sigCV;

%% Specify inputs and outputs

XCV = dataCVStandardized(1:end-1);
YCV = dataCVStandardized(2:end);


%% Defind LSTM Structures:

numFeatures = 1;
numResponses = 1;

if numofLayers ==1

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
end

if numofLayers ==2

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
end
if numofLayers ==3

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)
        lstmLayer(numHiddenUnits)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
end

if Bi == 1
    layers = [ ...
        sequenceInputLayer(numFeatures)
        bilstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
end
    
% options = trainingOptions('adam', ...
%     'MaxEpochs',NumofIT, ...
%     'GradientThreshold',1, ...
%     'InitialLearnRate',LearningRate, ... %0.005
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',125, ...
%     'LearnRateDropFactor',0.2, ...
%     'Verbose',0, ...
%     'Plots','none');
%     %'Plots','training-progress');

    options = trainingOptions('adam', ...
    'MaxEpochs',NumofIT, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',LearningRate, ... %0.005
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Train Network
%net.trainParam.showWindow = 0;
rng(0,'combRecursive');
netTrain = trainNetwork(XTrain,YTrain,layers,options);

%% Based on the hypothesis, compute the cost of the original training set
% netTrain = predictAndUpdateState(netTrain,XTrain);
% [net,YPred] = predictAndUpdateState(netTrain,XTrain(1));
% 
% numTimeStepsTrain = numel(XTrain);
% for i = 2:numTimeStepsTrain
%     [net,YPred(:,i)] = predictAndUpdateState(netTrain,YPred(:,i-1),'ExecutionEnvironment','cpu');
% end
%% Cost Train
% YPred = sig*YPred + mu;
% 
% YPredTrain = YPred;
% 
% YTrain = dataTrain(2:end);
% costTrain = 1/2*mean((YPredTrain-YTrain).^2);

%% Forecasting Future Steps CV

net = predictAndUpdateState(netTrain,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsCV = numel(XCV);
for i = 2:numTimeStepsCV
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%% Cost CV
YPred = sigCV*YPred + muCV;

YPredCV = YPred;

YCV = dataCV(2:end);
costCV = 1/2*mean((YPredCV-YCV).^2);



end

%%
% 1. Learing Curve
% 2. Tune number of layers








