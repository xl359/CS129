clc
clear all
rng(0)
load sisre.mat
load svn.mat
load svn_meas.mat
%%

data = svn_meas(33,2:700);
%data = [data{:}];
%epochs = epochs(1:699);
figure
plot(data,'LineWidth',2)
xlabel("Time Series",'FontSize', 14)
ylabel("Measurement",'FontSize', 14)
title("Measurement Data Vs Epochs Times",'FontSize', 14)



%% Divide by Training , CV, and Testing


% Divide the training set to 0.7 training, 0.2 CV and 0.1 test
% This is to prevent overit on the testing. We use the difference between
% CV and testing to tune the parameters and use test for testing. 
numTimeStepsTrain = floor(0.8*numel(data));
numTimeStepsCV = floor(0.9*numel(data));

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

%% Defind LSTM Structures:

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 500;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%     options = trainingOptions('adam', ...
%     'MaxEpochs',250, ...
%     'GradientThreshold',1, ...
%     'InitialLearnRate',0.005, ...
%     'LearnRateSchedule','none', ...
%     'Verbose',0, ...
%     'Plots','none');
%     %'Plots','training-progress');

%% Train Network
rng(0,'combRecursive');

netTrain = trainNetwork(XTrain,YTrain,layers,options);

%% Forecasting Future Steps CV

dataCVStandardized = (dataCV - mu) / sig;
XCV = dataCVStandardized(1:end-1);


net = predictAndUpdateState(netTrain,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsCV = numel(XCV);
for i = 2:numTimeStepsCV
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%% Plotting CV
YPred = sig*YPred + mu;

YPredCV = YPred;

YCV = dataCV(2:end);
rmse = sqrt(mean((YPred-YCV).^2))

% figure
% plot(dataTrain(1:end-1))
% hold on
% idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsCV);
% plot(idx,[data(numTimeStepsTrain) YPred],'.-')
% hold off
% xlabel("Epochs Time")
% ylabel("Measurement")
% title("Measurement Data Vs Epochs Times")
% legend(["Observed" "Forecast"])

% 
% figure
% subplot(2,1,1)
% plot(YCV)
% hold on
% plot(YPred,'.-')
% hold off
% legend(["Observed" "Forecast"])
% ylabel("Cases")
% title("Forecast")
% 
% subplot(2,1,2)
% stem(YPred - YCV)
% xlabel("Epochs")
% ylabel("Error")
% title("RMSE = " + rmse)

 %% Forecasting Future Steps Test

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);


%net = predictAndUpdateState(net,YPred(:,i-1));
% If we were to use the original model for prediction, we should keep
% predicting from the last update of CV to simulate a realistic useage of
% the updating. 
[net,YPred] = predictAndUpdateState(net,YPred(:,i-1));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%% Plotting Test
YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2));

figure
plot([dataTrain(1:end-1) dataCV(1:end-1)],'LineWidth',2)
hold on
plot([dataTrain(1:end-1) YPredCV],'LineWidth',2)
idx = numTimeStepsCV+numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsCV+numTimeStepsTest);
%plot(idx,[data(numTimeStepsCV+numTimeStepsTrain) YPred],'.-')
plot(idx,[YPredCV(end) YPred],'.-','LineWidth',2)

hold off
xlabel("Epochs Time",'FontSize', 14)
ylabel("Measurement",'FontSize', 14)
title("Measurement Data Vs Epochs Times",'FontSize', 14)
legend(["Observed" "Observed and Forcast CV" "Forecast"],'FontSize', 14)


figure
subplot(2,1,1)
plot(YTest,'LineWidth',2)
hold on
plot(YPred,'.-','LineWidth',2)
hold off
legend(["Observed" "Forecast"],'FontSize', 14)
ylabel("Cases",'FontSize', 14)
title("Forecast",'FontSize', 14)

subplot(2,1,2)
stem(YPred - YTest,'LineWidth',2)
xlabel("Epochs",'FontSize', 14)
ylabel("Error",'FontSize', 14)
title("RMSE = " + rmse,'FontSize', 14)



%%
% 1. Learing Curve
% 2. Tune number of layers


len = 10;
YPred = sig*YPred(1:len) + mu;

YTest = dataTest(2:len+1);
rmse = sqrt(mean((YPred-YTest).^2));

% If only predict 0, then gives 0.0548 rmse
% using this gives 0.0398 rmse. However, the error start to increase after
% 10 points. 


