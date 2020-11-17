clc
clear all
load sisre.mat
load svn_meas.mat
%% Load the Data and Plot
data = svn_meas(33,2:700);
data = data';
epochs = epochs(1:700-1);
figure
plot(epochs,data)
xlabel("Epochs Time")
ylabel("Measurement")
title("Measurement Data Vs Epochs Times")
%% Divide by Training , CV , and Test

% Divide the training set to 0.9 training and 0.1 test
numTimeStepsTrain = floor(0.8*numel(data));
numTimeStepsCV = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataCV = data(numTimeStepsTrain+1:numTimeStepsCV+1);
dataTest = data(numTimeStepsCV+1:end);

%% Mean Normalize the Data for Train, Specify inputs and outputs

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
%% Mean Normalize the Data for CV, Specify input output

% mu = mean(dataCV);
% sig = std(dataCV);

dataCVStandardized = (dataCV - mu) / sig;

XCV = dataCVStandardized(1:end-1);
YCV = dataCVStandardized(2:end);numTimeStepsCV = numel(XCV);
%% Mean Normalize the Data for Test, specify inputs outputs

% mu = mean(dataTest);
% sig = std(dataTest);

dataTestStandardized = (dataTest - mu) / sig;

XTest = dataTestStandardized(1:end-1);
YTest = dataTestStandardized(2:end);
numTimeStepsTest = numel(XTest);

%% Set up the ARIMA Model

sys = arima(7,1,11);
% how many features, 
Md1 = estimate(sys,dataTrainStandardized);

%% Do a Model Forcast for The Train
opt.InitialCondition = XTrain(1);
[YPred,YMSE] = forecast(Md1,length(YTrain),'Y0',dataTrainStandardized);
CostTrain = 1/2*mean((YPred - YTrain).^2);

%% Do a Model Forcast for The CV
[YPredCV,YMSE] = forecast(Md1,length(YCV),'Y0',dataTrainStandardized);
CostCV = 1/2*mean((YPredCV - YCV).^2);
%plot(1:length(dataTrainStandardized),dataTrainStandardized,'b',length(dataTrainStandardized):length(dataTrainStandardized)+length(yf),[dataTrainStandardized(end);yf],'r'), legend('measured','forecasted')
%% Do a Model Forcast for The Test
[YPred,YMSE] = forecast(Md1,length(YTest),'Y0',dataTrainStandardized);
CostTest = 1/2*mean((YPred - YTest).^2);
%%
% YPred = sig*YPred + mu;
% 
% YPredCV = YPred;
% 
% YCV = dataCV(2:end);
% rmse = sqrt(mean((YPred-YCV).^2));
% numTimeStepsCV = numel(XCV);
% 
% 
% figure
% plot(dataTrain(1:end-1),'LineWidth',2)
% hold on
% idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsCV);
% plot(idx,[data(numTimeStepsTrain) YPred'],'.-','LineWidth',2)
% hold off
% xlabel("Epochs Time",'FontSize', 14)
% ylabel("Measurement",'FontSize', 14)
% title("Measurement Data Vs Epochs Times",'FontSize', 14)
% legend(["Observed" "Forecast"],'FontSize', 14)
% 
% figure
% subplot(2,1,1)
% plot(YCV,'LineWidth',2)
% hold on
% plot(YPred,'.-','LineWidth',2)
% hold off
% legend(["Observed" "Forecast"],'FontSize', 14)
% ylabel("Cases",'FontSize', 14)
% title("Forecast",'FontSize', 14)
% 
% subplot(2,1,2)
% stem(YPred - YCV,'LineWidth',2)
% xlabel("Epochs")
% ylabel("Error")
% title("RMSE = " + rmse)
%%
YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2));

figure
plot([dataTrain(1:end-1); dataCV(1:end-1)],'LineWidth',2)
hold on
plot([dataTrain(1:end-1); YPredCV],'LineWidth',2)
idx = numTimeStepsCV+numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsCV+numTimeStepsTest);
%plot(idx,[data(numTimeStepsCV+numTimeStepsTrain) YPred],'.-')
plot(idx,[YPredCV(end); YPred],'.-','LineWidth',2)

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
len = 10;
YPred = sig*YPred(1:len) + mu;

YTest = dataTest(2:len+1);
rmse = sqrt(mean((YPred-YTest).^2))

% rmse is 0.0513
%% Do a Model Forecast for 

%%
% 1. tune the three parameters see how they behave
% 1. motivation 2. 