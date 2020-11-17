clc
clear all
load sisre.mat
load svn_meas.mat
rng(0)
% Do not use costTrain! Only do Analysis on costCV
%% Set up the Baseline
numofData = 700;
p = 7;%5;
d = 1;% 5;
q = 11;%5;
Percent = 0.8;
CostCV = Cost(svn_meas,numofData,p,d,q,Percent);
%%
cost = 10000;
for numofdata = 700%[500 600 700 800 1000 2000 3000]
    for percent =0.8 %0.65:0.05:0.85
        for p = 1:2:11
            for d = 1:2:11
                for q = 1:2:11
                    try
                    C = Cost(svn_meas,numofData,p,d,q,Percent);
                   
                    catch
                        warning('Problem using function.  Assigning a value of 0.');
                    end
                    if C < cost
                        cost = C;
                        NumofData = numofdata;
                        Percent = percent;
                        P = p;
                        D = d;
                        Q = q;
                        
                    end
                end
                
            end
        end
    end
end


%% Varying number of Data DO not do this! For comparison, we should use 700 data points!
VaryingnumofData = [500 600 700 800 1000 2000 3000];
ind = 0;
CostCV = zeros(length(VaryingnumofData),1);
for i = VaryingnumofData
    ind = ind + 1;
    costCV = Cost(svn_meas,i,p,d,q,Percent);
    CostCV(ind) = costCV;
end
figure
plot(VaryingnumofData(1:end-1),CostCV(1:end-1),'LineWidth',2)
xlabel('Number of Data','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Comparing The Effect of Amount of Data on Cost Function','FontSize', 14)
%% Varying p
VaryingP = [0 1 2 4 5 6 7 8 10];
ind = 0;
CostCV = zeros(length(VaryingP),1);
for i = VaryingP
    ind = ind + 1;
    costCV = Cost(svn_meas,numofData,i,d,q,Percent);
    CostCV(ind) = costCV;
end
figure
plot(VaryingP,CostCV,'LineWidth',2)
xlabel('Order of AR Model','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Comparing The Effect of Order of AR Model on Cost Function','FontSize', 14)
%% Varying d
VaryingD = [1 2 3 5 6 7 8 9 10];%4
ind = 0;
CostCV = zeros(length(VaryingD),1);
for i = VaryingD
    ind = ind + 1;
    costCV = Cost(svn_meas,numofData,p,i,q,Percent);
    CostCV(ind) = costCV;
end
figure
plot(VaryingD,CostCV,'LineWidth',2)
xlabel('Order of Differencing','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Comparing The Effect of Order of Differecing on Cost Function','FontSize', 14)
%% Varying q
VaryingQ = 0:1:10;
ind = 0;
CostCV = zeros(length(VaryingQ),1);
for i = VaryingQ
    ind = ind + 1;
    costCV = Cost(svn_meas,numofData,p,d,i,Percent);
    CostCV(ind) = costCV;
end
figure
plot(VaryingQ,CostCV,'LineWidth',2)
xlabel('Order of MA Model','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Comparing The Effect of Order of MA Model on Cost Function','FontSize', 14)
%% Varying Percent Do not do this! for comparision purorse, we should use 0.8
VaryingPercent = 0.65:0.05:0.85;
ind = 0;
CostCV = zeros(length(VaryingPercent),1);
for i = VaryingPercent
    ind = ind + 1;
    costCV = Cost(svn_meas,numofData,p,d,q,i);
    CostCV(ind) = costCV;
end
figure
plot(VaryingPercent,CostCV,'LineWidth',2)
xlabel('Fraction to Train','FontSize', 14)
ylabel('Cost Value','FontSize', 14)
title('Comparing The Effect of Training Fraction on Cost Function','FontSize', 14)
%% Load the Data and Plot
BaselineCost = Cost(svn_meas,numofData,p,d,q,Percent);

%%
function CostCV = Cost(svn_meas,numofData,p,d,q,Percent)
data = svn_meas(33,2:numofData);
data = data';

%% Divide by Training , CV , and Test

% Divide the training set to 0.9 training and 0.1 test
numTimeStepsTrain = floor(Percent*numel(data));
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

%mu = mean(dataCV);
%sig = std(dataCV);

dataCVStandardized = (dataCV - mu) / sig;

XCV = dataCVStandardized(1:end-1);
YCV = dataCVStandardized(2:end);
%% Mean Normalize the Data for Test, specify inputs outputs

mu = mean(dataTest);
sig = std(dataTest);

dataTestStandardized = (dataTest - mu) / sig;

XTest = dataTestStandardized(1:end-1);
YTest = dataTestStandardized(2:end);

%% Set up the ARIMA Model
sys = arima(p,d,q);
% how many features, 
Md1 = estimate(sys,dataTrainStandardized);

%% Do a Model Forcast for The Train
%opt = forecastOptions('InitialCondition',XTrain(1));
[YPred,YMSE] = forecast(Md1,length(YTrain),'Y0',dataTrainStandardized);
% CostTrain = 1/2*mean((YMSE).^2);
% YMSE1 = 1/2*mean((YMSE).^2)
%% Do a Model Forcast for The CV
[YPred,YMSE] = forecast(Md1,length(YCV),'Y0',dataTrainStandardized);
CostCV = 1/2*mean((YPred - YCV).^2);
%YMSE2 = 1/2*mean((YMSE).^2);
%plot(1:length(dataTrainStandardized),dataTrainStandardized,'b',length(dataTrainStandardized):length(dataTrainStandardized)+length(yf),[dataTrainStandardized(end);yf],'r'), legend('measured','forecasted')
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
% plot(dataTrain(1:end-1))
% hold on
% idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsCV);
% plot(idx,[data(numTimeStepsTrain) YPred'],'.-')
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
end

%% Do a Model Forecast for 

%%
% 1. tune the three parameters see how they behave
% 1. motivation 2. 