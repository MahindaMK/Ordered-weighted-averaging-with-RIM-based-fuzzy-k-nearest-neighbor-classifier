% First example case for the use of OWARIM-FKNN classifier

% Created by Mahinda Mailagaha Kumbure & Pasi Luukka, 11/2024
% ==============================================================

clear all; close all; clc

% Load the data (example data of ionosphere)
load ionosphere
    % X: features
    % Y: cell array of the class labels (g:good and b:bad)

% Convert class labels to numeric 
Y      = categorical(Y);
labels = zeros(length(Y),1);

labels(Y=='g') = 1;
labels(Y=='b') = 2;

data = [X labels];

% cross validation
val = 0.8; % percentage for holdout validation
cv  = cvpartition(size(data,1),'HoldOut', val);
idx = cv.test;

% Separate to training and test data
Xtrain  = data(~idx,1:end-1); % train data with n samples and m features
Ytrain  = data(~idx,end);     % class labels of train samples 
Xtest   = data(idx,1:end-1);  % test data with D samples and m features
Ytest   = data(idx,end);      % class labels of test samples

%parameter initialization
k = 10;      % initialization of the number of nearest neighbors
alpha = 0.5; % parameter alpha for RIM quantifier


% OWARIM-FKNN classifier call
[accuracy, y_predicted] = owarim_fknn(Xtrain, Ytrain, Xtest, Ytest, k, alpha);


% Classification accuracy
classification_accuracy  = accuracy