% Second example case for the use of OWARIM-FKNN classifier

% Created by Mahinda Mailagaha Kumbure & Pasi Luukka, 11/2024
% ==============================================================

close all; clear all; clc

% load the data
original_data = readtable('appendicitis.dat'); 


% convert table data to an array
data = table2array(original_data);
data(data(:,end)==0,end)=2; % replace class 0 with 2

% cross validation
val = 0.8; % Percentage for holdout validation
cv  = cvpartition(size(data,1),'HoldOut', val);
idx = cv.test;

% separate to training and test data
Xtrain  = data(~idx,1:end-1); % train data with n patterns and m features
Ytrain  = data(~idx,end); % class labels of train patters 

Xtest   = data(idx,1:end-1); % test data with D patterns and m features
Ytest   = data(idx,end); % class labels of test patterns

% parameter initialization
k = 10;      % initialization of the number of nearest neighbors
alpha = 0.5; % parameter alpha for RIM quantifier


% OWARIM-FKNN classifier call
[accuracy, y_predicted] = owarim_fknn(Xtrain, Ytrain, Xtest, Ytest, k, alpha);

        
classification_accuracy = accuracy