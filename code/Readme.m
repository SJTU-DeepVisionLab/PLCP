clear;clc;

load('sample.mat')
% parameter
parameter = 0.5;
ga_q = 2;
kp = exp(-1);
gamma = 0.05;
mutual = 5;

%training
par = 1*mean(pdist(train_data));

[a,b] = PLCP(train_data,train_p_target,test_data, test_target,par,parameter, ga_q, kp, gamma, mutual);


