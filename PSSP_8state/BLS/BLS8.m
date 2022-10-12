%%%%%%%%%%%%%%%%%%%%%%%%This is the demo for the bls models including the
%%%%%%%%%%%%%%%%%%%%%%%%proposed incremental learning algorithms when the
%%%%%%%%%%%%%%%%%%%%%%%%memory of PC is not sufficient%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%load the dataset MNIST dataset%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc
warning off all;
format compact;
% load datasets;

% load BLS_test_vate.mat;            
          XTest_va=[];
          YTest_va=[];
          XTest_te=[];
          YTest_te=[];
%%%%%%%%%%%%%%%%%%%%This is the model of broad learning system for%%%%%%
%%%%%%%%%%%%%%%%%%%% increment of m input patterns %%%%%%%%%%%%%%%%%%%%%%%%
train_xf=XTrain;train_yf=YTrain;
clear XTrain;clear YTrain;
train_x=train_xf(1:78501,:);train_y=train_yf(1:78501,:); % the selected input patterns of int incremental learning
C = 2^-30;%the regularization parameter for sparse regualarization
s = .8;%the shrinkage parameter for enhancement nodes
N11=40;%feature nodes  per window
N2=40;% number of windows of feature nodes
N33=10000;% number of enhancement nodes
epochs=1;% number of epochs 
m=100000;%number of added input patterns per increment step
l=28;% steps of incremental learning
N1=N11; N3=N33;  
for i=1:epochs
    [Accuracy] = bls_train_input_pro_pred78501(train_xf,train_yf,train_x,train_y,XTest10,YTest10,XTest11,YTest11,XTest12,YTest12,XTest1313,YTest1313,XTest14,YTest14,XTest13,YTest13,XTest_va,YTest_va,XTest_te,YTest_te,s,C,N1,N2,N3,m,l);     
end