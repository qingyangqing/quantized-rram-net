clc;clear;close all;
%% 2 layer example
% patterns and targets generation
load mnist_train.mat
load mnist_train_label.mat
NumTotal=length(mnist_train_label);
imsize=10; % resize images 
imresolution=4; % input resolution
mnist_train=patternResize(mnist_train, imsize, imresolution);
mnist_train_label=labelResize(mnist_train_label);
% initilize parameters
% learning rate
alpha=0.001;
hidden_size=64; % layer 1 size
W1=rands(hidden_size,imsize^2)*0.1; W2=rands(10,hidden_size)*0.1;
W1_q=W1; W2_q=W2;
%% BP MLP training 
% generate training set and validation set
temp=randperm(NumTotal,NumTotal*0.8);
train_set=mnist_train(:,temp);
train_label=mnist_train_label(:,temp);
temp=setdiff(linspace(1,NumTotal,NumTotal),temp);
val_set=mnist_train(:,temp);
val_label=mnist_train_label(:,temp);
% rram candidate
rram_level=8;
% g=1./linspace(10,500,rram_level);
% [gg, gg_mat]=ggGenerate(g);
g=linspace(0.02,1,rram_level);
gg=[-g 0 g];
% start traning 
accuracy_train=[];
accuracy_val=[];
for iteration=1:50
    for i=1:length(train_set)
        N1=W1_q*train_set(:,i);
        A1=poslin(N1);% ReLU function
        N2=W2_q*A1;
        A2=softmax(N2);
        % back propagation
        % second layer
        S2=A2-train_label(:,i);
        % first layer
        F1=diag(N1>0);
        S1=F1*W2_q'*S2;
        % update weights
        W2=W2-alpha*S2*A1';
        W1=W1-alpha*S1*train_set(:,i)';
        % quantize
        [W1_q]=WeightQuantize(W1, gg);
        [W2_q]=WeightQuantize(W2, gg);
    end
    disp('iteration=')
    disp(iteration)
    % traning accuracy 
    N1=W1_q*train_set;
    A1=poslin(N1);
    N2=W2_q*A1;
    % A2=softmax(N2);
    [~,j]=max(train_label);
    [~,k]=max(N2);
    accuracy=length(find(j==k))/length(train_set);
    fprintf('Accuracy for training: %1f \n', accuracy);
    accuracy_train=[accuracy_train, accuracy];
    % validation accuracy
    N1=W1_q*val_set;
    A1=poslin(N1);
    N2=W2_q*A1;
    % A2=softmax(N2);
    [~,j]=max(val_label);
    [~,k]=max(N2);
    accuracy=length(find(j==k))/length(val_set);
    fprintf('Accuracy for validation: %1f \n', accuracy);
    accuracy_val=[accuracy_val, accuracy];
end 
%% test 
load mnist_test.mat
load mnist_test_label.mat
mnist_test=patternResize(mnist_test, imsize, imresolution);
mnist_test_label=labelResize(mnist_test_label);
N1=W1_q*mnist_test;
A1=poslin(N1);
N2=W2_q*A1;
% A2=softmax(N2);
[~,j]=max(mnist_test_label);
[~,k]=max(N2);
accuracy=length(find(j==k))/length(mnist_test);
fprintf('Accuracy for testing: %1f \n', accuracy);
