%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code employs progressive learning technique to multi-class
%classification using extreme learning machines 
%Author: Rajasekar Venkatesan 
%Paper Reference: A novel progressive learning technique for
%multi-class classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

for Trial = 1:10
    
    display('Starting New Trial');
    
    N0 = 50;                        %Initial Block
    ActivationFunction = 'sig';     %Activation Function
    Block = 5;                      %Number of samples in each sequential learning block
    nHiddenNeurons = 100;           %Number of Hidden Layer Neurons
    
    load('waveform_dataset.mat')             %load dataset
    
    %Load Training and Testing data from dataset
    % train_data = traindata;
    % test_data = testdata;
    
    %Separate the input(features) from the output(target class)
    Train_output=train_data(:,1);
    Train_input=train_data(:,2:size(train_data,2));
    Test_output=test_data(:,1);
    Test_input=test_data(:,2:size(test_data,2));
    
    clear test_data train_data;
    
    nTrainingData=size(Train_input,1);
    nTestingData=size(Test_input,1);
    nInputNeurons=size(Train_input,2);
    
    %Randomize initialization of input weights and bias
    input_weight = rand(nHiddenNeurons,nInputNeurons)*2-1;
    bias = rand(1,nHiddenNeurons)*2-1;
    
    %%% Initial Block
    de = N0;
    blocksize = N0;
    data_input = Train_input(1:N0,:);
    data_output = Train_output(1:N0, :);
    label = unique(data_output);
    nclass = length(label);
    nOutputNeurons = nclass;
    
    
    data_output_bipolar = to_bipolar(data_output,label);
    H = SigActFun(data_input,input_weight,bias);
    M = pinv(H' * H);
    beta = pinv(H) * data_output_bipolar;
    beta2a = pinv(H);
    
    
    %%%%%%%%%%%%%% Sequential Learning Begins
    testplot = [];
    while de+Block < nTrainingData
        %  k = k+1;
        ds = de+1;
        de = de+Block;
        data_input = Train_input(ds:de,:);
        data_output = Train_output(ds:de,:);
        blabel = unique(data_output);
        
        new_class = setdiff(blabel,label);
        inc_class = length(new_class);
        
        if(inc_class ~= 0)
            label = union(label,blabel);
            nclass = length(label);
            disp('Number of classes has increased');
            disp('Adapting to new class and continuing to learn');
            nOutputNeurons = nclass;
            
            deltabetatemp = ones(blocksize,inc_class) * -1;
            deltabeta = beta2a * deltabetatemp;
            beta = [beta deltabeta];
        end
        
        blocksize = Block;
        data_output_bipolar = to_bipolar(data_output,label);
        H = SigActFun(data_input,input_weight,bias);
        M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M;
        beta1 = beta;
        beta2a = M * H';
        beta2 = beta2a * (data_output_bipolar - H * beta);
        beta = beta1+beta2;
        
        
        TestingAccuracy = Test_result(Test_input,input_weight,bias,beta,Test_output,label);
        testplot = [testplot;TestingAccuracy];
        
    end
    
    figure
    set(gcf,'color','w');
    tt = 1:length(testplot);
    plot(N0+(tt.*Block),testplot)
    
    
    TestingAccuracy = Test_result(Test_input,input_weight,bias,beta,Test_output,label)
    TA(Trial) = TestingAccuracy;
end
figure
stem(TA)
