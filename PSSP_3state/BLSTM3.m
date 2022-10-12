%% Build a network
inputSize = 41;
numHiddenUnits1 = 2500;
numHiddenUnits2 = 2500;
numClasses = 3;
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)
    bilstmLayer(numHiddenUnits2,'OutputMode','sequence')
    dropoutLayer(0.2)
    fullyConnectedLayer(20)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',10,...
    'MiniBatchSize',3,...
    'ValidationData',{XTest_va,YTest_va}, ...
    'ValidationFrequency',3900, ...
    'GradientThreshold',1, ...
    'L2Regularization',0.000001,...
    'InitialLearnRate',0.0001, ...%%%%%%
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',8, ... 
    'SequenceLength','longest', ...
    'SequencePaddingValue',0, ...
    'shuffle','never', ...
    'Verbose',true);
net = trainNetwork(XTrain,YTrain,layers,options);

%% Feature extraction
lstm20_xtrain = activations(net,XTrain,6,'MiniBatchSize',1);
lstm20_casp10 = activations(net,XTest10,6,'MiniBatchSize',1);
lstm20_casp11 = activations(net,XTest11,6,'MiniBatchSize',1);
lstm20_casp12 = activations(net,XTest12,6,'MiniBatchSize',1);
lstm20_casp13 = activations(net,XTest1313,6,'MiniBatchSize',1);
lstm20_casp14 = activations(net,XTest1313,6,'MiniBatchSize',1);
lstm20_cb513 = activations(net,XTest13,6,'MiniBatchSize',1);
lstm20_XTest_va = activations(net,XTest_va,6,'MiniBatchSize',1);
lstm20_XTest_te = activations(net,XTest_te,6,'MiniBatchSize',1);
save('blstm3cullpdb15fe2500.mat','lstm20_xtrain','lstm20_casp10','lstm20_casp11','lstm20_casp12','lstm20_casp13','lstm20_casp14','lstm20_cb513','lstm20_XTest_va','lstm20_XTest_te');