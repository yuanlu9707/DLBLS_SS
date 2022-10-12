clear all;clc;
%Data loading, data processing.
%fliplr Data
for i=1:numel(XTrain)
    XTrain{i,1}=fliplr(XTrain{i,1});
end
for i=1:numel(XTest10)
    XTest10{i,1}=fliplr(XTest10{i,1});
end
for i=1:numel(XTest11)
    XTest11{i,1}=fliplr(XTest11{i,1});
end
for i=1:numel(XTest12)
    XTest12{i,1}=fliplr(XTest12{i,1});
end
for i=1:numel(XTest1313)
    XTest1313{i,1}=fliplr(XTest1313{i,1});
end
for i=1:numel(XTest14)
    XTest14{i,1}=fliplr(XTest14{i,1});
end
for i=1:numel(XTest13)
    XTest13{i,1}=fliplr(XTest13{i,1});
end
for i=1:numel(XTest_va)
   XTest_va{i,1}=fliplr(XTest_va{i,1});
end
for i=1:numel(XTest_te)
   XTest_te{i,1}=fliplr(XTest_te{i,1});
end

for i=1:numel(YTrain)
    YTrain{i,1}=fliplr(YTrain{i,1});
end
for i=1:numel(YTest10)
    YTest10{i,1}=fliplr(YTest10{i,1});
end
for i=1:numel(YTest11)
    YTest11{i,1}=fliplr(YTest11{i,1});
end
for i=1:numel(YTest12)
    YTest12{i,1}=fliplr(YTest12{i,1});
end
for i=1:numel(YTest1313)
    YTest1313{i,1}=fliplr(YTest1313{i,1});
end
for i=1:numel(YTest14)
    YTest14{i,1}=fliplr(YTest14{i,1});
end
for i=1:numel(YTest13)
    YTest13{i,1}=fliplr(YTest13{i,1});
end
for i=1:numel(YTest_va)
    YTest_va{i,1}=fliplr(YTest_va{i,1});
end
for i=1:numel(YTest_te)
    YTest_te{i,1}=fliplr(YTest_te{i,1});
end
numObservations = numel(XTrain)
numClasses = numel(classes)

numFilters = 512;
filterSize = 7;
dropoutFactor = 0.005;
numBlocks = 4;

Name_input="input"+i;
layer = sequenceInputLayer(numFeatures,'Normalization',"rescale-symmetric",'Name',Name_input);
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    name_cov1="conv1_"+i;
    name_add="add_"+i;
    layers = [
        convolution1dLayer(filterSize,numFilters,'DilationFactor',dilationFactor,'Padding',"causal",'Name',name_cov1)
        layerNormalizationLayer
        spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,'DilationFactor','dilationFactor','Padding',"causal")
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,'DilationFactor','dilationFactor','Padding',"causal")
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(dropoutFactor,'Name',"se_input1")];
    lgraph = addLayers(lgraph,layers);

    layers = [
        globalAveragePooling2dLayer('Name',"avg1")
        fullyConnectedLayer(numFilters/2,'Name',"fc1")
        reluLayer
        fullyConnectedLayer(numFilters,'Name',"fc2")
        sigmoidLayer("Name","sigmoid_1")
        multiplicationLayer(2,"Name","multiplication_1");
        additionLayer(2,'Name',name_add)];
    lgraph = addLayers(lgraph,layers);

    
    % Add and connect layers.
%     lgraph = addLayers(lgraph,layers);
	lgraph = connectLayers(lgraph,"se_input1","avg1");
    lgraph = connectLayers(lgraph,"se_input1","multiplication_1/in2");
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,'Name',"convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end

layers = [
    fullyConnectedLayer(numClasses,'Name',"fc")
    softmaxLayer
    classificationLayer];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,"fc");

%看每次epochs结果
options = trainingOptions("adam", ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',50, ...
    'ValidationData',{XTest_va,YTest_va}, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',0.001, ...
    'LearnRateDropPeriod',20, ... 
    'Verbose',0);

net = trainNetwork(XTrain,YTrain,lgraph,options);

