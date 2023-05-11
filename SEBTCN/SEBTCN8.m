%forwardSETCNnumObservations = numel(XTrain);numClasses = numel(classes);numFeatures = size(s.XTrain{1},1);numFilters = 512;filterSize = 7;dropoutFactor = 0.005;numBlocks = 4;layer = [sequenceInputLayer(numFeatures,Normalization="rescale-symmetric",Name="input")];lgraph = layerGraph(layer);outputName = "input";for i = 1:numBlocks    dilationFactor = 2^(i-1);    name_cov1="conv1_"+i;    name_add="add_"+i;    layers = [        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)        layerNormalizationLayer        reluLayer        spatialDropoutLayer(dropoutFactor)        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")        layerNormalizationLayer        reluLayer        spatialDropoutLayer(dropoutFactor)        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")        layerNormalizationLayer        reluLayer        spatialDropoutLayer(dropoutFactor,Name="se_input1"+i)        globalAveragePooling1dLayer(Name="avg1_1"+i)        fullyConnectedLayer(numFilters,Name="fc1_1"+i)        reluLayer        fullyConnectedLayer(numFilters,Name="fc1_2"+i)        sigmoidLayer(Name="sigmoid"+i)        functionLayer(@(X) dlarray(X,"CBT"),Formattable=true,Name="m1"+i)        multiplicationLayer(2,Name="multiplication"+i)        additionLayer(2,Name="add_"+i)        reluLayer(Name="block"+i)        ];    lgraph = addLayers(lgraph,layers);% Add and connect layers.    lgraph = connectLayers(lgraph,"se_input1"+i,"multiplication"+ i +"/in2");    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);%     Skip connection.    if i == 1        % Include convolution in first skip connection.        layer = convolution1dLayer(1,numFilters,Name="convSkip");        lgraph = addLayers(lgraph,layer);        lgraph = connectLayers(lgraph,outputName,"convSkip");        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");    else        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");    end        % Update layer output name.%     outputName = "add_" + i;    outputName = "block" + i;endlayers = [    fullyConnectedLayer(numClasses,Name="fc20")    fullyConnectedLayer(numClasses,Name="fc")    softmaxLayer    classificationLayer];lgraph = addLayers(lgraph,layers);lgraph = connectLayers(lgraph,outputName,"fc");% figure% plot(lgraph)% title("SETCN")% % xlim([0.95 4.23])% ylim([1.1 26.2])options = trainingOptions("adam", ...    ExecutionEnvironment='gpu', ...    MaxEpochs=50, ...    ValidationData={XTest_va,YTest_va}, ...    miniBatchSize=32, ...    InitialLearnRate=0.001, ...    LearnRateDropPeriod=20, ...     Plots="training-progress", ...    Verbose=0);net = trainNetwork(XTrain,TTrain,lgraph,options);%feature_extractionforXTrain = activations(net,XTrain,"fc20",'MiniBatchSize',1);forXTest10 = activations(net,XTest10,"fc20",'MiniBatchSize',1);forXTest11 = activations(net,XTest11,"fc20",'MiniBatchSize',1);forXTest12 = activations(net,XTest12,"fc20",'MiniBatchSize',1);forXTest1313 = activations(net,XTest1313,"fc20",'MiniBatchSize',1);forXTest14 = activations(net,XTest14,"fc20",'MiniBatchSize',1);forXTest13 = activations(net,XTest13,"fc20",'MiniBatchSize',1);forXTest_va = activations(net,XTest_va,"fc20",'MiniBatchSize',1);forXTest_te = activations(net,XTest_te,"fc20",'MiniBatchSize',1);%backSETCN%fliplr Datafor i=1:numel(XTrain)    XTrain{i,1}=fliplr(XTrain{i,1});endfor i=1:numel(XTest10)    XTest10{i,1}=fliplr(XTest10{i,1});endfor i=1:numel(XTest11)    XTest11{i,1}=fliplr(XTest11{i,1});endfor i=1:numel(XTest12)    XTest12{i,1}=fliplr(XTest12{i,1});endfor i=1:numel(XTest1313)    XTest1313{i,1}=fliplr(XTest1313{i,1});endfor i=1:numel(XTest14)    XTest14{i,1}=fliplr(XTest14{i,1});endfor i=1:numel(XTest13)    XTest13{i,1}=fliplr(XTest13{i,1});endfor i=1:numel(XTest_va)   XTest_va{i,1}=fliplr(XTest_va{i,1});endfor i=1:numel(XTest_te)   XTest_te{i,1}=fliplr(XTest_te{i,1});endfor i=1:numel(YTrain)    YTrain{i,1}=fliplr(YTrain{i,1});endfor i=1:numel(YTest10)    YTest10{i,1}=fliplr(YTest10{i,1});endfor i=1:numel(YTest11)    YTest11{i,1}=fliplr(YTest11{i,1});endfor i=1:numel(YTest12)    YTest12{i,1}=fliplr(YTest12{i,1});endfor i=1:numel(YTest1313)    YTest1313{i,1}=fliplr(YTest1313{i,1});endfor i=1:numel(YTest14)    YTest14{i,1}=fliplr(YTest14{i,1});endfor i=1:numel(YTest13)    YTest13{i,1}=fliplr(YTest13{i,1});endfor i=1:numel(YTest_va)    YTest_va{i,1}=fliplr(YTest_va{i,1});endfor i=1:numel(YTest_te)    YTest_te{i,1}=fliplr(YTest_te{i,1});endnumObservations = numel(XTrain);numClasses = numel(classes);numFeatures = size(s.XTrain{1},1);numFilters = 512;filterSize = 7;dropoutFactor = 0.005;numBlocks = 4;layer = [sequenceInputLayer(numFeatures,Normalization="rescale-symmetric",Name="input")];lgraph = layerGraph(layer);outputName = "input";for i = 1:numBlocks    dilationFactor = 2^(i-1);    name_cov1="conv1_"+i;    name_add="add_"+i;    layers = [        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)        layerNormalizationLayer        reluLayer        spatialDropoutLayer(dropoutFactor)        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")        layerNormalizationLayer        reluLayer        spatialDropoutLayer(dropoutFactor)        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")        layerNormalizationLayer        reluLayer        spatialDropoutLayer(dropoutFactor,Name="se_input1"+i)        globalAveragePooling1dLayer(Name="avg1_1"+i)        fullyConnectedLayer(numFilters,Name="fc1_1"+i)        reluLayer        fullyConnectedLayer(numFilters,Name="fc1_2"+i)        sigmoidLayer(Name="sigmoid"+i)        functionLayer(@(X) dlarray(X,"CBT"),Formattable=true,Name="m1"+i)        multiplicationLayer(2,Name="multiplication"+i)        additionLayer(2,Name="add_"+i)        reluLayer(Name="block"+i)        ];    lgraph = addLayers(lgraph,layers);% Add and connect layers.    lgraph = connectLayers(lgraph,"se_input1"+i,"multiplication"+ i +"/in2");    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);%     Skip connection.    if i == 1        % Include convolution in first skip connection.        layer = convolution1dLayer(1,numFilters,Name="convSkip");        lgraph = addLayers(lgraph,layer);        lgraph = connectLayers(lgraph,outputName,"convSkip");        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");    else        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");    end        % Update layer output name.%     outputName = "add_" + i;    outputName = "block" + i;endlayers = [    fullyConnectedLayer(numClasses,Name="fc20")    fullyConnectedLayer(numClasses,Name="fc")    softmaxLayer    classificationLayer];lgraph = addLayers(lgraph,layers);lgraph = connectLayers(lgraph,outputName,"fc");% figure% plot(lgraph)% title("SETCN")% % xlim([0.95 4.23])% ylim([1.1 26.2])options = trainingOptions("adam", ...    ExecutionEnvironment='gpu', ...    MaxEpochs=50, ...    ValidationData={XTest_va,YTest_va}, ...    miniBatchSize=32, ...    InitialLearnRate=0.001, ...    LearnRateDropPeriod=20, ...     Plots="training-progress", ...    Verbose=0);net = trainNetwork(XTrain,TTrain,lgraph,options);%feature_extractionbackXTrain = activations(net,XTrain,"fc20",'MiniBatchSize',1);backXTest10 = activations(net,XTest10,"fc20",'MiniBatchSize',1);backXTest11 = activations(net,XTest11,"fc20",'MiniBatchSize',1);backXTest12 = activations(net,XTest12,"fc20",'MiniBatchSize',1);backXTest1313 = activations(net,XTest1313,"fc20",'MiniBatchSize',1);backXTest14 = activations(net,XTest14,"fc20",'MiniBatchSize',1);backXTest13 = activations(net,XTest13,"fc20",'MiniBatchSize',1);backXTest_va = activations(net,XTest_va,"fc20",'MiniBatchSize',1);backXTest_te = activations(net,XTest_te,"fc20",'MiniBatchSize',1);%%SEBTCN 1D-blockfor i=1:numel(forXTrain)    XTrain{i,1}=forXTrain{i,1}+fliplr(backXTrain{i,1});endclear forXTrain;clear backXTrain;for i=1:numel(forXTest_va)    XTest_va{i,1}=forXTest_va{i,1}+fliplr(backXTest_va{i,1});endclear forXTest_va;clear backXTest_va;for i=1:numel(forXTest_te)    XTest_te{i,1}=forXTest_te{i,1}+fliplr(backXTest_te{i,1});endclear forXTest_te;clear backXTest_te;for i=1:numel(forXTest10)    XTest10{i,1}=forXTest10{i,1}+fliplr(backXTest10{i,1});endclear forXTest10;clear backXTest10;for i=1:numel(forXTest11)    XTest11{i,1}=forXTest11{i,1}+fliplr(backXTest11{i,1});endclear forXTest11;clear backXTest11;for i=1:numel(forXTest12)    XTest12{i,1}=forXTest12{i,1}+fliplr(backXTest12{i,1});endclear forXTest12;clear backXTest12;for i=1:numel(forXTest1313)    XTest1313{i,1}=forXTest1313{i,1}+fliplr(backXTest1313{i,1});endclear forXTest1313;clear backXTest1313;for i=1:numel(forXTest14)    XTest14{i,1}=forXTest14{i,1}+fliplr(backXTest14{i,1});endclear forXTest14;clear backXTest14;for i=1:numel(forXTest13)    XTest13{i,1}=forXTest13{i,1}+fliplr(backXTest13{i,1});endclear forXTest13;clear backXTest13;%%Build the SEBTCN 1Dcov prediction.numObservations = numel(XTrain);class=['C','S','T','H','G','I','E','B'];classes = cell(8,1);for c=1:8    classes{c,1}=class(c);endnumClasses = numel(classes);numBlocks = 1;numFilters = 512;filterSize = 7;dropoutFactor = 0.05;hyperparameters = struct;hyperparameters.NumBlocks = numBlocks;hyperparameters.DropoutFactor = dropoutFactor;numInputChannels = 20;parameters = struct;numChannels = numInputChannels;for k = 1:numBlocks    parametersBlock = struct;    blockName = "Block"+k;        weights = initializeGaussian([filterSize, numChannels, numFilters]);    bias = zeros(numFilters, 1, 'single');    parametersBlock.Conv1.Weights = dlarray(weights);    parametersBlock.Conv1.Bias = dlarray(bias);        weights = initializeGaussian([filterSize, numFilters, numFilters]);    bias = zeros(numFilters, 1, 'single');    parametersBlock.Conv2.Weights = dlarray(weights);    parametersBlock.Conv2.Bias = dlarray(bias);        weights = initializeGaussian([filterSize, numFilters, numFilters]);    bias = zeros(numFilters, 1, 'single');    parametersBlock.Conv3.Weights = dlarray(weights);    parametersBlock.Conv3.Bias = dlarray(bias);        % If the input and output of the block have different numbers of channels,     % then add a convolution with filter size 1.    if numChannels ~= numFilters        weights = initializeGaussian([1, numChannels, numFilters]);        bias = zeros(numFilters, 1, 'single');        parametersBlock.Conv4.Weights = dlarray(weights);        parametersBlock.Conv4.Bias = dlarray(bias);    end    numChannels = numFilters;        parameters.(blockName) = parametersBlock;endweights = initializeGaussian([numClasses,20]);bias = zeros(numClasses,1,'single');weights20 = initializeGaussian([20,numChannels]);bias20 = zeros(20,1,'single');parameters.FC.Weights = dlarray(weights);parameters.FC.Bias = dlarray(bias);parameters.FC.Weights20 = dlarray(weights20);parameters.FC.Bias20 = dlarray(bias20);maxEpochs = 10;miniBatchSize = 32;initialLearnRate = 0.001;learnRateDropFactor = 0.1;learnRateDropPeriod = 30;gradientThreshold = 1;executionEnvironment = "gpu";plots = "training-progress";%%Train ModellearnRate = initialLearnRate;trailingAvg = [];trailingAvgSq = [];iteration = 0;flag=0;if rem((numObservations./miniBatchSize),1)==0    numIterationsPerEpoch = numObservations./miniBatchSize;else    numIterationsPerEpoch = floor(numObservations./miniBatchSize)+1;    flag=1;endstart = tic;aaa=[];% Loop over epochs.for epoch = 1:maxEpochs        %Shuffle the data.    idx = randperm(numObservations);    XTrain = XTrain(idx);    YTrain = YTrain(idx);         % Loop over mini-batches.    for i = 1:numIterationsPerEpoch        iteration = iteration + 1;                % Read mini-batch of data and apply the transformSequence preprocessing function.        if i==numIterationsPerEpoch&&flag==1            idx = (i-1)*miniBatchSize+1:numObservations;        else            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;        end                [X,Y,numTimeSteps] = transformSequences(XTrain(idx),YTrain(idx));                % Convert to dlarray.        dlX = dlarray(X);                % If training on a GPU, convert data to gpuArray.        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"            dlX = gpuArray(dlX);        end                % Evaluate the model gradients and loss using dlfeval.        [gradients, loss] = dlfeval(@modelGradients,dlX,Y,parameters,hyperparameters,numTimeSteps);                % Clip the gradients.        gradients = dlupdate(@(g) thresholdL2Norm(g,gradientThreshold),gradients);                % Update the network parameters using the Adam optimizer.        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...            trailingAvg, trailingAvgSq, iteration, learnRate);            end        % Reduce the learning rate after learnRateDropPeriod epochs	if epoch==40    learnRateDropPeriod=40;    end    if mod(epoch,learnRateDropPeriod) == 0        learnRate = learnRate*learnRateDropFactor;    end        %%Test Modelfor test_i=1:8    if test_i==1        XTest=XTest10;        YTest=YTest10;        str_len=str10_len;    elseif test_i==2        XTest=XTest11;        YTest=YTest11;        str_len=str11_len;    elseif test_i==3        XTest=XTest12;        YTest=YTest12;        str_len=str12_len;    elseif test_i==4        XTest=XTest1313;        YTest=YTest1313;        str_len=str1313_len;    elseif test_i==5        XTest=XTest14;        YTest=YTest14;        str_len=str14_len;    elseif test_i==6        XTest=XTest13;        YTest=YTest13;        str_len=str13_len;    elseif test_i==7        XTest=XTest_va;        YTest=YTest_va;        str_len=str_len_v;    else        XTest=XTest_te;        YTest=YTest_te;        str_len=str_len_t;    end    numObservationsTest = numel(XTest);[X,Y] = transformSequences(XTest,YTest);dlXTest = dlarray(X);doTraining = false;dlYPred = model(dlXTest,parameters,hyperparameters,doTraining);YPred = gather(extractdata(dlYPred));sum_true=0;for i = 1:numObservationsTest    [~,idxPred] = max(YPred(:,i,:),[],1);    [~,idxTest] = max(Y(:,i,:),[],1);     sum_true = sum_true + sum(idxPred(1,1,1:str_len(i)) == idxTest(1,1,1:str_len(i)));endaccuracy(test_i) = sum_true/num_test(test_i);enddisp([epoch,accuracy]);aaa(epoch,:)=accuracy;end