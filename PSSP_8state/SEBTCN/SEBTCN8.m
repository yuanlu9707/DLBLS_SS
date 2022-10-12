%%Build the SEBTCN 1Dcov prediction.
numObservations = numel(XTrain);
class=['C','S','T','H','G','I','E','B'];
classes = cell(8,1);
for c=1:8
    classes{c,1}=class(c);
end
numClasses = numel(classes);

numBlocks = 1;
numFilters = 512;
filterSize = 7;
dropoutFactor = 0.05;

hyperparameters = struct;
hyperparameters.NumBlocks = numBlocks;
hyperparameters.DropoutFactor = dropoutFactor;

numInputChannels = 20;

parameters = struct;
numChannels = numInputChannels;

for k = 1:numBlocks
    parametersBlock = struct;
    blockName = "Block"+k;
    
    weights = initializeGaussian([filterSize, numChannels, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv1.Weights = dlarray(weights);
    parametersBlock.Conv1.Bias = dlarray(bias);
    
    weights = initializeGaussian([filterSize, numFilters, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv2.Weights = dlarray(weights);
    parametersBlock.Conv2.Bias = dlarray(bias);
    
    weights = initializeGaussian([filterSize, numFilters, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv3.Weights = dlarray(weights);
    parametersBlock.Conv3.Bias = dlarray(bias);
    
    % If the input and output of the block have different numbers of channels, 
    % then add a convolution with filter size 1.
    if numChannels ~= numFilters
        weights = initializeGaussian([1, numChannels, numFilters]);
        bias = zeros(numFilters, 1, 'single');
        parametersBlock.Conv4.Weights = dlarray(weights);
        parametersBlock.Conv4.Bias = dlarray(bias);
    end
    numChannels = numFilters;
    
    parameters.(blockName) = parametersBlock;
end

weights = initializeGaussian([numClasses,20]);
bias = zeros(numClasses,1,'single');
weights20 = initializeGaussian([20,numChannels]);
bias20 = zeros(20,1,'single');

parameters.FC.Weights = dlarray(weights);
parameters.FC.Bias = dlarray(bias);
parameters.FC.Weights20 = dlarray(weights20);
parameters.FC.Bias20 = dlarray(bias20);

maxEpochs = 1;
miniBatchSize = 32;
initialLearnRate = 0.001;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 30;
gradientThreshold = 1;

executionEnvironment = "gpu";
plots = "training-progress";

%%Train Model
learnRate = initialLearnRate;
trailingAvg = [];
trailingAvgSq = [];

iteration = 0;
flag=0;
if rem((numObservations./miniBatchSize),1)==0
    numIterationsPerEpoch = numObservations./miniBatchSize;
else
    numIterationsPerEpoch = floor(numObservations./miniBatchSize)+1;
    flag=1;
end

start = tic;
aaa=[];
% Loop over epochs.
for epoch = 1:maxEpochs
    
    %Shuffle the data.
    idx = randperm(numObservations);
    XTrain = XTrain(idx);
    YTrain = YTrain(idx);
     
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and apply the transformSequence preprocessing function.
        if i==numIterationsPerEpoch&&flag==1
            idx = (i-1)*miniBatchSize+1:numObservations;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        
        [X,Y,numTimeSteps] = transformSequences(XTrain(idx),YTrain(idx));
        
        % Convert to dlarray.
        dlX = dlarray(X);
        
        % If training on a GPU, convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients and loss using dlfeval.
        [gradients, loss] = dlfeval(@modelGradients,dlX,Y,parameters,hyperparameters,numTimeSteps);
        
        % Clip the gradients.
        gradients = dlupdate(@(g) thresholdL2Norm(g,gradientThreshold),gradients);
        
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg, trailingAvgSq, iteration, learnRate);
        
    end
    
    % Reduce the learning rate after learnRateDropPeriod epochs
	if epoch==40
    learnRateDropPeriod=40;
    end
    if mod(epoch,learnRateDropPeriod) == 0
        learnRate = learnRate*learnRateDropFactor;
    end
    

    
%%Test Model
for test_i=1:8

    if test_i==1
        XTest=XTest10;
        YTest=YTest10;
        str_len=str10_len;
    elseif test_i==2
        XTest=XTest11;
        YTest=YTest11;
        str_len=str11_len;
    elseif test_i==3
        XTest=XTest12;
        YTest=YTest12;
        str_len=str12_len;
    elseif test_i==4
        XTest=XTest1313;
        YTest=YTest1313;
        str_len=str1313_len;
    elseif test_i==5
        XTest=XTest14;
        YTest=YTest14;
        str_len=str14_len;
    elseif test_i==6
        XTest=XTest13;
        YTest=YTest13;
        str_len=str13_len;
    elseif test_i==7
        XTest=XTest_va;
        YTest=YTest_va;
        str_len=str_len_v;
    else
        XTest=XTest_te;
        YTest=YTest_te;
        str_len=str_len_t;
    end
    
numObservationsTest = numel(XTest);

[X,Y] = transformSequences(XTest,YTest);
dlXTest = dlarray(X);

doTraining = false;
dlYPred = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

end

end