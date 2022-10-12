%Left Padding Function
function sequencePadded = rightPad(sequence,sequenceLength)

[numFeatures,numTimeSteps] = size(sequence);

paddingSize = sequenceLength - numTimeSteps;
padding = zeros(numFeatures,paddingSize);

sequencePadded = [sequence padding];

end