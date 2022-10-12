%Masked Cross-Entropy Loss Function
function loss = maskedCrossEntropyLoss(dlY, dlT, numTimeSteps)

numObservations = size(dlY,2);
loss = dlarray(zeros(1,numObservations,'like',dlY));

for i = 1:numObservations
    idx = 1:numTimeSteps(i);
    loss(i) = crossentropy(dlY(:,i,idx),dlT(:,i,idx),'DataFormat','CBT');
end

end