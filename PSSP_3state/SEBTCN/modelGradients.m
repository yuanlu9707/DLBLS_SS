%Model Gradients Function
function [gradients,loss] = modelGradients(dlX,T,parameters,hyperparameters,numTimeSteps)

dlY = model(dlX,parameters,hyperparameters,true);
dlY = softmax(dlY,'DataFormat','CBT');

dlT = dlarray(T,'CBT');
loss = maskedCrossEntropyLoss(dlY, dlT, numTimeSteps);

gradients = dlgradient(mean(loss),parameters);

end