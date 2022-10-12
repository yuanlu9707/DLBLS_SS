%Weights Initialization Function
function parameter = initializeGaussian(sz)

parameter = randn(sz,'single') .* 0.01;

end
