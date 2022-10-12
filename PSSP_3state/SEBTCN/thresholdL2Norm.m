%Gradient Clipping Function
function g = thresholdL2Norm(g,gradientThreshold)

gradientNorm = sqrt(sum(g.^2,'all'));
if gradientNorm > gradientThreshold
    g = g * (gradientThreshold / gradientNorm);
end

end