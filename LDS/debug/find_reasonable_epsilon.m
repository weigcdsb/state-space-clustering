function epsilon = find_reasonable_epsilon(theta0, grad0, logp0, f)
epsilon = 0.1;
r0 = randn(1, length(theta0));
% Figure out what direction we should be moving epsilon.
[tmp, rprime, tmp, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
acceptprob = exp(logpprime - logp0 - 0.5 * (rprime * rprime' - r0 * r0'));

% Here we presume that energy function returns NaN, if energy cannot be
% evaluated at the suggested hyperparameters so that we need smalled epsilon
if isnan(acceptprob)
    acceptprob=0;
end
a = 2 * (acceptprob > 0.5) - 1;
% Keep moving epsilon in that direction until acceptprob crosses 0.5.
while (acceptprob^a > 2^(-a))
    epsilon = epsilon * 2^a;
    [tmp, rprime, tmp, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
    acceptprob = exp(logpprime - logp0 - 0.5 * (rprime * rprime' - r0 * r0'));
end
end
