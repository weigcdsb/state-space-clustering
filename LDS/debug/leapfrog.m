function [thetaprime, rprime, gradprime, logpprime] = leapfrog(theta, r, grad, epsilon, f)
rprime = r + 0.5 * epsilon * grad;
thetaprime = theta + epsilon * rprime;
[logpprime, gradprime] = f(thetaprime);
rprime = rprime + 0.5 * epsilon * gradprime;
global nfevals;
nfevals = nfevals + 1;
end