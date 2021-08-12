function hess = hessdC(vecdC, X_fit)
T = size(X_fit, 2);
dC = reshape(vecdC, [], 1+size(X_fit, 1));
lamdC = exp(dC(:,2:end)*X_fit + dC(:,1));
XdC = kron([ones(1,T); X_fit]', eye(size(dC,1)));

hess = -XdC'*diag(lamdC(:))*XdC;

end