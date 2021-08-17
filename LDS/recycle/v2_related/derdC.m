function der = derdC(vecdC,Y, X_fit)

T = size(X_fit, 2);
dC = reshape(vecdC, [], 1+size(X_fit, 1));
lamdC = exp(dC(:,2:end)*X_fit + dC(:,1));
XdC = kron([ones(1,T); X_fit]', eye(size(Y,1)));
res = Y - lamdC;
der = XdC'*res(:);

end




