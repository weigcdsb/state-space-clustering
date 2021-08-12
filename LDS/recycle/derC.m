function der = derC(vecC,Y, X_fit, d_fit)

C = reshape(vecC, [], size(X_fit, 1));
lamC = exp(C*X_fit + d_fit);
XC = kron(X_fit', eye(size(C,1)));
res = Y - lamC;
der = XC'*res(:);


end