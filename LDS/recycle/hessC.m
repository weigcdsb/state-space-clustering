function hess = hessC(vecC, X_fit, d_fit)

C = reshape(vecC, [], size(X_fit, 1));
lamC = exp(C*X_fit + d_fit);
XC = kron(X_fit(:,:)', eye(size(C,1)));
hess = -XC'*diag(lamC(:))*XC;

end