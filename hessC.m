function hess = hessC(vecC, X_fit, d_fit, T)

C = reshape(vecC, [], size(X_fit, 1));

lamC = exp(C*X_fit + d_fit);
Xt = @(t) kron(X_fit(:,t)', eye(size(C,1)));
hess = zeros(length(vecC));

for t = 1:T
    hess = hess - Xt(t)'*diag(lamC(:,t))*Xt(t);
end




end