function der = derC(vecC,Y, X_fit, d_fit, T)

C = reshape(vecC, [], size(X_fit, 1));

lamC = exp(C*X_fit + d_fit);
Xt = @(t) kron(X_fit(:,t)', eye(size(C,1)));

der = vecC*0;
for t = 1:T
    der = der + Xt(t)'*(Y(:,t) - lamC(:,t));
end


end