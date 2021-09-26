Xnorm = zeros(size(X_fit, 1), g);

for k = 1:g
   for l = 1:size(X_fit, 1)
       Xnorm(l, k) = norm(X_fit(l,:,k));
   end
end
plot(Xnorm')


xdiff_norm = zeros(g,1);
for k = 2:g
    xdiff_norm(k) = norm(X_fit(:,:,k) - X_fit(:,:,k-1), 'fro');
end

plot(xdiff_norm)



Xdiff_norm = zeros(size(X_fit, 1), g);

for k = 2:g
   for l = 1:size(X_fit, 1)
       Xdiff_norm(l, k) = norm(X_fit(l,:,k) - X_fit(l,:,k-1));
   end
end
plot(Xdiff_norm')


lamX = @(X) exp(C_trans_tmp*X + d_tmp) ;




lamTmp = exp()

lpdf = @(vecX) sum(log(poisspdf(Y, lamX(reshape(vecX, [], T)))), 'all');


llhd_trace = zeros(g-1,1);
for l = 1:g-1
    disp(l)
    d_raw = d_fit(:,:,l);
    I = (1 : size(d_raw, 1)) .';
    k = sub2ind(size(d_raw), I, Lab');
    d_tmp = d_raw(k);
    C_tmp = C_fit(:,:,l);
    lam_tmp = exp(C_tmp*X_fit(:,:,l) + d_tmp);
    llhd_trace(l) = sum(log(poisspdf(Y, lam_tmp)), 'all');
end
    

plot(llhd_trace(1000:end)/ sum(Y, 'all'))


for g = 1:ng
    figure(1)
    subplot(3,2,1)
    plot(X(1:p,:)')
    title('true')
    subplot(3,2,2)
    plot(X_fit(1:p,:,g)')
    title('fit')
    subplot(3,2,3)
    plot(X(p+1:2*p,:)')
    subplot(3,2,4)
    plot(X_fit(p+1:2*p,:,g)')
    subplot(3,2,5)
    plot(X(2*p+1:3*p,:)')
    subplot(3,2,6)
    plot(X_fit(2*p+1:3*p,:,g)')
    
end



sval = zeros(nClus*p, g);

[U,S,V] = svd(C_fit(:,:,k)*X_fit(:,1,k), 0);


for k = 1:g
    disp(k)
    [~,S,~] = svd(C_fit(:,:,k)*X_fit(:,:,k), 0);
    diagTmp = diag(S);
    sval(:,k) = diagTmp(1:nClus*p);
end


plot(sval')





R = [1 4; 0 3];
xt = R*ones(2,1);

KC = randn(4, 2);
C = KC/(sqrtm(KC'*KC));

C*R*ones(2,1)
C*xt

[U,S,V] = svd(C*xt)

KU = randn(2, 2);
U = KU/(sqrtm(KU'*KU));

C*R
C*U'*(U*R*ones(2,1))


