
%%
d_norm_fro = zeros(g,1);
C_norm_fro = zeros(g, 1);
b_norm = zeros(g, 1);
A_norm_fro = zeros(g, 1);
X_norm_fro = zeros(g, 1);
K_norm_fro = zeros(g, 1);
Q_norm_fro = zeros(g, 1);

for k = 1:g
    d_norm_fro(k) = norm(d_fit(:,:,k), 'fro');
    C_norm_fro(k) = norm(C_fit(:,:,k), 'fro');
    b_norm(k) = norm(b_fit(:,k));
    A_norm_fro(k) = norm(A_fit(:,:,k), 'fro');
    X_norm_fro(k) = norm(X_fit(:,:,k), 'fro');
    K_norm_fro(k) = norm(K_fit(:,:,k), 'fro');
    Q_norm_fro(k) = norm(Q_fit(:,:,k), 'fro');
end

plot(K_norm_fro)
plot(muK_fit')
plot(sig2K_fit')


figure(1)
subplot(2,1,1)
plot(X_norm_fro)
title("Frobenius norm of X, dim: " + nClus*p + "\times" + T)
subplot(2,1,2)
plot(squeeze(X_fit(:,round(T/2),1:g))')
title("values of x_{500}, dim: " + size(X_fit, 1))

figure(2)
subplot(4,1,1)
plot(d_norm_fro)
title('norm of d')
subplot(4,1,2)
plot(b_norm)
title('norm of b')
subplot(4,1,3)
plot(A_norm_fro)
title('Frobenius norm of A')
subplot(4,1,4)
plot(Q_norm_fro)
title('Frobenius norm of Q')


figure(3)
subplot(2,1,1)
plot(X_norm_fro(1:100))
title("Frobenius norm of X, dim: " + nClus*p + "\times" + T)
subplot(2,1,2)
plot(squeeze(X_fit(:,round(T/2),1:100))')
title("values of x_{500}, dim: " + size(X_fit, 1))

figure(4)
subplot(4,1,1)
plot(d_norm_fro(1:100))
title('norm of d')
subplot(4,1,2)
plot(b_norm(1:100))
title('norm of b')
subplot(4,1,3)
plot(A_norm_fro(1:100))
title('Frobenius norm of A')
subplot(4,1,4)
plot(Q_norm_fro(1:100))
title('Frobenius norm of Q')

idx = round(ng/2): ng;

figure(5)
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()
subplot(1,2,2)
C_fit_mean = mean(C_fit(:,:,idx), 3);
imagesc(exp(C_fit_mean*mean(X_fit(:,:,idx), 3)))
set(gca,'CLim',cLim)
title('fit')
colorbar()


figure(6)
subplot(3,2,1)
plot(X(1:p,:)')
title('true')
subplot(3,2,2)
plot(mean(X_fit(1:p,:,idx), 3)')
title('fit')
subplot(3,2,3)
plot(X(p+1:2*p,:)')
subplot(3,2,4)
plot(mean(X_fit(p+1:2*p,:,idx), 3)')
subplot(3,2,5)
plot(X(2*p+1:3*p,:)')
subplot(3,2,6)
plot(mean(X_fit(2*p+1:3*p,:,idx), 3)')




