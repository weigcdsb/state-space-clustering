addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(1)
n = 40;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
pLab = repelem(1:nClus, p);

d = randn(N,1);
C_trans = zeros(N, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k))+1;
end

X = zeros(p*nClus, T);
x0 = zeros(p*nClus, 1);
Q0 = eye(nClus*p)*1e-2;
X(:,1) = mvnrnd(x0, Q0)';

b1 = ones(p,1)*0;
b2 = ones(p,1)*0;
b3 = ones(p,1)*0;
b = [b1;b2;b3];

Q1 = 1e-3*eye(p);
Q2 = 1e-3*eye(p);
Q3 = 1e-3*eye(p);
Q = blkdiag(Q1, Q2, Q3);

%
A = eye(size(Q,1));
while any(imag(eig(A))==0)
    A= randn(size(Q));
    A = A-diag(diag(A));
    A(squareform(pdist(pLab'))==0)=0;
    A = A./sqrt(sum((A-diag(diag(A))).^2,2))*0.1;
    A = A+eye(size(Q,1))*0.92;
end

% let's generate lambda
logLam = zeros(N, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));
clusterPlot(Y, Lab)


%% fitting: MCMC
rng(3)
ng = 1000;
X_fit = zeros(nClus*p, T, ng);
x0_fit = zeros(nClus*p, ng);
d_fit = repmat(d,1,ng); % true
C_fit = repmat([sum(C_trans(:,1:p:end), 2) sum(C_trans(:,2:p:end), 2)],...
    1,1,ng); % true
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);


% priors
Q0 = eye(nClus*p)*1e-2;
mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);
BA0 = [zeros(nClus*p,1) eye(nClus*p)]';
Lamb0 = eye(nClus*p + 1);
Psi0 = eye(nClus*p)*1e-4;
nu0 = nClus*p+2;

% initials
% initial for b_fit: 0
A_fit(:,:,1) = eye(nClus*p);
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

C_trans_tmp = zeros(N, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
end
x0_fit(:,1) = lsqr(C_trans_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

A_fit_norm = zeros(ng, 1);
A_fit_fro = zeros(ng, 1);
b_fit_norm = zeros(ng, 1);

b_fit_norm(1) = norm(b_fit(:,1));
A_fit_norm(1) = norm(A_fit(:,:,1));
A_fit_fro(1) = norm(A_fit(:,:,1), 'fro');


for g = 2:ng
    
    disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    C_trans_tmp = zeros(N, p*nClus);
    for k = 1:length(Lab)
        C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,g-1);
    end
    d_tmp = d_fit(:,g-1);
    x0_tmp = x0_fit(:,g-1);
    A_tmp = A_fit(:,:,g-1);
    b_tmp = b_fit(:,g-1);
    Q_tmp = Q_fit(:,:,g-1);
    
    X_tmp = X_fit(:,:,g-1);
    gradHess = @(vecX) gradHessX(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0, Q_tmp, A_tmp, b_tmp, Y);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    if(sum(isnan(muXvec)) ~= 0)
        disp('use adaptive smoother initial')
        X_tmp = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
        [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    end
    
    % use Cholesky decomposition to sample efficiently
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muXvec), 1) + R'*muXvec;
    Xsamp = R'\z;
    X_fit(:,:,g) = reshape(Xsamp,[], T);
    
    % (2) update x0_fit
    Sigx0 = inv(inv(Sigx00) + inv(Q0));
    mux0 = Sigx0*(Sigx00\mux00 + Q0\X_fit(:,1,g));
    x0_fit(:,g) = mvnrnd(mux0, Sigx0)';
    
    % (4)update Q
    Y_BA = X_fit(:,2:T,g)';
    X_BA = [ones(T-1,1) X_fit(:,1:(T-1),g)'];
    
    BAn = (X_BA'*X_BA + Lamb0)\(X_BA'*Y_BA + Lamb0*BA0);
    PsiQ = Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
        (BAn - BA0)'*Lamb0*(BAn - BA0);
    nuQ = T-1 + nu0;
    Q_fit(:,:,g) = iwishrnd(PsiQ,nuQ);
    % (5) update b_fit & A_fit
    
    Lambn = X_BA'*X_BA + Lamb0;
    BAvec = mvnrnd(BAn(:), kron(Q_fit(:,:,g), inv(Lambn)))';
    BAsamp = reshape(BAvec,[], nClus*p)';
    b_fit(:,g) = BAsamp(:,1);
    A_fit(:,:,g) = BAsamp(:,2:end);
    
    b_fit_norm(g) = norm(b_fit(:,g));
    A_fit_norm(g) = norm(A_fit(:,:,g));
    A_fit_fro(g) = norm(A_fit(:,:,g), 'fro');
    
    
    figure(1)
    subplot(1,3,1)
    plot(b_fit_norm(1:g))
    subplot(1,3,2)
    plot(A_fit_norm(1:g))
    subplot(1,3,3)
    plot(A_fit_fro(1:g))
end

figure
subplot(1,2,1)
plot(b_fit_norm(1:ng))
title('norm of b')
subplot(1,2,2)
plot(A_fit_fro(1:ng))
title('Frobenius norm of A')
% idx = 800:1000;
% idx = 500:1000;
% idx = 5000:ng;

figure
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

figure
subplot(1,2,1)
imagesc(A)
colorbar()
cLim = caxis;
title('true')
subplot(1,2,2)
imagesc(mean(A_fit(:,:,idx), 3))
colorbar()
set(gca,'CLim',cLim)
title('fit')

mean(Q_fit(:,:,idx), 3)


C_fit_mean = mean(C_fit(:,:,idx), 3);
C_trans_fit = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_fit(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit_mean(k,:);
end
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()
subplot(1,2,2)
imagesc(exp(C_trans_fit*mean(X_fit(:,:,idx), 3) + mean(d_fit(:,idx), 2)))
set(gca,'CLim',cLim)
title('fit')
colorbar()






