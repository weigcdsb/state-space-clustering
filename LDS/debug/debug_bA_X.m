addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));


%% simulation
rng(1)
n = 10;
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
ng = 10000;


%% fitting: MCMC
rng(3)
X_fit = zeros(nClus*p, T, ng);
x0_fit = zeros(nClus*p, ng);
d_fit = repmat(d,1,ng); % true
C_fit = repmat([sum(C_trans(:,1:p:end), 2) sum(C_trans(:,2:p:end), 2)],...
    1,1,ng); % true
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = repmat(Q,1,1,ng); % true

% priors
Q0 = eye(nClus*p)*1e-2;
mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);
mubA0_mat = [zeros(nClus*p,1) eye(nClus*p)];
SigbA0 = eye(p*(1+p*nClus))*0.25;

% initials
% initial for b_fit: 0
A_fit(:,:,1) = eye(nClus*p);

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
    
    % (4) update b_fit & A_fit
    for l = unique(Lab)
        
        latentId = ((l-1)*p+1):(l*p);
        mubA0 = mubA0_mat(latentId, :);
        mubA0 = mubA0(:);
        Z_tmp = X_fit(latentId,2:T,g);
        Z_tmp2 = Z_tmp(:);
        
        X_tmp = kron([ones(1,T-1); X_fit(:, 1:(T-1), g)]', eye(p));
        SigbA_tmp = inv(inv(SigbA0) + X_tmp'*kron(eye(T-1), inv(Q_fit(latentId,latentId,g-1)))*X_tmp);
        SigbA_tmp = (SigbA_tmp + SigbA_tmp')/2;
        mubA_tmp = SigbA_tmp*(inv(SigbA0)*mubA0 +...
            X_tmp'*kron(eye(T-1), inv(Q_fit(latentId,latentId,g-1)))*Z_tmp2);
        bAtmp = reshape(mvnrnd(mubA_tmp, SigbA_tmp)', [], 1+nClus*p);
        b_fit(latentId,g) = bAtmp(:,1);
        A_fit(latentId,:,g) = bAtmp(:,2:end);
    end
    
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

% idx = 200:1000;
figure
subplot(1,2,1)
plot(b_fit_norm(1:ng))
title('norm of b')
subplot(1,2,2)
plot(A_fit_fro(1:ng))
title('Frobenius norm of A')

idx = 5000:ng;

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



