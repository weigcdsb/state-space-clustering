addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));
%%
rng(1)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
pLab = repelem(1:nClus, p);

d = randn(n*nClus,1)*0;
C_trans = zeros(N, p*nClus);
C = zeros(N, p);
for k = 1:length(Lab)
    C(k,:) = 2*repmat(sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k)),1,2)+[-2 1];
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C(k,:);
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
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));
clusterPlot(Y, Lab)

% do transformation/ constraint: C_j'*C_j = I
S1 = {};
V1 = {};
for j = 1:nClus
    latId = id2id(j, p);
    [~,S1{j},V1{j}] = svd(C_trans(:,latId),0);
end
M = sparse(blkdiag(S1{:})*(blkdiag(V1{:}))');

X = M*X;
x0 = M*x0;
Q0 = M*Q0*M';
C_trans = C_trans/M;
b = M*b;
A = M*A/M;
Q = M*Q*M';

figure(1)
imagesc(A)
colorbar()
xlabel('sending')
ylabel('receiving')

figure(2)
figure(2)
subplot(1,2,1)
imagesc(exp(logLam))
colorbar()
subplot(1,2,2)
imagesc(exp(d + C_trans*X))
colorbar()

figure(3)
clusterPlot(Y, Lab)

figure(4)
subplot(1,3,1)
plot(X(1:p,:)')
subplot(1,3,2)
plot(X(p+1:2*p,:)')
subplot(1,3,3)
plot(X(2*p+1:3*p,:)')

%% d & C are cluster-dependent & Q: blk-diag
% for computation (update blk-by-blk) & clustering
rng(3)
ng = 10000;

X_fit = zeros(nClus*p, T, ng);
x0_fit = zeros(nClus*p, ng);

% do I need to update mean of K?
K_fit = zeros(N, nClus*p, ng);
muK_fit = zeros(nClus, ng);
sig2K_fit = ones(nClus, ng);
C_fit = zeros(N, nClus*p, ng);

A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

% priors...
Q0 = eye(nClus*p)*1e-2;

% priors for initial (x0)
mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);

% prior for K: C = K*(K'*K)^{-1/2}
% vec(K) ~ N(mu_k*1, sig2K*I)
% i.e. k_{ij} ~ (i.i.d.) N(mu_k, sig2K)
nuK0 = 1;
sig2K0 = 1;
deltaK0 = 0;
kK0 = 1;

% prior for linear dyanmics (b, A, Q)
BA0 =[0 1]';
Lamb0 = eye(2);
Psi0 = 1e-4;
nu0 = 1+2;

% initials...
% sig2d_fit(:,1) = ones(nClus,1)*1e-2;
% mud_fit(:,1) = zeros(nClus,1);
% d_fit(:,:,1) = zeros(N,nClus);

muK_fit(:,1) = zeros(nClus, 1);
sig2K_fit(:,1) = ones(nClus, 1);

K_raw = randn(N, p);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    K_tmp = K_raw(Lab == k, :);
    K_fit(Lab == k, ladid_tmp, 1) = K_tmp;
    C_fit(Lab == k, ladid_tmp, 1) = K_tmp/(sqrtm(K_tmp'*K_tmp));
end

A_fit(:,:,1) = eye(nClus*p);
b_fit(:,1) = zeros(nClus*p, 1);
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

d_tmp = zeros(N,1);
x0_fit(:,1) = lsqr(C_fit(:,:,1),(log(mean(Y(:,1:10),2))-d_tmp));
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_fit(:,:,1),d_tmp,...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

%% MCMC
optK.M=1;
optK.Madapt=0;
epsilon = 0.01*ones(nClus,1);
burnIn = round(ng/10);
flg = 0;
nX = p*T;

for g = 2:ng
    
    % disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    
    d_tmp = zeros(N,1);
    C_tmp = C_fit(:,:,g-1);
    
    x0_tmp = x0_fit(:,g-1);
    A_tmp = A_fit(:,:,g-1);
    b_tmp = b_fit(:,g-1);
    Q_tmp = Q_fit(:,:,g-1);
    X_tmp = X_fit(:,:,g-1);
    
    gradHess = @(vecX) gradHessX(vecX, d_tmp, C_tmp, x0_tmp, Q0, Q_tmp, A_tmp, b_tmp, Y);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    if(sum(isnan(muXvec)) ~= 0)
        disp('use adaptive smoother initial')
        X_tmp = ppasmoo_poissexp_v2(Y,C_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
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
    
    xnorm_change = norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro');
    if(g < round(burnIn/10))
        tuneState = 1; % change epsilon
        disp("iter " + g + ": " + xnorm_change + ", changing");
    elseif(flg == 1)
        tuneState = 3; % fix epsilon
        disp("iter " + g + ": " + xnorm_change + ", tuned");
    else
        if(g > burnIn || xnorm_change < sqrt(1e-2*nX))
            tuneState = 2; % tune epsilon
            flg = 1;
            disp("iter " + g + ": " + xnorm_change + ", tuning");
        else
            tuneState = 1; % change epsilon
            disp("iter " + g + ": " + xnorm_change + ", changing");
        end
    end
    
    % (3) update  C_fit (K_fit): cluster-wise
    % NUTS (no U turn sampler)
    for l = unique(Lab(:))'
        latentId = id2id(l,p);
        
        lpdf = @(Kvec) dKlpdf_blk_nod(Y(Lab == l,:),...
            reshape(Kvec(:), [], p), X_fit(latentId,:,g),...
            muK_fit(l,g-1), sig2K_fit(l,g-1));
        glpdf = @(Kvec) getGrad(lpdf, Kvec);
        fg=@(Kvec_r) deal(lpdf(Kvec_r'), glpdf(Kvec_r')'); % log density and gradient
        
        Kvec0 = reshape(K_fit(Lab == l,latentId,g-1), [], 1);
        
        switch tuneState
            case 1
                [K_NUTS, ~, ~]=hmc_nuts(fg, Kvec0',optK);
            case 2
                optK.Madapt=50;
                [K_NUTS, ~, diagn]=hmc_nuts(fg, Kvec0',optK);
                epsilon(l) = diagn.opt.epsilonbar;
                optK.Madapt=0;
            case 3
                optK.epsilon = epsilon(l);
                [K_NUTS, ~, ~]=hmc_nuts(fg, Kvec0',optK);
        end
        
        %K_fit & C_fit
        Kfit_tmp = reshape(K_NUTS(end,:), [], p);
        K_fit(Lab == l,latentId,g) = Kfit_tmp;
        C_fit(Lab == l,latentId,g) = Kfit_tmp/(sqrtm(Kfit_tmp'*Kfit_tmp));
        
        % (5) update muK_fit & sig2K_fit
        k_tmp = reshape(K_fit(Lab == l,latentId,g), [], 1);
        nk = length(k_tmp);
        kbar = mean(k_tmp);
        kKn = kK0 + nk;
        
        % sig2K_fit
        alph = (nuK0 + nk)/2;
        beta = (nuK0*sig2K0 + sum((k_tmp - kbar).^2) +...
            (kK0*nk/kKn)*((kbar - deltaK0)^2))/2;
        sig2K_fit(l, g) = 1/gamrnd(alph, 1/beta);
        
        % muK_fit
        muK_fit(l,g) = normrnd((kK0*deltaK0 + sum(k_tmp))/kKn,...
            sqrt(sig2K_fit(l, g)/kKn));
    end
    
    
    for k = 1:size(X_fit, 1)
        
       % (5)update Q
        Y_BA = X_fit(k,2:T,g)';
        X_BA = [ones(T-1,1) X_fit(k,1:(T-1),g)'];
        
        BAn = (X_BA'*X_BA + Lamb0)\(X_BA'*Y_BA + Lamb0*BA0);
        PsiQ = Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
            (BAn - BA0)'*Lamb0*(BAn - BA0);
        nuQ = T-1 + nu0;
        Q_fit(k,k,g) = iwishrnd(PsiQ,nuQ);
        
        % (6) update b_fit & A_fit
        Lambn = X_BA'*X_BA + Lamb0;
        BAsamp = mvnrnd(BAn(:), kron(Q_fit(k,k,g), inv(Lambn)))';
        b_fit(k,g) = BAsamp(1);
        A_fit(k,k,g) = BAsamp(2);
    end
    
    
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

save('C:\Users\gaw19004\Desktop\LDS_backup\new2\lds_NUTS_CCI_blk_AQ_diag_d_off.mat')
