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

% do transformation/ constraint: C'*C = I (not C_trans)
% S1 = {};
% V1 = {};
% for j = 1:nClus
%     latId = id2id(j, p);
%     [~,S1{j},V1{j}] = svd(C_trans(:,latId),0);
% end
% M = sparse(blkdiag(S1{:})*(blkdiag(V1{:}))');

[~, S, V] = svd(C, 0);
Mcore = S*V';
M = kron(eye(nClus), Mcore);

X = M*X;
x0 = M*x0;
Q0 = M*Q0*M';
C = C/Mcore;
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

%% d is cluster-dependent, but C is not. (maybe let C be cluster-dependent later)
% pre-setting...
rng(3)
ng = 1000;

X_fit = zeros(nClus*p, T, ng);
mud_fit = zeros(nClus, ng);
sig2d_fit = zeros(nClus, ng);
d_fit = zeros(N, nClus, ng);
K_fit = zeros(N, p, ng);
C_fit = zeros(N, p, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

% priors...
Q0 = eye(nClus*p)*1e-2;

% priors for initial (x0)
mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);

% prior for K: C = K*(K'*K)^{-1/2}
% just each entry of K is i.i.d. N(0,1)

% prior for d
nud0 = 1;
sig2d0 = 1e-2;
deltad0 = 0;
k0 = 1;

% prior for linear dyanmics (b, A, Q)
BA0_all = [zeros(nClus*p,1) eye(nClus*p)]';
Lamb0 = eye(nClus*p + 1);
Psi0 = eye(p)*1e-4;
nu0 = p+2;

% initials...
% a0 = nud0/2;
% b0 = nud0*sig2d0/2;
% 1./gamrnd(a0, 1/b0, nClus, 1);
sig2d_fit(:,1) = ones(nClus,1)*1e-2;
mud_fit(:,1) = zeros(nClus,1);
d_fit(:,:,1) = zeros(N,nClus);

K_fit(:,:,1) = randn(N, p);
C_fit(:,:,1) = K_fit(:,:,1)/(sqrtm(K_fit(:,:,1)'*K_fit(:,:,1)));
C_tmp = zeros(N, p*nClus);
for k = 1:length(Lab)
    C_tmp(k, id2id(Lab(k), p)) = C_fit(k,:,1);
end

A_fit(:,:,1) = eye(nClus*p);
b_fit(:,1) = zeros(nClus*p, 1);
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

x0_fit(:,1) = lsqr(C_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

%% MCMC
optdK.M=1;
optdK.Madapt=0;
epsilon = 0.01;
burnIn = round(ng/10);
flg = 0;

for g = 2:ng
    
    % disp(g)
    
    % (1) update X_fit
    d_raw = d_fit(:,:,g-1);
    I = (1 : size(d_raw, 1)) .';
    k = sub2ind(size(d_raw), I, Lab');
    d_tmp = d_raw(k);
    C_tmp = zeros(N, p*nClus);
    for k = 1:length(Lab)
        C_tmp(k, id2id(Lab(k), p)) = C_fit(k,:,1);
    end
    
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
    
    % (3) update d_fit & C_fit (K_fit)
    % NUTS (no U turn sampler)
    
    if(g < round(burnIn/10))
        tuneState = 1; % change epsilon
        disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
            norm(X_fit(:,:,g-1), 'fro') + ", changing");
    elseif(flg == 1)
        tuneState = 3; % fix epsilon
        disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
            norm(X_fit(:,:,g-1), 'fro') + ", tuned");
    else
        if(g > burnIn ||...
                norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
                norm(X_fit(:,:,g-1), 'fro') < 1e-1)
            tuneState = 2; % tune epsilon
            flg = 1;
            disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
                norm(X_fit(:,:,g-1), 'fro') + ", tuning");
        else
            tuneState = 1; % change epsilon
            disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
                norm(X_fit(:,:,g-1), 'fro') + ", changing");
        end
    end
    
    lpdf = @(dKvec) dKlpdf(Y, dKvec(1:N), reshape(dKvec((N+1):end), [], p),...
        X_fit(:,:,g-1), Lab, mud_fit(:,g-1), sig2d_fit(:,g-1));
    glpdf = @(dKvec) getGrad(lpdf, dKvec);
    fg=@(dKvec_r) deal(lpdf(dKvec_r'), glpdf(dKvec_r')'); % log density and gradient
    
    dKvec0 = [d_tmp;reshape(K_fit(:,:,g-1),[],1)];
    
    switch tuneState
        case 1
            [dK_NUTS, ~, ~]=hmc_nuts(fg, dKvec0',optdK);
        case 2
            optdK.Madapt=50;
            [dK_NUTS, ~, diagn]=hmc_nuts(fg, dKvec0',optdK);
            epsilon = diagn.opt.epsilonbar;
            optdK.Madapt=0;
        case 3
            optdK.epsilon = epsilon;
            [dK_NUTS, ~, ~]=hmc_nuts(fg, dKvec0',optdK);
    end
    
    % d_fit
    dback_tmp = d_fit(:,:,g);
    I = (1 : N) .';
    k = sub2ind(size(dback_tmp), I, Lab');
    dback_tmp(k) = dK_NUTS(2,1:N)';
    d_fit(:,:,g) = dback_tmp;
    
    % K_fit & C_fit
    K_fit(:,:,g) = reshape(dK_NUTS(2,(N+1):end), [], p);
    C_fit(:,:,g) = K_fit(:,:,g)/(sqrtm(K_fit(:,:,g)'*K_fit(:,:,g)));
    
    % (4) update mud_fit & sig2d_fit
    for l = unique(Lab(:))'
        nj = sum(Lab == l);
        dj = d_fit(Lab == l, l, g);
        dbar = mean(dj);
        kn = k0 + nj;
        
        % sig2d_fit
        alph = (nud0 + nj)/2;
        beta = (nud0*sig2d0 + sum((dj - dbar).^2) +...
            (k0*nj/kn)*((dbar - deltad0)^2))/2;
        sig2d_fit(l, g) = 1/gamrnd(alph, 1/beta);
        
        % mud_fit
        mud_fit(l,g) = normrnd((k0*deltad0 + sum(dj))/kn,...
            sqrt(sig2d_fit(l, g)/kn));
        
        latentId = id2id(l,p);
        % (4)update Q
        Y_BA = X_fit(latentId,2:T,g)';
        X_BA = [ones(T-1,1) X_fit(:,1:(T-1),g)'];
        
        BA0 = BA0_all(:,latentId);
        BAn = (X_BA'*X_BA + Lamb0)\(X_BA'*Y_BA + Lamb0*BA0);
        PsiQ = Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
            (BAn - BA0)'*Lamb0*(BAn - BA0);
        nuQ = T-1 + nu0;
        Q_fit(latentId,latentId,g) = iwishrnd(PsiQ,nuQ);
        
        % (5) update b_fit & A_fit
        Lambn = X_BA'*X_BA + Lamb0;
        BAvec = mvnrnd(BAn(:), kron(Q_fit(latentId,latentId,g), inv(Lambn)))';
        BAsamp = reshape(BAvec,[], p)';
        b_fit(latentId,g) = BAsamp(:,1);
        A_fit(latentId,:,g) = BAsamp(:,2:end);
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

%%

d_norm = zeros(g-1, 1);
C_norm_fro = zeros(g-1, 1);
b_norm = zeros(g-1, 1);
A_norm_fro = zeros(g-1, 1);
X_norm_fro = zeros(g-1, 1);

for k = 1:g-1
    d_norm(k) = norm(d_fit(:,:,k), 'fro');
    C_norm_fro(k) = norm(C_fit(:,:,k), 'fro');
    b_norm(k) = norm(b_fit(:,k));
    A_norm_fro(k) = norm(A_fit(:,:,k), 'fro');
    X_norm_fro(k) = norm(X_fit(:,:,k), 'fro');
end

figure
plot(X_norm_fro)
title("Frobenius norm of X, dim: " + nClus*p + "\times" + T)


figure
subplot(2,2,1)
plot(d_norm)
title('norm of d')
subplot(2,2,2)
plot(C_norm_fro)
title('Frobenius norm of C')
subplot(2,2,3)
plot(b_norm)
title('norm of b')
subplot(2,2,4)
plot(A_norm_fro)
title('Frobenius norm of A')

%%
idx = round(ng/2):ng;

figure
subplot(1,3,1)
plot(d,sum(mean(d_fit(:,:,idx), 3),2),'rx');
ylabel('estimate')
title('d')
C_fit_mean = mean(C_fit(:,:,idx), 3);
subplot(1,3,2)
plot(sum(C_trans(:,1:p:end), 2),C_fit_mean(:,1),'rx');
title('1st column of C')
xlabel('true')
subplot(1,3,3)
plot(sum(C_trans(:,2:p:end), 2),C_fit_mean(:,2),'rx');
title('2nd column of C')
sgtitle('No Update of Prior')



subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()
subplot(1,2,2)
C_tmp = zeros(N, p*nClus);
for k = 1:length(Lab)
    C_tmp(k, id2id(Lab(k), p)) = C_fit_mean(k,:);
end
imagesc(exp(C_tmp*mean(X_fit(:,:,idx), 3) + sum(mean(d_fit(:,:,idx), 3),2)))
set(gca,'CLim',cLim)
title('fit')
colorbar()
