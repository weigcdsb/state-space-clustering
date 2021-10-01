%%
rng(1)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
d = randn(n*nClus,1)/5;
C_trans = zeros(N, p*nClus);
C = zeros(N, p);
for k = 1:length(Lab)
    C(k,:) = sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k)) + [-1 .5];
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C(k,:);
end

X = zeros(p*nClus, T);
x0 = zeros(nClus*p,1);
Q0 = eye(nClus*p)*1e-2;
X(:,1) = mvnrnd(x0, Q0)';

b1 = ones(p,1)*0.01;
b2 = ones(p,1)*0;
b3 = ones(p,1)*-0.03;
b = [b1;b2;b3];
b = b*0;

Q1 = 1e-3*eye(p);
Q2 = 1e-3*eye(p);
Q3 = 1e-3*eye(p);
Q = blkdiag(Q1, Q2, Q3);

% Generate X offline (A unspecified)
for i=1:size(Q,1)
    k = ceil(rand()*20)+15;
    X(i,:) = interp1(linspace(0,1,k),...
        randn(k,1),linspace(0,1,T),'spline');
end
plot(X')


% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    logLam(:, t) = d + C_trans*X(:,t);
end

figure(1)
Y = poissrnd(exp(logLam));
clusterPlot(Y, Lab)

[~, S, V] = svd(C, 0);
Mcore = S*V';
M = kron(eye(nClus), Mcore);

X = M*X;
x0 = M*x0;
Q0 = M*Q0*M';
C = C/Mcore;
C_trans = C_trans/M;
% b = M*b;
% A = M*A/M;
Q = M*Q*M';

% center X around 0
g = mean(X, 2);
X = X - g;
x0 = x0 - g;
d = d - C_trans* g;
% b = b + g - A*g;

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
plot(X')

figure(5)
subplot(1,3,1)
plot(X(1:p,:)')
subplot(1,3,2)
plot(X(p+1:2*p,:)')
subplot(1,3,3)
plot(X(2*p+1:3*p,:)')
%%

% nClus = 1;
% Lab = ones(1,N);

rng(3)
ng = 10000;

X_fit = zeros(nClus*p, T, ng);
x0_fit = zeros(nClus*p, ng);

d_fit = zeros(N, nClus, ng);
mud_fit = zeros(nClus, ng);
sig2d_fit = zeros(nClus, ng);

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

% prior for d
nud0 = 1;
sig2d0 = 1e-2;
deltad0 = 0;
kd0 = 1;

% prior for linear dyanmics (b, A, Q)
BA0 =[0 1]';
Lamb0 = eye(2);
Psi0 = 1e-4;
nu0 = 1+2;

% initials...
sig2d_fit(:,1) = ones(nClus,1)*1e-2;
mud_fit(:,1) = zeros(nClus,1);
d_fit(:,:,1) = zeros(N,nClus);

muK_fit(:,1) = zeros(nClus, 1);
sig2K_fit(:,1) = ones(nClus, 1);

K_raw = randn(N, p);
d_tmp = zeros(N,1);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    K_tmp = K_raw(Lab == k, :);
    K_fit(Lab == k, ladid_tmp, 1) = K_tmp;
    C_fit(Lab == k, ladid_tmp, 1) = K_tmp/(sqrtm(K_tmp'*K_tmp));
    d_tmp(Lab == k) = d_fit(Lab ==k, k);
end

A_fit(:,:,1) = eye(nClus*p);
b_fit(:,1) = zeros(nClus*p, 1);
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

x0_fit(:,1) = lsqr(C_fit(:,:,1),(log(mean(Y(:,1:10),2))-d_tmp));
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_fit(:,:,1),d_tmp,...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));
X_fit(:,:,1) = X_fit(:,:,1) - mean(X_fit(:,:,1), 2);


%% MCMC
optdK.M=1;
optdK.Madapt=0;
epsilon = 0.01*ones(nClus,1);
burnIn = round(ng/10);
flg = 0;
nX = p*T;

xnorm = zeros(ng,1);
xnorm(1) = norm(X_fit(:,:,1), 'fro');
for g = 58:ng
    
    % disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    
    d_raw = d_fit(:,:,g-1);
    I = (1 : size(d_raw, 1)) .';
    k = sub2ind(size(d_raw), I, Lab');
    d_tmp = d_raw(k);
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
    % hard centering...
    X_fit(:,:,g) = X_fit(:,:,g) - mean(X_fit(:,:,g), 2);
    
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
        nj = sum(Lab == l);
        
        lpdf = @(dKvec) dKlpdf_blk(Y(Lab == l,:), dKvec(1:nj),...
            reshape(dKvec((nj+1):end), [], p), X_fit(latentId,:,g),...
            mud_fit(l,g-1), sig2d_fit(l,g-1), muK_fit(l,g-1), sig2K_fit(l,g-1));
        glpdf = @(dKvec) getGrad(lpdf, dKvec);
        fg=@(dKvec_r) deal(lpdf(dKvec_r'), glpdf(dKvec_r')'); % log density and gradient
        
        dKvec0 = [d_fit(Lab == l,l, g-1); reshape(K_fit(Lab == l,latentId,g-1), [], 1)];
        
        switch tuneState
            case 1
                [dK_NUTS, ~, ~]=hmc_nuts(fg, dKvec0',optdK);
            case 2
                optdK.Madapt=50;
                [dK_NUTS, ~, diagn]=hmc_nuts(fg, dKvec0',optdK);
                epsilon(l) = diagn.opt.epsilonbar;
                optdK.Madapt=0;
            case 3
                optdK.epsilon = epsilon(l);
                [dK_NUTS, ~, ~]=hmc_nuts(fg, dKvec0',optdK);
        end
        
        % d_fit
        d_fit(Lab == l,l,g) = dK_NUTS(end,1:nj)';
        Kfit_tmp = reshape(dK_NUTS(end,(nj+1):end), [], p);
        K_fit(Lab == l,latentId,g) = Kfit_tmp;
        C_fit(Lab == l,latentId,g) = Kfit_tmp/(sqrtm(Kfit_tmp'*Kfit_tmp));
        
        % (4) update mud_fit & sig2d_fit
        dj = d_fit(Lab == l, l, g);
        dbar = mean(dj);
        kdn = kd0 + nj;
        
        % sig2d_fit
        alph = (nud0 + nj)/2;
        beta = (nud0*sig2d0 + sum((dj - dbar).^2) +...
            (kd0*nj/kdn)*((dbar - deltad0)^2))/2;
        sig2d_fit(l, g) = 1/gamrnd(alph, 1/beta);
        
        % mud_fit
        mud_fit(l,g) = normrnd((kd0*deltad0 + sum(dj))/kdn,...
            sqrt(sig2d_fit(l, g)/kdn));
        
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
    
%     xnorm(g) = norm(X_fit(:,:,g), 'fro');subplot(1,2,1)
    
    figure(2)
    subplot(1,2,1)
    imagesc(exp(C_trans*X + d))
    cLim = caxis;
    title('true')
    colorbar()
    subplot(1,2,2)
    C_fit_mean = mean(C_fit(:,:,g), 3);
    imagesc(exp(C_fit_mean*mean(X_fit(:,:,g), 3)))
    set(gca,'CLim',cLim)
    title('fit')
    colorbar()
end