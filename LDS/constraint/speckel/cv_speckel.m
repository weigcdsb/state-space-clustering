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

% X_jX_j' = I
% X' = X*'R --> inv(R')X = X*
gtmp = -mean(X, 2);
Mdiag = {};
for l = unique(Lab)
    latidTmp = id2id(l,p);
    [QX, RX] = mgson(X(latidTmp,:)');
    Mdiag{l} = inv(RX');
end
M = sparse(blkdiag(Mdiag{:}));
g = M*gtmp;

X = M*X + g;
x0 = M*x0 + g;
d = d - (C_trans/M)*g;
C_trans = C_trans/M;
A = M*A/M;
b = M*b + g - (M*A/M)*g;
Q = M*Q*M';
Q0 = M*Q0*M';


% diagonalize Q within each block
M2 = zeros(nClus*p);
for l = unique(Lab)
    latid = id2id(l,p);
    [M2(latid,latid),~] = eig(Q(latid, latid));
end

X = M2*X;
x0 = M2*x0;
C_trans = C_trans/M2;
A = M2*A/M2;
b = M2*b;
Q = M2*Q*M2';
Q0 = M2*Q0*M2';


%%
rng(1)
nTrain = round(T*3/4);
Y_train = nan*Y;
Y_test = nan*Y;
for k = 1:N
    trIdx = randsample(T, nTrain);
    teIdx = setdiff(1:T, trIdx);
    Y_train(k,trIdx) = Y(k,trIdx);
    Y_test(k, teIdx) = Y(k,teIdx);
end


clusterPlot(Y_train, Lab)
clusterPlot(Y_test, Lab)

%%
nClus = 1;
Lab =ones(1,N);
p = 2;

rng(2)
ng = 1000;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(N, nClus, ng);
C_fit = zeros(N, nClus*p, ng);
mudc_fit = zeros(p+1, nClus, ng);
Sigdc_fit = zeros(p+1, p+1, nClus, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);


% priors
% place-holder...
Q0 = eye(nClus*p)*0.5^2;

mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1);

% Psidc0 = eye(p+1);
Psidc0 = eye(p+1);
nudc0 = p+1+2;

% prior for linear dyanmics (b, A, Q)
BA0 =[0 1]';
Lamb0 = eye(2);
Psi0 = 1e-4;
nu0 = 1+2;

% initials
% initial for d_fit: 0
C_raw = reshape(normrnd(0,1e-2,N*p,1), [], p);
d_tmp = zeros(N,1);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    C_fit(Lab == k, ladid_tmp, 1) = C_raw(Lab == k, :);
    d_tmp(Lab == k) = d_fit(Lab ==k, k, 1);
end

Sigdc_fit(:,:,1:nClus,1) = repmat(eye(p+1)*1e-2,1,1,nClus);

A_fit(:,:,1) = eye(nClus*p);
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

x0_fit(:,1) = lsqr(C_fit(:,:,1),(log(nanmean(Y_train(:,1:10),2))-d_tmp));
[X_tmp,~,~] = ppasmoo_poissexp_na(Y_train,C_fit(:,:,1),d_tmp,...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));
gradHess = @(vecX) gradHessX_na(vecX, d_tmp, C_fit(:,:,1), x0_fit(:,1), Q0,...
    Q_fit(:,:,1), A_fit(:,:,1), b_fit(:,1), Y_train);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-10,1000);
X_fit(:,:,1) = reshape(muXvec, [], T);

gtmp = -mean(X_fit(:,:,1), 2);
Mdiag = {};
for l = unique(Lab)
    latidTmp = id2id(l,p);
    [U,S,V] = svd(X_fit(latidTmp,:,1)' - mean(X_fit(latidTmp,:,1),2)', 'econ');
%     [U,S,V] = svd(X_fit(latidTmp,:,1)', 'econ');
    Mdiag{l} = V*inv(S)*V';
end
M = sparse(blkdiag(Mdiag{:}));
gtrans = M*gtmp;

X_fit(:,:,1) = M*X_fit(:,:,1) + gtrans;
x0_fit(:,1) = M*x0_fit(:,1) + gtrans;
d_tmp = d_tmp - (C_fit(:,:,1)/M)*gtrans;
for l = unique(Lab)
   d_fit(Lab == l,l,1) = d_tmp(Lab == l);
end
C_fit(:,:,1) = C_fit(:,:,1)/M;

%% MCMC
optdc.M=1;
optdc.Madapt=0;
epsilon = 0.01*ones(N,1);
burnIn = round(ng/10);
flg = 0;
nX = numel(X_fit(:,:,1));

xnorm = zeros(ng,1);
xnorm(1) = norm(X_fit(:,:,1), 'fro');
qnorm = zeros(ng,1);
qnorm(1) = norm(Q_fit(:,:,1), 'fro');
for g = 2:ng
    
    % disp(g)
    if(g < burnIn)
        tuneState = 1; % change epsilon
        disp("iter " + g + ", changing");
    elseif(g == burnIn)
        tuneState = 2; % tune epsilon
        disp("iter " + g + ", tuning");
    else
        tuneState = 3; % fix epsilon
        disp("iter " + g + ", tuned");
    end
    
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
    
    gradHess = @(vecX) gradHessX_na(vecX, d_tmp, C_tmp, x0_tmp, Q0,...
        Q_tmp, A_tmp, b_tmp, Y_train);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    if(sum(isnan(muXvec)) ~= 0)
        disp('use adaptive smoother initial')
        X_tmp = ppasmoo_poissexp_na(Y_train,C_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
        [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    end
    
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muXvec), 1) + R'*muXvec;
    x_all = R'\z;
    X_fit(:,:,g) = reshape(x_all,[], T);
    
    % toc;
    
    % (3) update d_fit & C_fit
    
    for i = 1:N
        l = Lab(i);
        obsIdx = find(~isnan(Y_train(i,:)));
        
        latentId = id2id(l,p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        X_tmp = X_tmp(obsIdx,:);
        lamdc = @(dc) exp(X_tmp*dc);
        
        lpdf = @(dc) sum(log(poisspdf(Y_train(i,obsIdx)', lamdc(dc)))) +...
            log(mvnpdf(dc, mudc_fit(:,l,g-1), Sigdc_fit(:,:,l,g-1)));
        glpdf = @(dc) X_tmp'*(Y_train(i,obsIdx)' - lamdc(dc))...
            - Sigdc_fit(:,:,l,g-1)\(dc - mudc_fit(:,l,g-1));
        fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
        dc0 = [d_fit(i,l, g-1) C_fit(i,latentId,g-1)]';
        
        switch tuneState
            case 1
                [dc_NUTS, ~, ~]=hmc_nuts(fg, dc0',optdc);
            case 2
                optdc.Madapt=50;
                [dc_NUTS, ~, diagn]=hmc_nuts(fg, dc0',optdc);
                epsilon(i) = diagn.opt.epsilonbar;
                optdc.Madapt=0;
            case 3
                optdc.epsilon = epsilon(i);
                [dc_NUTS, ~, ~]=hmc_nuts(fg, dc0',optdc);
        end
        
        d_fit(i,l,g) = dc_NUTS(end,1);
        C_fit(i,latentId,g) = dc_NUTS(end,2:end);
    end
    
    gtmp = -mean(X_fit(:,:,g), 2);
    Mdiag = {};
    for l = unique(Lab)
        latidTmp = id2id(l,p);
        [U,S,V] = svd(X_fit(latidTmp,:,g)' - mean(X_fit(latidTmp,:,g),2)', 'econ');
        Mdiag{l} = V*inv(S)*V';
    end
    M = sparse(blkdiag(Mdiag{:}));
    gtrans = M*gtmp;
    
    X_fit(:,:,g) = M*X_fit(:,:,g) + gtrans;
    d_raw = d_fit(:,:,g-1);
    I = (1 : size(d_raw, 1)) .';
    k = sub2ind(size(d_raw), I, Lab');
    d_tmp = d_raw(k);
    d_tmp = d_tmp - (C_fit(:,:,g)/M)*gtrans;
    for l = unique(Lab)
        d_fit(Lab == l,l,g) = d_tmp(Lab == l);
    end
    C_fit(:,:,g) = C_fit(:,:,g)/M;
    
    % (2) update x0_fit
    Sigx0 = inv(inv(Sigx00) + inv(Q0));
    mux0 = Sigx0*(Sigx00\mux00 + Q0\X_fit(:,1,g));
    x0_fit(:,g) = mvnrnd(mux0, Sigx0)';
    
    % (4) update mudc_fit & Sigdc_fit
    for l = unique(Lab)
        dc_tmp = [d_fit(Lab == l, l, g) C_fit(Lab == l, id2id(l,p), g)];
        invTaudc = inv(Taudc0) + sum(Lab == l)*inv(Sigdc_fit(:,:,l,g-1));
        deltadc = invTaudc\(Taudc0\deltadc0 + Sigdc_fit(:,:,l,g-1)\sum(dc_tmp,1)');
        mudc_fit(:,l,g) = mvnrnd(deltadc, inv(invTaudc));
        
        % assume different covariances
        dcRes = dc_tmp' - mudc_fit(:,l,g);
        Psidc = Psidc0 + dcRes*dcRes';
        nudc = sum(Lab == l) + nudc0;
        Sigdc_fit(:,:,l,g) = iwishrnd(Psidc,nudc);
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
    
%     figure(1)
%     subplot(3,2,1)
%     plot(X(1:p,:)')
%     title('true')
%     subplot(3,2,2)
%     plot(X_fit(1:p,:,g)')
%     title('fit')
%     subplot(3,2,3)
%     plot(X(p+1:2*p,:)')
%     subplot(3,2,4)
%     plot(X_fit(p+1:2*p,:,g)')
%     subplot(3,2,5)
%     plot(X(2*p+1:3*p,:)')
%     subplot(3,2,6)
%     plot(X_fit(2*p+1:3*p,:,g)')
    
    
    figure(5)
    subplot(1,2,1)
    imagesc(exp(C_trans*X + d))
    cLim = caxis;
    title('true')
    colorbar()
    subplot(1,2,2)
    C_fit_mean = mean(C_fit(:,:,g), 3);
    imagesc(exp(C_fit_mean*mean(X_fit(:,:,g), 3) + sum(d_fit(:,:,g), 2)))
    set(gca,'CLim',cLim)
    title('fit')
    colorbar()
end

%%
idx = 500:1000;
CX_fit_sum = zeros(N, T);
d_fit_sum = zeros(N,1);
for k = idx
    CX_fit_sum = CX_fit_sum + C_fit(:,:,k)*X_fit(:,:,k);
    d_fit_sum = d_fit_sum + sum(d_fit(:,:,k),2);
end
CX_fit_mean = CX_fit_sum/length(idx);
d_fit_mean = d_fit_sum/length(idx);
lamAll = exp(CX_fit_mean + d_fit_mean);
nansum(log(poisspdf(Y_test,lamAll)), 'all')/nansum(Y_test, 'all')

imagesc(lamAll)


















