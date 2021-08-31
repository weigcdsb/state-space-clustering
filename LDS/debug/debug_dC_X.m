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
C_trans = zeros(n*nClus, p*nClus);
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
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));
ng = 50;
idx = 30:ng;

%% fitting: MCMC, no update prior
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(n*nClus, ng);
C_fit = zeros(n*nClus, p, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = repmat(A,1,1,ng); % true
b_fit = repmat(b,1,ng); % true
Q_fit = repmat(Q,1,1,ng); % true


% priors
Q0 = eye(nClus*p)*1e-2;
mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p)*1e2;
mudc0 = zeros(p+1,1);
Sigdc0 = eye(p+1);

% initials
% initial for d_fit: 0
C_fit(:,:,1) = reshape(normrnd(0,1e-2,n*nClus*p,1), [], p);
C_trans_tmp = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
end
x0_fit(:,1) = lsqr(C_trans_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

for g = 2:ng
    
    disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    C_trans_tmp = zeros(n*nClus, p*nClus);
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
    
    % (3) update d_fit & C_fit
    % Laplace approximation
    for i = 1:N
        l = Lab(i);
        latentId = ((l-1)*p+1):(l*p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - inv(Sigdc0)*(dc - mudc0);
        hessdc = @(dc) -X_tmp'*diag(lamdc(dc))*X_tmp - inv(Sigdc0);
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
            [d_fit(i, g-1) C_fit(i,:,g-1)]',1e-8,1000);
        
        if(sum(isnan(mudc)) ~= 0)
            [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
                mudc0,1e-8,1000);
        end
        
        
        Sigdc = -inv(niSigdc);
        Sigdc = (Sigdc + Sigdc')/2;
        dc = mvnrnd(mudc, Sigdc);
        d_fit(i,g) = dc(1);
        C_fit(i,:,g) = dc(2:end);
    end
end

figure
subplot(1,3,1)
plot(d,mean(d_fit(:,idx), 2),'rx');
title('true vs estimated d')
C_fit_mean = mean(C_fit(:,:,idx), 3);
subplot(1,3,2)
plot(sum(C_trans(:,1:p:end), 2),C_fit_mean(:,1),'rx');
title('true vs estimated 1st column of C')
subplot(1,3,3)
plot(sum(C_trans(:,2:p:end), 2),C_fit_mean(:,2),'rx');
title('true vs estimated 2nd column of C')

%% fitting: MCMC, update priors

X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(N, nClus, ng);
C_fit = zeros(N, nClus*p, ng);
mudc_fit = zeros(p+1, nClus, ng);
Sigdc_fit = zeros(p+1, p+1, nClus, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = repmat(A,1,1,ng); % true
b_fit = repmat(b,1,ng); % true
Q_fit = repmat(Q,1,1,ng); % true

% priors
% place-holder...
Q0 = eye(nClus*p)*1e-2;

mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1);

Psidc0 = eye(p+1);
nudc0 = p+1+2;

% initials
% initial for d_fit: 0
C_raw = reshape(normrnd(0,1e-2,N*p,1), [], p);
d_tmp = zeros(N,1);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    C_fit(Lab == k, ladid_tmp, 1) = C_raw(Lab == k, :);
    d_tmp(Lab == k) = d_fit(Lab ==k, k);
end

Sigdc_fit(:,:,1:nClus,1) = repmat(eye(p+1)*1e-2,1,1,nClus);


x0_fit(:,1) = lsqr(C_fit(:,:,1),(log(mean(Y(:,1:10),2))-d_tmp));
[X_tmp,~,~] = ppasmoo_poissexp_v2(Y,C_fit(:,:,1),d_tmp,...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));
gradHess = @(vecX) gradHessX(vecX, d_tmp, C_fit(:,:,1), x0_fit(:,1), Q0,...
    Q_fit(:,:,1), A_fit(:,:,1), b_fit(:,1), Y);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-10,1000);
X_fit(:,:,1) = reshape(muXvec, [], T);

for g = 2:ng
    
    disp(g)
    
    % (1) update X_fit
    
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
    
    % tic;
    % use Cholesky decomposition to sample efficiently
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muXvec), 1) + R'*muXvec;
    Xsamp = R'\z;
    X_fit(:,:,g) = reshape(Xsamp,[], T);
    % toc;
    
    % (2) update x0_fit
    Sigx0 = inv(inv(Sigx00) + inv(Q0));
    mux0 = Sigx0*(Sigx00\mux00 + Q0\X_fit(:,1,g));
    x0_fit(:,g) = mvnrnd(mux0, Sigx0)';
    % disp(x0_fit(:,g))
    
    % (3) update d_fit & C_fit
    % Laplace approximation
    
    for i = 1:N
        l = Lab(i);
        latentId = id2id(l,p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - Sigdc_fit(:,:,l,g-1)\(dc - mudc_fit(:,l,g-1));
        hessdc = @(dc) -X_tmp'*diag(lamdc(dc))*X_tmp - inv(Sigdc_fit(:,:,l,g-1));
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
            [d_fit(i,l, g-1) C_fit(i,latentId,g-1)]',1e-8,1000);
        if(sum(isnan(mudc)) ~= 0)
            [mudc,~,niSigdc,~] = newton(derdc,hessdc,deltadc0,1e-8,1000);
        end
        
        Sigdc = -inv(niSigdc);
        Sigdc = (Sigdc + Sigdc')/2;
        dc = mvnrnd(mudc, Sigdc);
        
        d_fit(i,l,g) = dc(1);
        C_fit(i,latentId,g) = dc(2:end);
    end
    
    
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
    
end

figure
subplot(1,3,1)
plot(d,sum(mean(d_fit(:,:,idx), 3),2),'rx');
title('true vs estimated d')
C_fit_mean = mean(C_fit(:,:,idx), 3);
subplot(1,3,2)
plot(sum(C_trans(:,1:p:end), 2),sum(C_fit_mean(:,1:p:end), 2),'rx');
title('true vs estimated 1st column of C')
subplot(1,3,3)
plot(sum(C_trans(:,2:p:end), 2),sum(C_fit_mean(:,2:p:end), 2),'rx');
title('true vs estimated 2nd column of C')


