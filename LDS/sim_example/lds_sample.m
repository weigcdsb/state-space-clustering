%% simulation
rng(2)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
pLab = repelem(1:nClus, p);

d = randn(n*nClus,1)*0;
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

figure(1)
imagesc(A)
colorbar()
xlabel('sending')
ylabel('receiving')

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

figure(2)
plot(X')

figure(3)
imagesc(exp(logLam))
colorbar()

Y = poissrnd(exp(logLam));
figure(4)
imagesc(Y)
colorbar()

Y = poissrnd(exp(logLam));
clusterPlot(Y, Lab)

%%
rng(3)
ng = 100;

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
Q0 = eye(nClus*p);

mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1)*1e2;

Psidc0 = eye(p+1)*1e-4;
nudc0 = p+1+2;

mubA0_mat = [zeros(nClus*p,1) eye(nClus*p)];
SigbA0 = eye(p*(1+p*nClus))*0.25;

Psi0 = eye(p)*1e-4;
nu0 = p+2;

% initials
% initial for d_fit: 0
C_raw = reshape(normrnd(0,1e-2,N*p,1), [], p);
d_tmp = zeros(N,1);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    C_fit(Lab == k, ladid_tmp, 1) = C_raw(Lab == k, :);
    d_tmp(Lab == k) = d_fit(Lab ==k, k);
end

% mudc_fit(:,:,1) = zeros(p+1, 1);
Sigdc_fit(:,:,1:nClus,1) = repmat(eye(p+1)*1e-2,1,1,nClus);

A_fit(:,:,1) = eye(nClus*p);
% initial for b_fit: 0
Q_fit(:,:,1) = eye(nClus*p)*1e-4;


x0_fit(:,1) = lsqr(C_fit(:,:,1),(log(mean(Y(:,1:10),2))-d_tmp));
[X_tmp,~,~] = ppasmoo_poissexp_v2(Y,C_fit(:,:,1),d_tmp,...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));
gradHess = @(vecX) gradHessX(vecX, d_tmp, C_fit(:,:,1), x0_fit(:,1), Q0,...
    Q_fit(:,:,1), A_fit(:,:,1), b_fit(:,1), Y);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-10,1000);
X_fit(:,:,1) = reshape(muXvec, [], T);



%% MCMC
for g = 2:ng
    
    disp(g)
    
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
%     X_tmp = X_fit(:,:,g-1);
    X_tmp = ppasmoo_poissexp_v2(Y,C_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
%     hess_tmp = hessX(muX(:), d_tmp, C_tmp, Q0, Q_tmp, A_tmp, Y);
    
    % if use Newton?
    gradHess = @(vecX) gradHessX(vecX, d_tmp, C_tmp, x0_tmp, Q0, Q_tmp, A_tmp, b_tmp, Y);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-10,1000);
    muX = reshape(muXvec, [], T);
    
    % tic;
    % use Cholesky decomposition to sample efficiently
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muX(:)), 1) + R'*muX(:);
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
        
        % tic;
        % use warm start
        % invSigdc_star = inv(Sigdc_fit(:,:,l,g-1)) + X_tmp'*diag(lamdc(mudc_fit(:,l,g-1)))*X_tmp;
        % mudc_star = mudc_fit(:,l,g-1) + invSigdc_star\(X_tmp'*(Y(i,:)' - lamdc(mudc_fit(:,l,g-1))));
        % [mudc,~,niSigdc,~] = newton(derdc,hessdc,mudc_star,1e-8,1000);
        % toc;
        
        % [mudc [d(i) C_all(i,:)]']
        
        Sigdc = -inv(niSigdc);
        Sigdc = (Sigdc + Sigdc')/2;
        dc = mvnrnd(mudc, Sigdc);
        
        % dTest(i) = mudc(1);
        % CTest(i,:) = mudc(2:end);
        d_fit(i,l,g) = dc(1);
        C_fit(i,latentId,g) = dc(2:end);
    end
    
    %     C_fit(:,:,g)
    %     d_fit(:,:,g)
    
    % (4) update mudc_fit & Sigdc_fit
%    dcRes_all = [];
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
        
        % assume single covariance
%         dcRes_all = [dcRes_all dc_tmp' - mudc_fit(:,l,g)];
    end
    
    % assume single covariance
%     Psidc = Psidc0 + dcRes_all*dcRes_all';
%     nudc = N + nudc0;
%     Sigdc_fit(:,:,:,g) = repmat(iwishrnd(Psidc,nudc),1,1,nClus);
    
%     mudc_fit(:,:,g)
%     Sigdc_fit(:,:,:,g)
    
    % (5) update b_fit & A_fit
    for l = unique(Lab)
        
        latentId = id2id(l,p);
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
    
    %     [b_fit(:,g) b]
    %     imagesc(abs(A_fit(:,:,g) - A))
    %     colorbar()
    
    % (6) update Q_fit
    for l = unique(Lab)
        latentId = ((l-1)*p+1):(l*p);
        mux = A_fit(latentId,:,g)*X_fit(:,1:(T-1),g) +...
            repmat(b_fit(latentId, g), 1, T-1);
        xq = X_fit(latentId,2:T,g) - mux;
        
        PsiQ = Psi0 + xq*xq';
        nuQ = T-1 + nu0;
        Q_fit(latentId,latentId,g) = iwishrnd(PsiQ,nuQ);
    end
end
%%
idx = 50:ng;

subplot(1,2,1)
plot(mean(X_fit(:,:,idx), 3)')
subplot(1,2,2)
plot(X')

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

mean(d_fit(:,:,idx), 3)
mean(C_fit(:,:,idx), 3)
mean(Q_fit(:,:,idx), 3)
mean(mudc_fit(:,:,idx), 3)
mean(Sigdc_fit(:,:,:,idx), 4)

subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()
subplot(1,2,2)
imagesc(exp(mean(C_fit(:,:,idx), 3)*mean(X_fit(:,:,idx), 3) + sum(mean(d_fit(:,:,idx), 3),2)))
set(gca,'CLim',cLim)
title('fit')
colorbar()

