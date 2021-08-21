addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(2)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
d1 = ones(n,1)*0;
d2 = ones(n,1)*0.1;
d3 = ones(n,1)*0.2;
d = [d1;d2;d3];
C_all1 = reshape(normrnd(0.06,1e-3,n*p,1), [], p);
C_all2 = reshape(normrnd(0.08,1e-3,n*p,1), [], p);
C_all3 = reshape(normrnd(0.1,1e-3,n*p,1), [], p);
C_all = [C_all1; C_all2; C_all3];

C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_all(k,:);
end

X = zeros(p*nClus, T);
x0 = [1.2 1.2 0.5 0.5 1 1]*10;
Q0 = eye(nClus*p)*1e-2;
X(:,1) = mvnrnd(x0, Q0)';

b1 = ones(p,1)*0.01;
b2 = ones(p,1)*0;
b3 = ones(p,1)*-0.03;
b = [b1;b2;b3];

Q1 = 1e-5*eye(p);
Q2 = 1e-4*eye(p);
Q3 = 1e-3*eye(p);
Q = blkdiag(Q1, Q2, Q3);


A = [1 0 0 0 0.4 -0.4;...
    0 1 0 0 -0.3 0.305;
    0 0 1 0 -0.2 0.19;
    0 0 0 1 0.11 -0.1;
    0 0 0 0 1 0;
    0 0 0 0 0 1];

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
% imagesc(logLam)
imagesc(exp(logLam))
colorbar()

Y = poissrnd(exp(logLam));
figure(4)
imagesc(Y)
colorbar()

figure(5)
clusterPlot(Y, Lab)

%%
ng = 100;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(N, ng);
C_fit = zeros(N, p, ng);
mudc_fit = zeros(p+1, ng);
Sigdc_fit = zeros(p+1, p+1, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);


% priors
% place-holder...
Q0 = eye(nClus*p);

mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p)*1e2;

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1)*1e-1;

Psidc0 = eye(p+1)*1e-2;
nudc0 = p+1+2;

mubA0_mat = [zeros(nClus*p,1) eye(nClus*p)];
SigbA0 = eye(p*(1+p*nClus))*0.25;

Psi0 = eye(p)*1e-4;
nu0 = p+2;

% initials
% initial for d_fit: 0
C_fit(:,:,1) = reshape(normrnd(0,1e-2,N*p,1), [], p);
mudc_fit(:,1) = zeros(p+1, 1);
Sigdc_fit(:,:,1) = eye(p+1)*1e-2;

A_fit(:,:,1) = eye(nClus*p);
% initial for b_fit: 0
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

C_trans_tmp = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
end
x0_fit(:,1) = lsqr(C_trans_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

%% MCMC
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
    
    [muX,~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
    hess_tmp = hessX(muX(:), d_tmp, C_trans_tmp, Q0, Q_tmp, A_tmp, Y);
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
    % dTest = zeros(N,1);
    % CTest = zeros(N, p);
    for i = 1:N
        l = Lab(i);
        latentId = ((l-1)*p+1):(l*p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - inv(Sigdc_fit(:,:,g-1))*(dc - mudc_fit(:,g-1));
        hessdc = @(dc) -X_tmp'*diag(lamdc(dc))*X_tmp - inv(Sigdc_fit(:,:,g-1));
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
            [d_fit(i, g-1) C_fit(i,:,g-1)]',1e-8,1000);
        
        % [mudc [d(i) C_all(i,:)]']
        
        Sigdc = -inv(niSigdc);
        Sigdc = (Sigdc + Sigdc')/2;
        dc = mvnrnd(mudc, Sigdc);
        
        % dTest(i) = mudc(1);
        % CTest(i,:) = mudc(2:end);
        d_fit(i,g) = dc(1);
        C_fit(i,:,g) = dc(2:end);
    end
    
    %     [CTest C_all]
    %     [dTest d]
    
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
    
    %     [b_fit(:,g) b]
    %     imagesc(abs(A_fit(:,:,g) - A))
    %     colorbar()
    
    % (5) update Q_fit
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






