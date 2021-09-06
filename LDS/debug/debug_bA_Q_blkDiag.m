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
ng = 2;
idx = 2;

%% fitting: MCMC
rng(3)
X_fit = repmat(X, 1, 1, ng); % true
x0_fit = repmat(x0,1,ng); %true
d_fit = repmat(d,1,ng); % true
C_fit = repmat([sum(C_trans(:,1:p:end), 2) sum(C_trans(:,2:p:end), 2)],...
    1,1,ng); % true

% need to estimate
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

% priors
BA0_all = [zeros(nClus*p,1) eye(nClus*p)]';
Lamb0 = eye(nClus*p + 1);
Psi0 = eye(p)*1e-4;
nu0 = p+2;

% initials
% initial for b_fit: 0
A_fit(:,:,1) = eye(nClus*p);
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

for g = 2:ng
    
    disp(g)
    
    for l = unique(Lab)
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
end

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
