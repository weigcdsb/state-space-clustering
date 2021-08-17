addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(2)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
d = ones(n*nClus,1)*0;
C_all = reshape(normrnd(0.08,1e-3,n*nClus*p,1), [], p);
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

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));

%% MCMC setting
alphaDP = 0.01;
ng = 50;

% pre-allocation
Z_fit = zeros(N, ng);
d_fit = zeros(N, ng);
C_fit = zeros(N, p, ng);
b_fit = zeros(N*p, ng);
A_fit = zeros(N*p, N*p, ng);
Q_fit = zeros(N*p, N*p, ng);
x0_fit = zeros(N*p, ng);
Q0_fit = zeros(N*p, N*p, ng);
X_fit = zeros(N*p, T, ng);


% initials
% start from each full clusters
% Z_fit(:,1) = 1:N;

% test
Z_fit(:,1) = [1 1 1 2 1 2 1 2 1 3 1 4 1 5 5 5 6 6 10 1 1 2 10 10 10 6 6 6 5 4];

% reorder Y by labels
[Zsort_tmp,id] = sort(Z_fit(:,1));
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);

% initial for d_fit: 0
C_fit(:,:,1) = reshape(normrnd(0,1e-2,N*p,1), [], p);
C_trans_tmp = zeros(N, p*nClus_tmp);
for k = 1:nClus_tmp
    idx_tmp = (Zsort_tmp == uniZsort_tmp(k));
    C_trans_tmp(idx_tmp, ((k-1)*p+1):k*p) = C_fit(idx_tmp,:,1);
end
% imagesc(C_trans_tmp)
% initial for b_fit: 0
A_fit(:,:,1) = eye(N*p);
Q_fit(:,:,1) = eye(N*p)*1e-4;

latID = id2id(uniZsort_tmp, p);
x0_fit(latID,1) = lsqr(C_trans_tmp,(log(mean(Y(id,1:10),2))-d_fit(id,1)));
Q0_fit(:, :, 1) = eye(N*p);
X_fit(latID, :, 1) = ppasmoo_poissexp_v2(Y(id,:),C_trans_tmp,d_fit(id,1),...
    x0_fit(latID,1),Q0_fit(latID, latID, 1),...
    A_fit(latID, latID,1),b_fit(latID, 1),Q_fit(latID, latID,1));

% priors
% place-holder
x0 = zeros(N*p, 1);
Q0 = eye(N*p);

mudc0 = zeros(p+1,1);
Sigdc0 = sparse(eye(p+1)*1e-2);

mubA0_all = sparse([zeros(N*p,1) eye(N*p)]);
SigbA0_f = @(nClus) sparse(eye(p*(1+p*nClus))*1e-2);

Psi0 = eye(p)*1e-4;
nu0 = p+2;




%% MCMC
for g = 2:ng
    
    disp(g)
    % (1) update eta: lower
    z_star = max(Z_fit(:,g-1));
    eta_tmp = zeros(1, z_star);
    for m = 1:z_star
        eta_tmp(m) = betarnd(1 + sum(Z_fit(:,g-1) == m),...
            N - sum(Z_fit(:,g-1) <= m)+ alphaDP);
    end
    
    rho_tmp = eta2rho(eta_tmp);
    z_rho_tab = table((1:z_star)', rho_tmp','VariableNames',{'z', 'rho'});
    
    % (2) update u
    z_tab = table(Z_fit(:,g-1),'VariableNames',{'z'});
    u_tmp = rand(N, 1).*...
        join(z_tab, z_rho_tab).rho;
    
    % (3) update eta: upper
    eta_tmp2 = etaExt(eta_tmp, u_tmp, alphaDP);
    s_star = length(eta_tmp2);
    
    % (4) update THETA: model related parameters
    [] = gibbsLoop();
    
    
    
    
    
    
    
end












