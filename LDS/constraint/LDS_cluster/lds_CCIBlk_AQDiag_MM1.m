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
clusterPlot(Y, Lab)
%%
rng(3)
ng = 50;
kMM = 3;

% pre-allocation
Z_fit = zeros(N, ng);
RHO_fit = zeros(kMM, ng);

X_fit = zeros(kMM*p, T, ng);
x0_fit = zeros(kMM*p, ng);

d_fit = zeros(N, kMM, ng);
mud_fit = zeros(kMM, ng);
sig2d_fit = zeros(kMM, ng);

% do I need to update mean of K?
K_fit = zeros(N, kMM*p, ng);
muK_fit = zeros(kMM, ng);
sig2K_fit = ones(kMM, ng);
C_fit = zeros(N, kMM*p, ng);

A_fit = zeros(kMM*p, kMM*p, ng);
b_fit = zeros(kMM*p, ng);
Q_fit = zeros(kMM*p, kMM*p, ng);

% priors...
delta0 = ones(1, kMM);
Q0 = eye(kMM*p)*1e-2;

% priors for initial (x0)
mux00 = zeros(kMM*p, 1);
Sigx00 = eye(kMM*p);

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

% initials
Z_fit(:,1) = ones(1, N);
% Z_fit(:,1) = randsample(kMM, N, true);
RHO_fit(:,1) = ones(kMM,1)/kMM;

d_fit(:,:,1) = zeros(N,kMM);
mud_fit(:,1) = zeros(kMM, 1);
sig2d_fit(:, 1) = ones(kMM, 1);

muK_fit(:,1) = zeros(kMM, 1);
sig2K_fit(:,1) = ones(kMM, 1);

% reorder Y by labels
[Zsort_tmp,id] = sort(Z_fit(:,1));
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);

Csort_trans_tmp = zeros(N, p*nClus_tmp);
Zsort_trans_tmp = zeros(N, p*nClus_tmp);



K_raw = randn(N, p);
d_tmp = zeros(N,1);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    K_tmp = K_raw(Lab == k, :);
    K_fit(Lab == k, ladid_tmp, 1) = K_tmp;
    C_fit(Lab == k, ladid_tmp, 1) = K_tmp/(sqrtm(K_tmp'*K_tmp));
    d_tmp(Lab == k) = d_fit(Lab ==k, k);
end





C_fit(:,:,1) = reshape(normrnd(0,1e-2,N*p*kMM,1), N, []);

% initial for b_fit: 0
A_fit(:,:,1) = eye(kMM*p);
Q_fit(:,:,1) = eye(kMM*p)*1e-4;

% reorder Y by labels
[Zsort_tmp,id] = sort(Z_fit(:,1));
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);

Csort_trans_tmp = zeros(N, p*nClus_tmp);
for k = 1:nClus_tmp
    latid_old = id2id(uniZsort_tmp(k), p);
    latid_new = id2id(k,p);
    idx_old = (Z_fit(:, 1) == uniZsort_tmp(k));
    idx_sort = (Zsort_tmp == uniZsort_tmp(k));
    Csort_trans_tmp(idx_sort, latid_new) = C_fit(idx_old,latid_old,1);
end


d_raw = d_fit(:,:,1);
I = (1 : size(d_raw, 1)) .';
k = sub2ind(size(d_raw), I, Z_fit(:,1));
d_tmp = d_raw(k);

latID = id2id(uniZsort_tmp, p);
x0_fit(latID,1) =...
    lsqr(Csort_trans_tmp,(log(mean(Y(id,1:10),2))-d_tmp(id)));
X_fit(latID, :, 1) = ppasmoo_poissexp_v2(Y(id,:),Csort_trans_tmp,d_tmp(id,1),...
    x0_fit(latID,1),Q0(latID, latID),...
    A_fit(latID, latID,1),b_fit(latID, 1),Q_fit(latID, latID,1));

% no labels: generate by priors
outLab = setdiff(1:kMM, uniZsort_tmp);
if(~isempty(outLab))
    outLatID = id2id(outLab , p);
    x0_fit(outLatID,1) = mvnrnd(mux00(outLatID), Sigx00(outLatID,outLatID))';
    X_fit(outLatID, 1,1) = mvnrnd(x0_fit(outLatID,1), Q0(outLatID,outLatID))';
    for t= 2:T
        X_fit(outLatID, t,1) = mvnrnd(A_fit(outLatID,:,1)*X_fit(:,t-1,1) +...
            b_fit(outLatID,1), Q_fit(outLatID,outLatID,1));
    end
end











