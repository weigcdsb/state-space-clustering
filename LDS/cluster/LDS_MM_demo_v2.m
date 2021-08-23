% addpath(genpath('D:\github\state-space-clustering'));
addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));

%% simulation
rng(2)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
d1 = ones(n,1)*0;
d2 = ones(n,1)*1;
d3 = ones(n,1)*-5;
d = [d1;d2;d3];
C_all1 = reshape(normrnd(0.08,1e-4,n*p,1), [], p);
C_all2 = reshape(normrnd(-0.02,1e-4,n*p,1), [], p);
C_all3 = reshape(normrnd(-0.18,1e-4,n*p,1), [], p);
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

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));
% clusterPlot(Y, Lab)


%% MCMC setting
rng(3)
ng = 100;
kMM = 4;

% pre-allocation
Z_fit = zeros(N, ng);
RHO_fit = zeros(kMM, ng);
d_fit = zeros(N, kMM, ng);
C_fit = zeros(N, p*kMM, ng);
mudc_fit = zeros(p+1, kMM, ng);
Sigdc_fit = zeros(p+1, p+1, kMM, ng);
b_fit = zeros(kMM*p, ng);
A_fit = zeros(kMM*p, kMM*p, ng);
Q_fit = zeros(kMM*p, kMM*p, ng);
x0_fit = zeros(kMM*p, ng);
X_fit = zeros(kMM*p, T, ng);

% priors
delta0 = ones(1, kMM);

Q0 = eye(kMM*p);

mux00 = zeros(kMM*p, 1);
Sigx00 = eye(kMM*p)*25;

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1)*1e6;

Psidc0 = eye(p+1)*1e-4;
nudc0 = p+1+2;

mubA0_mat = [zeros(kMM*p,1) eye(kMM*p)];
SigbA0_f = @(nClus) sparse(eye(p*(1+p*nClus))*0.25);

Psi0 = eye(p)*1e-4;
nu0 = p+2;

% initials
Z_fit(:,1) = ones(1, N);
RHO_fit(:,1) = ones(kMM,1)/kMM;

mudc_fit(:,:,1) = zeros(p+1, kMM);
Sigdc_fit(:,:,:,1) = repmat(eye(p+1)*1e-2,1,1,kMM);
% initial for d_fit: 0
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


% check
% Sigdc_fit = repmat(eye(p+1)*1e-4,1,1,kMM,ng);

%% MCMC
for g = 2:ng
    
    disp(g)
    % (1) update Z_fit
    LAM_tmp = zeros(N, T, kMM);
    LLHD = zeros(N, kMM);
    for k = 1:kMM
        latID = id2id(k,p);
        LAM_tmp(:,:,k) = exp(C_fit(:,latID,g-1)*X_fit(latID,:,g-1) + d_fit(:,k,g-1));
        LLHD(:, k) = sum(log(poisspdf(Y, LAM_tmp(:,:,k))), 2);
    end
    
    logp_tmp = repmat(log(RHO_fit(:,g-1)'), N, 1) + LLHD;
    clus_tmp = mnrnd(ones(N, 1), softmax(logp_tmp')');
    [Z_fit(:,g), ~] = find(clus_tmp');
    
    % (2) update RHO_fit
    nClus = histc(Z_fit(:,g),1:kMM);
    RHO_fit(:,g) = drchrnd(delta0 + nClus', 1);
    
    % (3) update model-related parameters
    [X_fit(:,:,g),x0_fit(:,g),d_fit(:,:,g),C_fit(:,:,g),...
    mudc_fit(:,:,g), Sigdc_fit(:,:,:,g),...
    b_fit(:,g),A_fit(:,:,g),Q_fit(:,:,g)] =...
    blockDiag_gibbsLoop_MM_v2(Y,X_fit(:,:,g-1), Z_fit(:,g), d_fit(:,:,g-1), C_fit(:,:,g-1),... % cluster-invariant
    mudc_fit(:,:,g-1), Sigdc_fit(:,:,:,g-1),...
    x0_fit(:,g-1), b_fit(:,g-1), A_fit(:,:,g-1), Q_fit(:,:,g-1), kMM,... % cluster-related
    Q0, mux00, Sigx00, deltadc0, Taudc0,Psidc0,nudc0,...
    mubA0_mat, SigbA0_f, Psi0,nu0);
    
%     figure(1)
%     clusterPlot(Y, Z_fit(:,g)')
end


for k = 1:ng
    figure(k)
    clusterPlot(Y, Z_fit(:,k)')
end











