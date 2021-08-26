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
figure(1)
clusterPlot(Y, Lab)

%% MCMC setting
rng(3)
alphaDP = 1;
ng = 50;

% pre-allocation
Z_fit = zeros(N, ng);

% priors
Q0_f = @(nClus) eye(nClus*p)*1e-2;

mux00_f = @(nClus) zeros(nClus*p, 1);
Sigx00_f = @(nClus) eye(nClus*p);

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1);

Psidc0 = eye(p+1)*1e-4;
nudc0 = p+1+2;

mubA0_all_f = @(nClus) sparse([zeros(nClus*p,1) eye(nClus*p)]);
SigbA0_f = @(nClus) sparse(eye(p*(1+p*nClus))*0.25);

Psi0 = eye(p)*1e-4;
nu0 = p+2;

% initials
% single cluster
Z_fit(:,1) = ones(N, 1);
% N cluster
Z_fit(:,1) = 1:N;


% reorder Y by labels
clusMax = max(Z_fit(:,1));
[Zsort_tmp,id] = sort(Z_fit(:,1));
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);

mudc_fit{1} = zeros(p+1, clusMax);
Sigdc_fit{1} = repmat(eye(p+1)*1e-2,1,1,clusMax);
d_fit{1} = zeros(N, clusMax);
C_fit{1} = reshape(normrnd(0,1e-2,N*p*clusMax,1), N, []);

b_fit{1} = zeros(clusMax*p,1);
A_fit{1} = eye(clusMax*p);
Q_fit{1} = eye(clusMax*p)*1e-4;
x0_fit{1} = zeros(clusMax*p, 1);
X_fit{1} = zeros(clusMax*p, T);
latID = id2id(uniZsort_tmp, p);

Csort_trans_tmp = zeros(N, p*nClus_tmp);
for k = 1:nClus_tmp
    latid_old = id2id(uniZsort_tmp(k), p);
    latid_new = id2id(k,p);
    idx_old = (Z_fit(:, 1) == uniZsort_tmp(k));
    idx_sort = (Zsort_tmp == uniZsort_tmp(k));
    Csort_trans_tmp(idx_sort, latid_new) = C_fit{1}(idx_old,latid_old);
end

d_raw = d_fit{1};
I = (1 : size(d_raw, 1)) .';
k = sub2ind(size(d_raw), I, Z_fit(:,1));
d_tmp = d_raw(k);

x0_fit{1}(latID) =...
    lsqr(Csort_trans_tmp,(log(mean(Y(id,1:10),2))-d_tmp(id)));
X_fit{1}(latID,:) = ppasmoo_poissexp_v2(Y(id,:),Csort_trans_tmp,d_tmp(id),...
    x0_fit{1}(latID),Q0_f(nClus_tmp),...
    A_fit{1}(latID, latID),b_fit{1}(latID),Q_fit{1}(latID, latID));

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
    [X_fit{g},x0_fit{g},d_fit{g},C_fit{g},...
    mudc_fit{g}, Sigdc_fit{g},...
    b_fit{g},A_fit{g},Q_fit{g}] =...
    blockDiag_gibbsLoop_DP_v2(Y,X_fit{g-1}, Z_fit(:,g-1), d_fit{g-1}, C_fit{g-1},...
    mudc_fit{g-1}, Sigdc_fit{g-1},...
    x0_fit{g-1}, b_fit{g-1}, A_fit{g-1}, Q_fit{g-1}, s_star,...
    Q0_f, mux00_f, Sigx00_f, deltadc0, Taudc0,Psidc0,nudc0,...
    mubA0_all_f, SigbA0_f, Psi0,nu0);
    
    
    % (5) update Z
    LAM_tmp = zeros(N, T, s_star);
    LLHD = zeros(N, s_star);
    for k=1:s_star
        latID = id2id(k,p);
        LAM_tmp(:,:,k) = exp(C_fit{g}(:,latID)* X_fit{g}(latID,:) + d_fit{g}(:,k));
        LLHD(:, k) = sum(log(poisspdf(Y, LAM_tmp(:,:,k))), 2);
    end
    
    rho_tmp2 = eta2rho(eta_tmp2);
    LLHD2 = ones(N, s_star)*-Inf;
    LLHD2(u_tmp < rho_tmp2) = LLHD(u_tmp < rho_tmp2);
    clus_tmp = mnrnd(ones(N, 1), softmax(LLHD2')');
    Z_fit(:,g) = Z_fit(:,g-1);
    [Z_fit(~isnan(clus_tmp(:,1)),g), ~] = find(clus_tmp(~isnan(clus_tmp(:,1)),:)');
    
    figure(2)
    clusterPlot(Y, Z_fit(:,g)')
    
end















