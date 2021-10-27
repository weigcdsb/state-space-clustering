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
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) =...
        sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k))+[-2 2];
end

X = zeros(p*nClus, T);
x0 = zeros(p*nClus, 1);
Q0 = eye(nClus*p)*1e-2;
X(:,1) = mvnrnd(x0, Q0)';

b1 = randn(p, 1)*1e-3;
b2 = randn(p, 1)*1e-3;
b3 = randn(p, 1)*1e-4;
b = [b1;b2;b3];

Q1 = 1e-3*eye(p);
Q2 = 1e-4*eye(p);
Q3 = 1e-5*eye(p);
Q = blkdiag(Q1, Q2, Q3);

%
A = eye(size(Q,1));
while any(imag(eig(A))==0)
    A= randn(size(Q));
    A = A-diag(diag(A));
    A(squareform(pdist(pLab'))==0)=0;
    A = A./sqrt(sum((A-diag(diag(A))).^2,2))*0.1;
    %     A = A+eye(size(Q,1))*0.92;
    A = A + diag(randn(nClus*p, 1)*1e-1 + 0.92);
end

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));

gtmp = -min(X,[], 2);
M = inv(diag(range(X,2)));
g = M*gtmp;

X = M*X + g;
x0 = M*x0 + g;
d = d - (C_trans/M)*g;
C_trans = C_trans/M;
A = M*A/M;
b = M*b + g - (M*A/M)*g;
Q = M*Q*M';
Q0 = M*Q0*M';

%%
% Y is the only observation...
% some functions
lAbsGam = @(x) log(abs(gamma(x)));

%%
ng = 100;
alphaDP = 5;
t_max = N;

% this is the DP setting, replace to MFM later...
log_v = (1:t_max+1)*log(alphaDP) - lAbsGam(alphaDP+N) + lAbsGam(alphaDP);
a = 1;
b = 0;
logNb = log((1:N) + b);

% pre-allocation
t_fit = zeros(1, ng);
Z_fit = zeros(N, ng);
numClus_fit = zeros(t_max + 3, ng);

t_fit(1) = 1;
Z_fit(:,1) = ones(N, 1);
numClus_fit(1,1) = N;
actList = zeros(t_max+3,1); actList(1) = 1;
c_next = 2;

% t_fit(1) = 30;
% Z_fit(:,1) = 1:N;
% numClus_fit(1:N,1) = ones(N,1);
% actList = zeros(t_max+3,1);
% actList(1:N) = 1:N;
% c_next = N+1;

% t_fit(1) = 4;
% Z_fit(:,1) = [ones(1,n) 2*ones(1,n/2) 3*ones(1,n/2) 4*ones(1,n)];
% numClus_fit(1:4,1) = [n n/2 n/2 n];
% actList = zeros(t_max+3,1);
% actList(1:4) = 1:4;
% c_next = 5;


% priors
Q0_f = @(nClus) eye(nClus*p)*1e-2;

mux00_f = @(nClus) zeros(nClus*p, 1);
Sigx00_f = @(nClus) eye(nClus*p);

deltadc0 = [0;ones(p,1)];
Taudc0 = eye(p+1);

Psidc0 = eye(p+1)*1e-2;
nudc0 = p+1+2;

BA0 =[0 1]';
Lamb0 = eye(2);
Psi0 = 1e-2;
nu0 = 1+2;

clusMax = t_max+3;
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
X_fit{1}(latID,:) = X_fit{1}(latID,:) - min(X_fit{1}(latID,:),[], 2);
X_fit{1}(latID,:) = (diag(range(X_fit{1}(latID,:),2)))\X_fit{1}(latID,:);

figure(2)
clusterPlot(Y, Z_fit(:,1)')

%% MCMC
optdc.M=1;
optdc.Madapt=0;
epsilon = 0.01*ones(N,1);
burnIn = 10;

for g = 2:ng
    
    % (1) update parameters
    [X_fit{g},x0_fit{g},d_fit{g},C_fit{g},...
        mudc_fit{g}, Sigdc_fit{g},...
        b_fit{g},A_fit{g},Q_fit{g}, optdc, epsilon] =...
        update_all(Y,X_fit{g-1}, Z_fit(:,g-1), d_fit{g-1}, C_fit{g-1},...
        mudc_fit{g-1}, Sigdc_fit{g-1},...
        x0_fit{g-1}, b_fit{g-1}, A_fit{g-1}, Q_fit{g-1},...
        Q0_f, mux00_f, Sigx00_f, deltadc0, Taudc0,Psidc0,nudc0,...
        BA0, Lamb0, Psi0,nu0, g, optdc, epsilon, burnIn);
    
%     figure(3)
%     subplot(2,2,1)
%     plot(X_fit{g}(1:2,:)')
%     subplot(2,2,2)
%     plot(X_fit{g}(3:4,:)')
%     subplot(2,2,3)
%     plot(X_fit{g}(5:6,:)')
%     subplot(2,2,4)
%     plot(X_fit{g}(7:8,:)')
    % (2) resamle Z
    t_fit(g) = t_fit(g-1);
    numClus_fit(:,g) = numClus_fit(:,g-1);
    
    for ii = 1:N
        
        % (a) remove point i from its cluster
        c = Z_fit(ii, g-1);
        numClus_fit(c,g) = numClus_fit(c,g) - 1;
        if(numClus_fit(c,g) > 0)
            c_prop = c_next;
            latID_prop = id2id(c_prop, p);
            [d_fit{g}(:,c_prop), C_fit{g}(:,latID_prop), X_fit{g}(latID_prop,:)] =...
                sample_prior(Q0_f, mux00_f, Sigx00_f, deltadc0, Taudc0,Psidc0,nudc0,...
                Psi0,nu0, N, T, p);
        else
            c_prop = c;
            actList = ordered_remove(c, actList, t_fit(g));
            t_fit(g) = t_fit(g) - 1;
        end
        
        %(b) compute probabilities for resampling
        log_p = zeros(t_fit(g)+1,1);
        for j = 1:t_fit(g)
            cc = actList(j);
            latID = id2id(cc,p);
            lamTmp = exp(C_fit{g}(ii,latID)* X_fit{g}(latID,:) + d_fit{g}(ii,cc));
            log_p(j) = logNb(numClus_fit(cc,g)) + sum(log(poisspdf(Y(ii,:), lamTmp)));
        end
        
        latID_prop = id2id(c_prop, p);
        lamTmp_prop = exp(C_fit{g}(ii,latID_prop)* X_fit{g}(latID_prop,:) + d_fit{g}(ii,c_prop));
        log_p(t_fit(g)+1) = log_v(t_fit(g)+1)-log_v(t_fit(g)) +...
            log(a) + sum(log(poisspdf(Y(ii,:), lamTmp_prop)));
        
        % (c) sample a new cluster for it
        j = randlogp(log_p, t_fit(g)+1);
        
        % (d) add point i to its new clusters
        if j <= t_fit(g)
            c = actList(j);
        else
            c = c_prop;
            actList = ordered_insert(c, actList, t_fit(g));
            t_fit(g) = t_fit(g) + 1;
            c_next = ordered_next(actList);
        end
        
        Z_fit(ii,g) = c;
        numClus_fit(c,g) = numClus_fit(c,g) + 1;
    end
    
%     % (3) split-merge
%     % (a) randomly choose a pair of indices
%     zs = ones(N,1);
%     S = zeros(N,1);
%     
%     rdIdx = randsample(N,2);
%     ism = rdIdx(1);
%     jsm = rdIdx(2);
%     
%     ci0 = Z_fit(ism,g);
%     cj0 = Z_fit(jsm,g);
%     
%     ns = 0;
%     for k = 1:N
%         if(Z_fit(k,g) == ci0 || Z_fit(k,g) == cj0)
%             ns = ns + 1;
%             S(ns) = k;
%         end
%     end
%     
%     % (b) find available cluster IDs for merge & split parameters
%     k = 1;
%     while(actList(k) == k); k = k+1;end;cm = k;
%     while(actList(k) == k+1); k = k+1;end;ci = k+1;
%     while(actList(k) == k+2); k = k+1;end;cj = k+2;
    
    
    
    
    
    figure(2)
    clusterPlot(Y, Z_fit(:,g)')
    
end




