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
d(1:10) = d(1:10) + 1;
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) =...
        sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k))+[-1 2];
end

X = zeros(p*nClus, T);
x0 = zeros(p*nClus, 1);
Q0 = eye(nClus*p);
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

clusterPlot(Y, Lab)
%%
% Y is the only observation...
% some functions
lAbsGam = @(x) log(abs(gamma(x)));

%% MCMC settings
rng(1)
ng = 1000;
t_max = N;

% this is the DP setting, replace to MFM later...
% DPMM = true;
% alpha_random = true;
% sigma_alpha = 0.1; % scale for MH proposals in alpha move
% alphaDP = 1;
% log_v = (1:t_max+1)*log(alphaDP) - lAbsGam(alphaDP+N) + lAbsGam(alphaDP);
% a = 1;
% b = 0;

% MFM: 
DPMM = false;
alpha_random = false;
MFMgamma = 1;
% K ~ Geometric(r)
% determine r: use CDF cutoff
% F(k) = 1-(1-r)^k
% let F(N/3) = 0.95
% r = 1 - (1-0.95)^(1/15);
r = 0.2;
log_pk = @(k) log(r) + (k-1)*log(1-r);
% K-1 ~ Poisson(lam)
% lam = 1;
% log_pk = @(k) log(poisspdf(k-1, lam));
% pk = zeros(N,1);
% for gg = 1:N
%    pk(gg) = exp(log_pk(gg)); 
% end
% plot(pk)
a = MFMgamma;
b = MFMgamma;
log_v = MFMcoeff(log_pk, MFMgamma, N, t_max + 1);

logNb = log((1:N) + b);

% priors...
% prior.Q0 = eye(p)*1e-2;
prior.Q0 = eye(p)*0.5^2;
prior.mux00 = zeros(p, 1);
prior.Sigx00 = eye(p);

prior.deltadc0 = [0;ones(p,1)];
prior.Taudc0 = eye(p+1);

prior.Psidc0 = eye(p+1)*1e-2;
prior.nudc0 = p+1+2;

prior.BA0 =[0 1]';
prior.Lamb0 = eye(2);
prior.Psi0 = 1e-2;
prior.nu0 = 1+2;

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


for k = 1:t_max+3
    THETA{1}(k) = sample_prior2(prior, N, T, p, true);
end

%% MCMC
useSplitMerge = false;
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
end

burnIn = round(ng/10);
epsilon = 0.01*ones(N,1);


n_split = 5;
n_merge = 5;
zs = ones(N,1);
S = zeros(N,1);

for g = 2:ng
    
    if(g < burnIn);disp("iter " + g + ", changing"); % change epsilon
    elseif(g == burnIn) % tune epsilon
        for(k = 1:N);OPTDC{k}.Madapt=50;end
        disp("iter " + g + ", tuning");
    else;disp("iter " + g + ", tuned");
    end % fix epsilon
    
    % (1) update parameters
    THETA{g} = THETA{g-1};
    for j = 1:t_fit(g-1)
        c = actList(j);
        obsIdx = find(Z_fit(:,g-1) == c);
        
        [THETA{g}(c), epsilon(obsIdx), log_pdf] =...
            update_cluster(Y(obsIdx,:),THETA{g-1}(c),THETA{g}(c),...
            prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx));
    end
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
    
    Z_fit(:,g) = Z_fit(:,g-1);
    numClus_fit(:,g) = numClus_fit(:,g-1);
    t_fit(g) = t_fit(g-1);
    
    
    if DPMM && alpha_random
        % MH move for DP concentration parameter (using p_alpha(a) = exp(-a) = Exp(a|1))
        aprop = alphaDP*exp(rand*sigma_alpha);
        top = t_fit(g)*log(aprop) - lAbsGam(aprop+N) + lAbsGam(aprop) - aprop + log(aprop);
        bot = t_fit(g)*log(alphaDP) - lAbsGam(alphaDP+N) +...
            lAbsGam(alphaDP) - alphaDP + log(alphaDP);
        if rand < min(1, exp(top-bot))
            alphaDP = aprop;
        end
        log_v = (1:t_max+1)*log(alphaDP) - lAbsGam(alphaDP+N) + lAbsGam(alphaDP);
        disp(alphaDP)
    end
    
    
    % (2) split and merge
    if(useSplitMerge) % useSplitMerge; g<burnIn;
        [Z_fit(:,g), actList, numClus_fit(:,g), t_fit(g), THETA{g}] =...
            splitMerge(Y, Z_fit(:,g), zs, S, THETA{g}, actList, N, T, p,...
            numClus_fit(:,g), t_fit(g), prior, a, b, log_v, n_split, n_merge, OPTDC);
        c_next = ordered_next(actList);
    end
    
    
    % (3) resamle Z
    for ii = 1:N
        
        % (a) remove point i from its cluster
        c = Z_fit(ii, g);
        numClus_fit(c,g) = numClus_fit(c,g) - 1;
        if(numClus_fit(c,g) > 0)
            c_prop = c_next;
            THETA{g}(c_prop) = sample_prior2(prior, N, T, p, true);
        else
            c_prop = c;
            actList = ordered_remove(c, actList, t_fit(g));
            t_fit(g) = t_fit(g) - 1;
        end
        
        %(b) compute probabilities for resampling
        log_p = zeros(t_fit(g)+1,1);
        for j = 1:t_fit(g)
            cc = actList(j);
            lamTmp = exp(THETA{g}(cc).C(ii,:)*THETA{g}(cc).X + THETA{g}(cc).d(ii));
            log_p(j) = logNb(numClus_fit(cc,g)) + sum(log(poisspdf(Y(ii,:), lamTmp)));
        end
        
        lamTmp_prop = exp(THETA{g}(c_prop).C(ii,:)*THETA{g}(c_prop).X + THETA{g}(c_prop).d(ii));
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
    
    
    figure(2)
    clusterPlot(Y, Z_fit(:,g)')
    
end


%%
figure(1)
plot(t_fit)
title('number of cluster')

figure(2)
zMax = max(Z_fit(:));
Z_trace = Z_fit;
Z_trace2 = Z_trace + 0.2*rand(N,1);
hold on
for k = 1:N
    p(k)=plot(Z_trace2(k,:));
    if k < n+1
        set(p(k),'Color', 'r');
    elseif k < 2*n+1
        set(p(k),'Color', 'g');
    else
        set(p(k),'Color', 'b');
    end
end
hold off
ylim([0 zMax+1])
yticks(1:zMax)
% xlim([0 10])
clusLab = [];
for c = 1:zMax
    clusLab{c} = 'cluster ' + string(c);
end
yticklabels(clusLab)
title('cluster trace for each neuron')


figure(3)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()



