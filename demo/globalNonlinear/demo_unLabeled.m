addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%% simulation
close all;
rng(123)
n = 5;
nClus = 10;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);

d = randn(n*nClus,1)/5;
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = (sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k)) + [-1.5 0])*0.5;
end


X = zeros(p*nClus, T);
x0 = zeros(p*nClus,1);
Q0 = eye(nClus*p)*1e-2;
X(:,1) = mvnrnd(x0, Q0)';

b = zeros(p*nClus,1);
Q = eye(nClus*p)*1e-3;

% Generate X offline (A unspecified)
for i=1:size(Q,1)
    k = ceil(rand()*20)+10;
    X(i,:) = interp1(linspace(0,1,k),randn(k,1),linspace(0,1,T),'spline');
end

for kk = [3 5 8 9]
    idTmp = id2id(kk,p);
    k = ceil(rand()*25)+10;
    for j = 1:p
        X(idTmp(j),:) = interp1(linspace(0,1,k),randn(k,1),linspace(0,1,T),'spline');
    end
end

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    logLam(:, t) = d + C_trans*X(:,t);
end

% figure(1)
Y = poissrnd(exp(logLam));
% clusterPlot(Y, Lab)


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
Q = M*Q*M';
Q0 = M*Q0*M';

subplot(1,2,1)
imagesc(exp(C_trans*X + d))
colorbar()
subplot(1,2,2)
imagesc(Y)


%%
% Y is the only observation...
% some functions
lAbsGam = @(x) log(abs(gamma(x)));

%% MCMC settings
rng(1)
p=1;
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
r = 0.2;
log_pk = @(k) log(r) + (k-1)*log(1-r);

a = MFMgamma;
b = MFMgamma;
log_v = MFMcoeff(log_pk, MFMgamma, N, t_max + 1);

logNb = log((1:N) + b);

% priors...
% theta = (d x)
prior.theta0 = zeros(1+p,1);
prior.Q0 = eye(1+p);

prior.muC0 = zeros(p,1);
prior.SigC0 = eye(p);

prior.BA0 =[0 1]';
prior.Lamb0 = eye(2);
prior.Psi0 = 1e-3;
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

for k = 1:t_fit(1)
    THETA{1}(k) = sample_prior_new(prior, N, T, p, false, Inf);
end

%% MCMC
burnIn = round(ng/20);
epsilon = 0.05*ones(N,1);

for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
%     OPTDC{k}.epsilon = epsilon(k);
end

simMat = zeros(N,N);
for k = 1:size(simMat, 1)
    simMat(k,:) = simMat(k,:) + (Z_fit(k, 1) == Z_fit(:, 1))';
end
fitMFRTrace = zeros(N, T, ng);
llhd_trace = zeros(ng,1);

for g = 2:ng
    
    if(g < burnIn);disp("iter " + g + ", changing"); % change epsilon
    elseif(g == burnIn) % tune epsilon
        for(k = 1:N);OPTDC{k}.Madapt=50;end
        disp("iter " + g + ", tuning");
    else;disp("iter " + g + ", tuned");
    end % fix epsilon
    
    THETA{g} = THETA{g-1};
    for j = 1:t_fit(g-1)
        c = actList(j);
        obsIdx = find(Z_fit(:,g-1) == c);
        
        [THETA{g}(c), epsilon(obsIdx), log_pdf] =...
            update_cluster_new(Y(obsIdx,:),THETA{g-1}(c),THETA{g}(c),...
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
    
    
    % (3) resamle Z
    for ii = 1:N
        
        % (a) remove point i from its cluster
        c = Z_fit(ii, g);
        numClus_fit(c,g) = numClus_fit(c,g) - 1;
        if(numClus_fit(c,g) > 0)
            c_prop = c_next;
            THETA{g}(c_prop) = sample_prior_new(prior, N, T, p, false, Inf);
        else
            c_prop = c;
            actList = ordered_remove(c, actList, t_fit(g));
            t_fit(g) = t_fit(g) - 1;
        end
        
        %(b) compute probabilities for resampling
        log_p = zeros(t_fit(g)+1,1);
        for j = 1:t_fit(g)
            cc = actList(j);
            logMar = poiLogMarg(Y(ii,:)', THETA{g}(cc).X', THETA{g}(cc).d');
            log_p(j) = logNb(numClus_fit(cc,g)) + logMar;
        end
        
        logMar = poiLogMarg(Y(ii,:)', THETA{g}(c_prop).X', THETA{g}(c_prop).d');
        log_p(t_fit(g)+1) = log_v(t_fit(g)+1)-log_v(t_fit(g)) +...
            log(a) + logMar;
        
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
    
    figure(1)
    clusterPlot(Y, Z_fit(:,g)')
    
    figure(2)
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit(k, g) == Z_fit(:, g))';
    end
    
    imagesc(simMat/g)
    colormap(flipud(hot))
    colorbar()
    
    
    figure(3)
    subplot(1,2,1)
    imagesc(Y)
    cLim = caxis;
    title('true')
    colorbar()
    subplot(1,2,2)
    fitMFR = zeros(N, T);
    for k  = 1:N
        fitMFR(k,:) = exp([1 THETA{g}(Z_fit(k,g)).C(k,:)]*...
            [THETA{g}(Z_fit(k,g)).d ;THETA{g}(Z_fit(k,g)).X]);
    end
    fitMFRTrace(:,:,g) = fitMFR;
    imagesc(fitMFR)
    set(gca,'CLim',cLim)
    colorbar()
    title('fit')
    
    llhd_trace(g) = nansum(log(poisspdf(Y,fitMFR)), 'all')/nansum(Y, 'all');
    
    
    figure(4)
    plot(t_fit(1:g))
    
end

%%
close all;
figure(1)
plot(t_fit)
title('number of cluster')

figure(2)
idx = round(g/2):g;
simMat = zeros(N,N);
for g = idx
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit(k, g) == Z_fit(:, g))';
    end
end

imagesc(simMat/length(idx))
colormap(flipud(hot))
colorbar()








