addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%%
load('719161530_spikes.mat') % spike times for each unit
tab = readtable('719161530_units.csv'); % meta-data
% tab.ecephys_structure_acronym % names of anatomical structures for each single-unit
subset = find(tab.snr>3 & cellfun(@length,Tlist')>1000);
idx50 = find(ismember(string(tab.ecephys_structure_acronym),...
    ["VISp" "VPM" "SUB" "VISam"])); % LP

% "CA1" "LGd" "POL" "VISp" "VISl"
% VISam
%     Var1 Freq
% 1    APN  176
% 2    CA1  108
% 7    LGd   71
% 11   POL   55
% 17  VISp   52
% 12   SUB   42
% 16  VISl   40
% 15 VISam   37
% 8     LP   28
% 21   VPM   24
% 6   grey   19
% 18 VISpm   18
% 9    NOT   16
% 3    CA3   14
% 4     DG   14
% 19 VISrl   10
% 13    TH    9
% 14 VISal    9
% 20    VL    7
% 10    PO    5
% 5    Eth    1

subset2 = intersect(subset, idx50);
N = length(subset2);
lab_all_str = string(tab.ecephys_structure_acronym);
[lab_num_sub2, clusIdx] = findgroups(lab_all_str(subset2));

Tbase = 2000;
T = 1000;
dt = 0.5;
Yraw = zeros(N, T);
for n = 1:N
    for k = 1:T
        Yraw(n,k) = sum((Tlist{subset2(n)} > (dt*(k-1)) + Tbase) &...
            (Tlist{subset2(n)} < (dt*k + Tbase)));
    end
end

figure(1)
[Lab, idx] = sort(lab_num_sub2);
Y = Yraw(idx,:);
clusterPlot(Y, Lab')
title(clusIdx)

figure(2)
imagesc(Y)
colorbar()


%%
p = 4;
%%
% Y is the only observation...
% some functions
lAbsGam = @(x) log(abs(gamma(x)));

%% MCMC settings
rng(1)
ng = 1000;
t_max = 30;

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
r = 1 - (1-0.95)^(1/8);
% r = 0.6;
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

% t_fit(1) = 4;
% Z_fit(:,1) = randsample(4, N, true);
% numClus_fit(1:4,1) = histcounts(Z_fit(:,1));
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

burnIn = 100;
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

%% post-analysis

plot(t_fit)
idxMax = max(Z_fit(:));

countMat = zeros(idxMax, length(unique(Lab)), ng-1);
for k = 2:ng
    for l = Lab(:)'
        countMat(:,l,k-1) = histcounts(Z_fit(Lab == l, k), 1:(idxMax+1));
    end
end

figure(1)
histogram(t_fit(burnIn:ng))

figure(2)
meanCount = mean(countMat(:,:,(burnIn-1):(ng-1)), 3);
imagesc(meanCount([1 2 3 7 6 4 5], :))
colorbar()
ylabel('cluster-model')
xlabel('cluster-true')
title('fitted vs. true: neuron counts')

figure(3)
imagesc(Y)
colorbar()
title('spking counts')




