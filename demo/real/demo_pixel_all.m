addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
addpath(genpath('C:\Users\gaw19004\Documents\GitHub\data'));
% addpath(genpath('D:\github\state-space-clustering'));

%%

load('719161530_spikes.mat') % spike times for each unit
tab = readtable('719161530_units.csv'); % meta-data
% tab.ecephys_structure_acronym % names of anatomical structures for each single-unit
subset = find(tab.snr>3 & cellfun(@length,Tlist')>1000);
idx50 = find(ismember(string(tab.ecephys_structure_acronym),...
    ["VISam" "LP" "VPM"]));

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

% spontaneous
% Tstart = 29.8301073815904; Tend = 89.89682738;
Tstart =  1001.89177167499; Tend = 1290.88309738159;

% drifting_gratings 
% Tstart = 1591.13385738159; Tend = 2190.63454310612;

dt = 0.1;
T = min(floor((Tend-Tstart)/dt), 1000);
% T = floor((Tend-Tstart)/dt);
Yraw = zeros(N, T);
for n = 1:N
    for k = 1:T
        Yraw(n,k) = sum((Tlist{subset2(n)} > (dt*(k-1)) + Tstart) &...
            (Tlist{subset2(n)} < (dt*k + Tstart)));
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

% clear Tlist

%%
lAbsGam = @(x) log(abs(gamma(x)));
rng(1)
p=1;
ng = 5000;
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
r = 0.3;
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
prior.Psi0 = 1e-4;
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
    THETA{1}(k) = sample_prior_new(prior, N, T, p, true, Inf);
end

%% MCMC
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
end

burnIn = round(ng/20);
epsilon = 0.01*ones(N,1);

simMat = zeros(N,N);
for k = 1:size(simMat, 1)
    simMat(k,:) = simMat(k,:) + (Z_fit(k, 1) == Z_fit(:, 1))';
end

fitMFRTrace = zeros(N, T, ng);


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
            prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx), Y);
%         [THETA{g}(c), epsilon(obsIdx), log_pdf] =...
%             update_cluster_new_aug(Y(obsIdx,:),THETA{g-1}(c),THETA{g}(c),...
%             prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx), Y);
    end
    
    for k  = 1:N
        fitMFRTrace(k,:, g) = exp([1 THETA{g}(Z_fit(k,g-1)).C(k,:)]*...
            [THETA{g}(Z_fit(k,g-1)).d ;THETA{g}(Z_fit(k,g-1)).X]);
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
            THETA{g}(c_prop) = sample_prior_new(prior, N, T, p, true, Inf);
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
    
    figure(2)
    clusterPlot(Y, Z_fit(:,g)')
    
    
    figure(3)
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit(k, g) == Z_fit(:, g))';
    end
    
    imagesc(simMat/g)
    colormap(flipud(hot))
    colorbar()
    
    figure(4)
    subplot(1,2,1)
    imagesc(Y)
    colorbar()
    title('true')
    subplot(1,2,2)
    imagesc(fitMFRTrace(:,:, g))
    colorbar()
    title('fit')
    
    figure(5)
    plot(t_fit(1:g))
    
end

