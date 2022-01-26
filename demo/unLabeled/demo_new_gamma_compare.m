%% split train & test
rng(0)
propTrain = 1/4;
nTrain = round(T*propTrain);
Y_train = nan*Y;
Y_test = nan*Y;
for k = 1:N
    trIdx = randsample(T, nTrain);
    teIdx = setdiff(1:T, trIdx);
    Y_train(k,trIdx) = Y(k,trIdx);
    Y_test(k, teIdx) = Y(k,teIdx);
end



%% cluter-on
rng(1)
p=1;
ng = 1000;
t_max = N;

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

burnIn = round(ng/10);
epsilon = 0.01*ones(N,1);
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
%     OPTDC{k}.epsilon = epsilon(k);
end

fitMFRTrace = zeros(N, T, ng);
llhd_spk_train = zeros(ng,1);
llhd_spk_test = zeros(ng,1);

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
            update_cluster_new_na(Y_train(obsIdx,:),THETA{g}(c),THETA{g}(c),...
                prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx), Y_train);
    end
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
    
    
    % test-llhd-spk
    fitMFR = zeros(N, T);
    for j  = 1:t_fit(g-1)
        c = actList(j);
        N_tmp = sum(Z_fit(:,g-1) == c);
        fitMFR(Z_fit(:,g-1) == c,:) = exp([ones(N_tmp,1) THETA{g}(c).C(Z_fit(:,g-1) == c,:)]*...
            [THETA{g}(c).d ;THETA{g}(c).X]);
    end
    
    fitMFRTrace(:,:,g) = fitMFR;
    llhd_spk_train(g) = nansum(log(poisspdf(Y_train,fitMFR)), 'all')/nansum(Y_train, 'all');
    llhd_spk_test(g) = nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
    
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
    
    figure(1)
    clusterPlot(Y, Z_fit(:,g)')
    
    figure(3)
    plot(llhd_spk_test(2:g))
    
    figure(4)
    plot(t_fit(1:g))
    
end

%% no cluster...choose p
rng(2)
p_set = 1:6;
ng = 100;
nClus = 1;
Lab_sing = ones(1,N);
llhd_spk_test_sing = zeros(length(p_set), ng);

for kk = 1:length(p_set)
    p = p_set(kk);
    
    prior.theta0 = zeros(1+p,1);
    prior.Q0 = eye(1+p);
    prior.muC0 = zeros(p,1);
    prior.SigC0 = eye(p);
    prior.BA0 =[0 1]';
    prior.Lamb0 = eye(2);
    prior.Psi0 = 1e-2;
    prior.nu0 = 1+2;
    
    for k = 1:nClus
        THETA{1}(k) = sample_prior_new(prior, N, T, p, false, Inf);
    end
    
    
    for k = 1:N
        optdc.M=1;
        optdc.Madapt=0;
        OPTDC{k} = optdc;
    end
    
    burnIn = round(ng/10);
    epsilon = 0.01*ones(N,1);
    
    for g = 2:ng
        
        if(g == burnIn)
            for(k = 1:N);OPTDC{k}.Madapt=50;end
        end
        
        THETA{g} = THETA{g-1};
        for j = 1:nClus
            obsIdx = find(Lab_sing == j);
            
            [THETA{g}(j), epsilon(obsIdx), log_pdf] =...
                update_cluster_new_na(Y_train(obsIdx,:),THETA{g-1}(j),THETA{g}(j),...
                prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx), Y_train);
        end
        
        if(g == burnIn)
            for k = 1:N
                OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
            end
        end
        
        % test-llhd-spk
        fitMFR = zeros(N, T);
        for k  = 1:nClus
            N_tmp = sum(Lab_sing == k);
            fitMFR(Lab_sing == k,:) = exp([ones(N_tmp,1) THETA{g}(k).C(Lab_sing == k,:)]*...
                [THETA{g}(k).d ;THETA{g}(k).X]);
        end
        
        
        llhd_spk_test_sing(kk,g) =...
            nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
        figure(98)
        plot(llhd_spk_test_sing(1:kk, 2:g)')
    end
    
end

[~,pid_sing] = max(mean(llhd_spk_test_sing(:,(round(ng/2):ng)), 2));

%% no cluster...
rng(3)
ng = 1000;
nClus = 1;
Lab_sing = ones(1,N);
llhd_spk_test_sing2 = zeros(ng,1);

p = p_set(pid_sing);

prior.theta0 = zeros(1+p,1);
prior.Q0 = eye(1+p);
prior.muC0 = zeros(p,1);
prior.SigC0 = eye(p);
prior.BA0 =[0 1]';
prior.Lamb0 = eye(2);
prior.Psi0 = 1e-2;
prior.nu0 = 1+2;

for k = 1:nClus
    THETA{1}(k) = sample_prior_new(prior, N, T, p, false, Inf);
end


burnIn = round(ng/10);
epsilon = 0.01*ones(N,1);
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
    OPTDC{k}.epsilon = epsilon(k);
end


for g = 2:ng
    
    if(g == burnIn)
        for(k = 1:N);OPTDC{k}.Madapt=50;end
    end
    
    THETA{g} = THETA{g-1};
    for j = 1:nClus
        obsIdx = find(Lab_sing == j);
        
        [THETA{g}(j), epsilon(obsIdx), log_pdf] =...
            update_cluster_new_na(Y_train(obsIdx,:),THETA{g-1}(j),THETA{g}(j),...
            prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx), Y_train);
    end
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
    
    % test-llhd-spk
    fitMFR = zeros(N, T);
    for k  = 1:nClus
        N_tmp = sum(Lab_sing == k);
        fitMFR(Lab_sing == k,:) = exp([ones(N_tmp,1) THETA{g}(k).C(Lab_sing == k,:)]*...
            [THETA{g}(k).d ;THETA{g}(k).X]);
    end
    
    llhd_spk_test_sing2(g) =...
        nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
    figure(98)
    hold on
    plot(llhd_spk_test(2:g), 'r')
    plot(llhd_spk_test_sing2(2:g), 'b')
    hold off
end

%%
ng = 1000;
idx = round(ng/2):ng;
hold on
histogram(llhd_spk_test(idx))
histogram(llhd_spk_test_sing2(idx))
hold off






