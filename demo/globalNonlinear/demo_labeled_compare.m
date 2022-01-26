addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%%
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


%%
nRep = 50;
propTrain = [3/4 1/2];
ng = 50;
idx = round(ng/2):ng;

Ytest_spk = zeros(nRep, 2);
llhd_spk_test_3clus = zeros(nRep, 2);
llhd_spk_test_1clus_tune = zeros(nRep, 2);
ptune = zeros(nRep,1);

p_set = 4:(p*nClus-1);


rng(1)
warning('off')
for prop = 1:length(propTrain)
    
    propTrain_tmp = propTrain(prop);
    for r = 1:nRep
        
        disp("iter " + r);
        % sample train-test
        nTrain = round(T*propTrain_tmp);
        Y_train = nan*Y;
        Y_test = nan*Y;
        for k = 1:N
            trIdx = randsample(T, nTrain);
            teIdx = setdiff(1:T, trIdx);
            Y_train(k,trIdx) = Y(k,trIdx);
            Y_test(k, teIdx) = Y(k,teIdx);
        end
        
        
        %% 3 clusters, p = 1
        nClus = 10;
        p = 1;
        Lab = repelem(1:nClus, n);
        llhd_spk_test = zeros(ng,1);
        
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
                obsIdx = find(Lab == j);
                
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
                N_tmp = sum(Lab == k);
                fitMFR(Lab == k,:) = exp([ones(N_tmp,1) THETA{g}(k).C(Lab == k,:)]*...
                    [THETA{g}(k).d ;THETA{g}(k).X]);
            end
            llhd_spk_test(g) =...
                nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
            figure(1)
            plot(llhd_spk_test(2:g)')
        end
        
        llhd_spk_test_3clus(r, prop) = mean(llhd_spk_test(idx));
        Ytest_spk(r, prop) = nansum(Y_test, 'all');
        
        
        %% 1 cluster, p tune
        llhd_test_tmp = zeros(ng, length(p_set));
        llhd_test_tmp_mean = zeros(length(p_set),1)*-Inf;
        
        for pp = 1:length(p_set)
            nClus = 1;
            p=p_set(pp);
            Lab = ones(1,N);
            
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
                    obsIdx = find(Lab == j);
                    
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
                    N_tmp = sum(Lab == k);
                    fitMFR(Lab == k,:) = exp([ones(N_tmp,1) THETA{g}(k).C(Lab == k,:)]*...
                        [THETA{g}(k).d ;THETA{g}(k).X]);
                end
                llhd_test_tmp(g, pp) =...
                    nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
                figure(3)
                hold on
                plot(llhd_spk_test(2:g), 'k', 'LineWidth', 2)
                plot(llhd_test_tmp(2:g, 1:pp))
                hold off
            end
            
            llhd_test_tmp_mean(pp) = mean(llhd_test_tmp(idx, pp));
            if llhd_test_tmp_mean(pp) ~= max(llhd_test_tmp_mean)
                break;
            end
        end
        
        [llhd_spk_test_1clus_tune(r, prop), pIdx] = max(llhd_test_tmp_mean);
        ptune(r, prop) = p_set(pIdx);
        close all;
    end
    
end


%%
hold on
plot(llhd_spk_test_3clus(:,1))
plot(llhd_spk_test_1clus_tune(:,1))
hold off

hold on
plot(llhd_spk_test_3clus(:,2))
plot(llhd_spk_test_1clus_tune(:,2))
hold off





