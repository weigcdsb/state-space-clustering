addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%%
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
        (sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k))+[-1 2])*0.5;
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
% imagesc(Y)
% colorbar()

%%
nRep = 100;
propTrain = 3/4;
ng = 100;
Ytest_spk = zeros(nRep, 1);
llhd_spk_test_3clus = zeros(nRep, 1);
llhd_spk_test_1clus = zeros(nRep, 1);

rng(1)
for r = 1:nRep
    
    disp("iter " + r);
    % sample train-test
    nTrain = round(T*propTrain);
    Y_train = nan*Y;
    Y_test = nan*Y;
    for k = 1:N
        trIdx = randsample(T, nTrain);
        teIdx = setdiff(1:T, trIdx);
        Y_train(k,trIdx) = Y(k,trIdx);
        Y_test(k, teIdx) = Y(k,teIdx);
    end
    
    
    % 3 clusters, p = 1
    nClus = 3;
    p = 1;
    Lab = repelem(1:nClus, n);
    
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
    
    burnIn = 10;
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
    end
    
    idx = round(ng/2):ng;
    dSum = zeros(nClus, T);
    XSum = zeros(nClus*p, T);
    CSum = zeros(N, p);
    
    for g = idx
        for ii = 1:nClus
            dSum(ii,:) = dSum(ii,:) + THETA{g}(ii).d;
            XSum(id2id(ii,p),:) = XSum(id2id(ii,p),:) + THETA{g}(ii).X;
            CSum(Lab == ii, :) = CSum(Lab == ii, :) +...
                THETA{g}(ii).C(Lab == ii, :);
        end
    end
    
    dMean = dSum/length(idx);
    XMean = XSum/length(idx);
    CMean = CSum/length(idx);
    
    fitMFR = zeros(N, T);
    for k  = 1:nClus
        N_tmp = sum(Lab == k);
        fitMFR(Lab == k,:) = exp([ones(N_tmp,1) CMean(Lab == k,:)]*...
            [dMean(k,:) ;XMean(id2id(k,p),:)]);
    end
    
    llhd_spk_test_3clus(r) =...
        nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
    Ytest_spk(r) = nansum(Y_test, 'all');
    
    % 1 cluster, p = 5
    nClus = 1;
    p=2*3-1;
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
    
    burnIn = 10;
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
    end
    
    idx = round(ng/2):ng;
    dSum = zeros(nClus, T);
    XSum = zeros(nClus*p, T);
    CSum = zeros(N, p);
    
    for g = idx
        for ii = 1:nClus
            dSum(ii,:) = dSum(ii,:) + THETA{g}(ii).d;
            XSum(id2id(ii,p),:) = XSum(id2id(ii,p),:) + THETA{g}(ii).X;
            CSum(Lab == ii, :) = CSum(Lab == ii, :) +...
                THETA{g}(ii).C(Lab == ii, :);
        end
    end
    
    dMean = dSum/length(idx);
    XMean = XSum/length(idx);
    CMean = CSum/length(idx);
    
    fitMFR = zeros(N, T);
    for k  = 1:nClus
        N_tmp = sum(Lab == k);
        fitMFR(Lab == k,:) = exp([ones(N_tmp,1) CMean(Lab == k,:)]*...
            [dMean(k,:) ;XMean(id2id(k,p),:)]);
    end
    
    llhd_spk_test_1clus(r) =...
        nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
    
end

%% plot
hold on
plot(llhd_spk_test_3clus)
plot(llhd_spk_test_1clus)
hold off

boxplot([llhd_spk_test_3clus - llhd_spk_test_1clus;...
    llhd_spk_test_3clus_half - llhd_spk_test_1clus_half],...
    repelem(1:2, nRep))

%% get trace of llhd

propTrain_set = [3/4, 1/2];
llhd_spk_train = zeros(4, ng);
llhd_spk_test = zeros(4, ng);
c = 0;

nClus_set = [3,1];
p_set = [1, 5];
Lab_set{1} = repelem(1:3, n);
Lab_set{2} = ones(1,N);

for kk = 1:2
    nTrain = round(T*propTrain_set(kk));
    Y_train = nan*Y;
    Y_test = nan*Y;
    for k = 1:N
        trIdx = randsample(T, nTrain);
        teIdx = setdiff(1:T, trIdx);
        Y_train(k,trIdx) = Y(k,trIdx);
        Y_test(k, teIdx) = Y(k,teIdx);
    end
    
    for ll = 1:2
        c = c + 1;
        disp("iter " + c);
        nClus = nClus_set(ll);
        p = p_set(ll);
        Lab = Lab_set{ll};
        
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
        
        burnIn = 10;
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
            
            % train_llhd-spk
            fitMFR = zeros(N, T);
            for k  = 1:nClus
                N_tmp = sum(Lab == k);
                fitMFR(Lab == k,:) = exp([ones(N_tmp,1) THETA{g}(k).C(Lab == k,:)]*...
                    [THETA{g}(k).d ;THETA{g}(k).X]);
            end
            
            llhd_spk_train(c,g) =...
                nansum(log(poisspdf(Y_train,fitMFR)), 'all')/nansum(Y_train, 'all');
            
            % test-llhd-spk
            llhd_spk_test(c,g) =...
                nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all');
        end
    end
end


plot(llhd_spk_train(1:2,50:end)')
plot(llhd_spk_test(1:2,50:end)')

plot(llhd_spk_train(3:4,50:end)')
plot(llhd_spk_test(3:4,50:end)')














