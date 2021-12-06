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
C_trans = zeros(N, p*nClus);
C = zeros(N, p);
for k = 1:length(Lab)
    C(k,:) = 2*repmat(sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k)),1,2)+[-2 1];
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C(k,:);
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
clusterPlot(Y, Lab)

% X_jX_j' = I
% X' = X*'R --> inv(R')X = X*
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


% diagonalize Q within each block
M2 = zeros(nClus*p);
for l = unique(Lab)
    latid = id2id(l,p);
    [M2(latid,latid),~] = eig(Q(latid, latid));
end

X = M2*X;
x0 = M2*x0;
C_trans = C_trans/M2;
A = M2*A/M2;
b = M2*b;
Q = M2*Q*M2';
Q0 = M2*Q0*M2';

%%
rng(1)
nTrain = round(T*1/2);
Y_train = nan*Y;
Y_test = nan*Y;
for k = 1:N
    trIdx = randsample(T, nTrain);
    teIdx = setdiff(1:T, trIdx);
    Y_train(k,trIdx) = Y(k,trIdx);
    Y_test(k, teIdx) = Y(k,teIdx);
end


clusterPlot(Y_train, Lab)
clusterPlot(Y_test, Lab)


%% MCMC settings

nClus = 1;
p=2*3-1;
Lab = ones(1,N);

% p=1;
rng(1)
ng = 1000;

% theta = (d x)
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

%%
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
end

burnIn = 10;
epsilon = 0.01*ones(N,1);

for g = 2:ng
    
    if(g < burnIn);disp("iter " + g + ", changing"); % change epsilon
    elseif(g == burnIn) % tune epsilon
        for(k = 1:N);OPTDC{k}.Madapt=50;end
        disp("iter " + g + ", tuning");
    else;disp("iter " + g + ", tuned");
    end % fix epsilon
    
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
    
%     figure(1)
%     subplot(3,3,1)
%     plot(X(1:p,:)')
%     title('true')
%     subplot(3,3,2)
%     plot(THETA{g}(1).d)
%     title('d')
%     subplot(3,3,3)
%     plot(THETA{g}(1).X')
%     title('X')
%     
%     subplot(3,3,4)
%     plot(X(p+1:2*p,:)')
%     subplot(3,3,5)
%     plot(THETA{g}(2).d)
%     subplot(3,3,6)
%     plot(THETA{g}(2).X')
%     
%     subplot(3,3,7)
%     plot(X(2*p+1:3*p,:)')
%     subplot(3,3,8)
%     plot(THETA{g}(3).d)
%     subplot(3,3,9)
%     plot(THETA{g}(3).X')
    
    
    figure(2)
    subplot(1,2,1)
    imagesc(exp(C_trans*X + d))
    cLim = caxis;
    title('true')
    colorbar()
    subplot(1,2,2)
    fitMFR = zeros(N, T);
    for k  = 1:nClus
        N_tmp = sum(Lab == k);
        fitMFR(Lab == k,:) = exp([ones(N_tmp,1) THETA{g}(k).C(Lab == k,:)]*...
            [THETA{g}(k).d ;THETA{g}(k).X]);
    end
    imagesc(fitMFR)
    set(gca,'CLim',cLim)
    colorbar()
    title('fit')
end


%%
idx = 500:1000;
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
imagesc(fitMFR)

nansum(log(poisspdf(Y_test,fitMFR)), 'all')/nansum(Y_test, 'all')








