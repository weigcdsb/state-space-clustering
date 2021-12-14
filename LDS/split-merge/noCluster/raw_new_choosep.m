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
%% MCMC settings
rng(1)
pMax = 5;
ng = 1000;

for p = 1:pMax
    
    prior(p).theta0 = zeros(1+p,1);
    prior(p).Q0 = eye(1+p);
    prior(p).muC0 = zeros(p,1);
    prior(p).SigC0 = eye(p);
    
    prior(p).BA0 =[0 1]';
    prior(p).Lamb0 = eye(2);
    prior(p).Psi0 = 1e-2;
    prior(p).nu0 = 1+2;
    
    for k = 1:nClus
        Ntmp = sum(Lab == k);
        THETApre{p,1}(k) = sample_prior_new2(prior(p), Ntmp, T, p, false, Inf);
    end
end

%% preRun
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
end
nPre = 40;
epsilon = 0.01*ones(N,1);

for g = 2:nPre
    disp("iter " + g)
    
    % pre-calculation to get approximate posterior
    for p = 1:pMax
        THETApre{p,g} = THETApre{p,g-1};
        for j = 1:nClus
            obsIdx = find(Lab == j);
            
            [THETApre{p,g}(j), epsilon(obsIdx), ~] =...
                update_cluster_new2(Y(obsIdx,:),...
                THETApre{p,g-1}(j),THETApre{p,g}(j),...
                prior(p), T, p, obsIdx, true, false, OPTDC(obsIdx));
        end
    end
end

%% calculate approximated mean & variance, by pre-samples
for p=1:pMax
    for j=1:nClus
        
        idxtmp = find(Lab == j);
        Ntmp = length(idxtmp);
        
        dtmp = zeros(T, size(THETApre, 2)-1);
        Xtmp = zeros(T, size(THETApre, 2)-1, p);
        Ctmp = zeros(Ntmp,size(THETApre, 2)-1, p);
        Atmp = zeros(p+1,size(THETApre, 2)-1);
        btmp = zeros(p+1,size(THETApre, 2)-1);
        Qtmp = zeros(p+1,size(THETApre, 2)-1);
        
        for gg = 2:size(THETApre, 2)
            dtmp(:,gg-1) = THETApre{p,gg}(j).d;
            Atmp(:,gg-1) = diag(THETApre{p,gg}(j).A);
            btmp(:,gg-1) = THETApre{p,gg}(j).b;
            Qtmp(:,gg-1) = diag(THETApre{p,gg}(j).Q);
            
            for k = 1:p
                Xtmp(:,gg-1,k) = THETApre{p,gg}(j).X(k,:)';
                Ctmp(:,gg-1,k) = THETApre{p,gg}(j).C(:,k);
            end
        end
        
        Qtran{p}(j).Md = mean(dtmp, 2);Qtran{p}(j).Vd = cov(dtmp');
        Qtran{p}(j).MX = zeros(T, p);Qtran{p}(j).VX = zeros(T,T,p);
        Qtran{p}(j).MC = zeros(Ntmp,p);Qtran{p}(j).VC = zeros(Ntmp,Ntmp,p);
        Qtran{p}(j).MA = mean(Atmp, 2);Qtran{p}(j).VA = cov(Atmp');
        Qtran{p}(j).Mb = mean(btmp, 2);Qtran{p}(j).Vb = cov(btmp');
        Qtran{p}(j).MQ = mean(Qtmp, 2);Qtran{p}(j).VQ = cov(Qtmp');
        for k = 1:p
            Qtran{p}(j).MX(:,k) = mean(Xtmp(:,:,k), 2);
            VXtmp = cov(Xtmp(:,:,k)');
            Qtran{p}(j).VX(:,:,k) = VXtmp;
            
            % perturbation of C
            Ctmp2 = Ctmp(:,:,k);
            idx0 = find(var(Ctmp2,0,2) <1e-6);
            Ctmp2(idx0,:) = Ctmp2(idx0,:) +...
                1e-4*randn(length(idx0), size(Ctmp2,2));
            Qtran{p}(j).MC(:,k) = mean(Ctmp2,2);
            VCtmp = cov(Ctmp2');
            Qtran{p}(j).VC(:,:,k) = VCtmp;
        end
    end
end

%% RJMCMC
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
end

burnIn = 10;
epsilon = 0.01*ones(N,1);

pFit = zeros(nClus,ng, 1);
pFit(:,1) = ones(3,1);

for j = 1:nClus
    THETA{1}(j) = THETApre{1,nPre}(j);
end


for g = 2:ng
    
    if(g < burnIn);disp("iter " + g + ", changing"); % change epsilon
    elseif(g == burnIn) % tune epsilon
        for(k = 1:N);OPTDC{k}.Madapt=50;end
        disp("iter " + g + ", tuning");
    else;disp("iter " + g + ", tuned");
    end % fix epsilon
    
    
    for j = 1:nClus
        obsIdx = find(Lab == j);
        Ntmp = length(obsIdx);
        
        % propose p
        if(pFit(j,g-1) == 1)
            pStar = pFit(j,g-1) + 1;
            ljStar = log(1);
            ljRev = log(0.5);
        elseif(pFit(j,g-1) == pMax)
            pStar = pFit(j,g-1) - 1;
            ljStar = log(1);
            ljRev = log(0.5);
        else
            pStar = pFit(j,g-1) + randsample([-1 1],1);
            ljStar = log(0.5);
            ljRev = log(0.5);
        end
        
        THETAstar.d = mvnrnd(Qtran{pStar}(j).Md, 0.81*Qtran{pStar}(j).Vd)';
        THETAstar.X = zeros(T, pStar);
        THETAstar.C = zeros(Ntmp, pStar);
        THETAstar.A = zeros(pStar+1);
        THETAstar.b = zeros(pStar+1,1);
        THETAstar.Q = zeros(pStar+1);
        
        THETAstar.A(1,1) = normrnd(Qtran{pStar}(j).MA(1),...
            0.81*Qtran{pStar}(j).VA(1,1));
        THETAstar.b(1) = normrnd(Qtran{pStar}(j).Mb(1),...
            0.81*Qtran{pStar}(j).Vb(1,1));
        THETAstar.Q(1,1) = 1/gamrnd(1.6, 1/(1.6*Qtran{pStar}(j).MQ(1)));
        
        for pp = 1:pStar
            THETAstar.X(:,pp) = mvnrnd(Qtran{pStar}(j).MX(:,pp),...
                0.81*Qtran{pStar}(j).VX(:,:,pp))';
            THETAstar.C(:,pp) = mvnrnd(Qtran{pStar}(j).MC(:,pp),...
                0.81*Qtran{pStar}(j).VC(:,:,pp))';
            THETAstar.A(pp+1,pp+1) = normrnd(Qtran{pStar}(j).MA(pp+1),...
            0.81*Qtran{pStar}(j).VA(pp+1,pp+1));
            THETAstar.b(pp+1) = normrnd(Qtran{pStar}(j).Mb(pp+1),...
            0.81*Qtran{pStar}(j).Vb(pp+1,pp+1));
            THETAstar.Q(pp+1,pp+1) = 1/gamrnd(1.6, 1/(1.6*Qtran{pStar}(j).MQ(pp+1)));
        end
        
        % MH ratio
        lpdfStar = sum(log(poisspdf(Y(obsIdx,:),...
            exp([ones(Ntmp,1) THETAstar.C]*...
            [THETAstar.d' ;THETAstar.X']))),'all');
        lPriorStar = logPrior(THETAstar.d,THETAstar.X,THETAstar.C,...
            THETAstar.b,THETAstar.A,THETAstar.Q,prior(pStar));
        lq = logQ(THETA{g-1}(j).d',THETA{g-1}(j).X',THETA{g-1}(j).C,...
            THETA{g-1}(j).b,THETA{g-1}(j).A,THETA{g-1}(j).Q,Qtran{pFit(j,g-1)}(j));
        
        lpdf = sum(log(poisspdf(Y(obsIdx,:),...
            exp([ones(Ntmp,1) THETA{g-1}(j).C]*...
            [THETA{g-1}(j).d ;THETA{g-1}(j).X]))),'all');
        lPrior = logPrior(THETA{g-1}(j).d',THETA{g-1}(j).X',...
            THETA{g-1}(j).C,...
            THETA{g-1}(j).b,THETA{g-1}(j).A,THETA{g-1}(j).Q,prior(pFit(j,g-1)));
        lqStar = logQ(THETAstar.d,THETAstar.X,THETAstar.C,...
            THETAstar.b,THETAstar.A,THETAstar.Q,Qtran{pStar}(j));
        
        log_alph = lpdfStar + lPriorStar + lq + ljStar-...
            (lpdf + lPrior + lqStar + ljRev);
        
        if rand < min(1, exp(log_alph))
            pFit(j,g) = pStar;
            % TODO...
            THETAstar.d = THETAstar.d';
            THETAstar.X = THETAstar.X';
            THETA{g}(j) = THETAstar;
        else
            pFit(j,g) = pFit(j,g-1);
            % TODO...
            % sample again...
            
            [THETA{g}(j), epsilon(obsIdx), ~] =...
            update_cluster_new2(Y(obsIdx,:),THETA{g-1}(j),THETA{g-1}(j),...
            prior(pFit(j,g)), T, pFit(j,g), obsIdx, true, false, OPTDC(obsIdx));
        end
        
    end
    
    disp(pFit(:,g))
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
end

