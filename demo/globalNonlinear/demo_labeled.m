addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%% simulation
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



%% MCMC settings
p=1;
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
Ctrace = zeros(nClus, ng);
for k = 1:nClus
   Ctrace(k,1) = norm(THETA{1}(k).C, 'fro'); 
end


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
            update_cluster_new(Y(obsIdx,:),THETA{g-1}(j),THETA{g}(j),...
            prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx));
    end
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
    
    for k = 1:nClus
        Ctrace(k,g) = norm(THETA{g}(k).C, 'fro');
    end
    
    figure(1)
    plot(Ctrace(:,1:g)')
    
    
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

%% some plots

dSum1 = zeros(1,T);
dSum2 = zeros(1,T);
dSum3 = zeros(1,T);
xSum1 = zeros(p,T);
xSum2 = zeros(p,T);
xSum3 = zeros(p,T);

dxSum1 = zeros(p+1,T);
dxSum2 = zeros(p+1,T);
dxSum3 = zeros(p+1,T);


x50_1 = zeros(p, ng);
x50_2 = zeros(p, ng);
x50_3 = zeros(p, ng);


c = 0;
for g= 1:ng
    
    x50_1(:, g) = THETA{g}(1).X(:, T/2);
    x50_2(:, g) = THETA{g}(2).X(:, T/2);
    x50_3(:, g) = THETA{g}(3).X(:, T/2);
    
    if (g >= 500)
        c= c+1;
        dSum1 = dSum1 + THETA{g}(1).d;
        dSum2 = dSum2 + THETA{g}(2).d;
        dSum3 = dSum3 + THETA{g}(3).d;
        
        xSum1 = xSum1 + THETA{g}(1).X;
        xSum2 = xSum2 + THETA{g}(2).X;
        xSum3 = xSum3 + THETA{g}(3).X;
        
        dxSum1 = dxSum1 + [THETA{g}(1).d;THETA{g}(1).X];
        dxSum2 = dxSum2 + [THETA{g}(2).d;THETA{g}(2).X];
        dxSum3 = dxSum3 + [THETA{g}(3).d;THETA{g}(3).X];
    end
end


dxMean1 = dxSum1/c;
dxMean2 = dxSum2/c;
dxMean3 = dxSum3/c;

dxMean1 = dxMean1 - mean(dxMean1, 2);
[dxMean1, ~] = mgson(dxMean1');

dxMean2 = dxMean2 - mean(dxMean2, 2);
[dxMean2, ~] = mgson(dxMean2');

dxMean3 = dxMean3 - mean(dxMean3, 2);
[dxMean3, ~] = mgson(dxMean3');


figure(3)
subplot(3,4,1)
plot(X(id2id(1,2),:)')
title('dX-trans: true')
subplot(3,4,2)
plot(dSum1'/c)
title('d:fit')
subplot(3,4,3)
plot(xSum1'/c)
title('X-raw:fit')
subplot(3,4,4)
plot(dxMean1)
title('dX-trans:fit')

subplot(3,4,5)
plot(X(id2id(2,2),:)')
subplot(3,4,6)
plot(dSum2'/c)
subplot(3,4,7)
plot(xSum2'/c)
subplot(3,4,8)
plot(dxMean2)

subplot(3,4,9)
plot(X(id2id(3,2),:)')
subplot(3,4,10)
plot(dSum3'/c)
subplot(3,4,11)
plot(xSum3'/c)
subplot(3,4,12)
plot(dxMean3)


% trace plot
dtrace = zeros(nClus, ng);
Xtrace = zeros(nClus, ng);
Ctrace = zeros(nClus, ng);

dtrace_all = zeros(ng,1);
Xtrace_all = zeros(ng,1);
Ctrace_all = zeros(ng,1);

for g = 1:ng
    
    dTmp = zeros(nClus,T);
    XTmp = zeros(p*nClus,T);
    
    for k = 1:nClus
        dTmp(k,:) = THETA{g}(k).d;
        XTmp(id2id(k,p),:) = THETA{g}(k).X;
        dtrace(k,g) = norm(THETA{g}(k).d);
        Xtrace(k,g) = norm(THETA{g}(k).X, 'fro');
        Ctrace(k,g) = norm(THETA{g}(k).C, 'fro');
        
    end
    dtrace_all(g) = norm(dTmp, 'fro');
    Xtrace_all(g) = norm(XTmp, 'fro');
end

gSub = [1:4000 6000:ng];
% gSub = 1:1000;

ngSub = length(gSub);
subplot(3,1,1)
plot(dtrace(:,gSub)')
ylim([0 60])
xlim([0 ngSub])
subplot(3,1,2)
plot(Xtrace(:,gSub)')
ylim([0 20])
xlim([0 ngSub])
subplot(3,1,3)
plot(Ctrace(:,gSub)')
xlim([0 ngSub])



