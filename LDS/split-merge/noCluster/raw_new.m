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
    THETA{1}(k) = sample_prior_new(prior, N, T, p, false);
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
            update_cluster_new(Y(obsIdx,:),THETA{g-1}(j),THETA{g}(j),...
            prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx));
    end
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
    
    figure(1)
    subplot(3,3,1)
    plot(X(1:p,:)')
    title('true')
    subplot(3,3,2)
    plot(THETA{g}(1).d)
    title('d')
    subplot(3,3,3)
    plot(THETA{g}(1).X')
    title('X')
    
    subplot(3,3,4)
    plot(X(p+1:2*p,:)')
    subplot(3,3,5)
    plot(THETA{g}(2).d)
    subplot(3,3,6)
    plot(THETA{g}(2).X')
    
    subplot(3,3,7)
    plot(X(2*p+1:3*p,:)')
    subplot(3,3,8)
    plot(THETA{g}(3).d)
    subplot(3,3,9)
    plot(THETA{g}(3).X')
    
    
    figure(2)
    subplot(1,2,1)
    imagesc(exp(C_trans*X + d))
    cLim = caxis;
    title('true')
    colorbar()
    subplot(1,2,2)
    fitMFR = zeros(N, T);
    for k  = 1:3
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

dMean1 = zeros(p,T);
dMean2 = zeros(p,T);
dMean3 = zeros(p,T);

xMean1 = zeros(p,T);
xMean2 = zeros(p,T);
xMean3 = zeros(p,T);

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
        dMean1 = dMean1 + THETA{g}(1).d;
        dMean2 = dMean2 + THETA{g}(2).d;
        dMean3 = dMean3 + THETA{g}(3).d;
        
        xMean1 = xMean1 + THETA{g}(1).X;
        xMean2 = xMean2 + THETA{g}(2).X;
        xMean3 = xMean3 + THETA{g}(3).X;
    end
end

figure(3)
subplot(3,3,1)
plot(X(1:p,:)')
title('true')
subplot(3,3,2)
plot(dMean1'/c)
title('d')
subplot(3,3,3)
plot(xMean1'/c)
title('X')

subplot(3,3,4)
plot(X(p+1:2*p,:)')
subplot(3,3,5)
plot(dMean2'/c)
subplot(3,3,6)
plot(xMean2'/c)

subplot(3,3,7)
plot(X(2*p+1:3*p,:)')
subplot(3,3,8)
plot(dMean3'/c)
subplot(3,3,9)
plot(xMean3'/c)

% trace plot
dtrace = zeros(nClus, ng);
Xtrace = zeros(nClus, ng);
dtrace_all = zeros(ng,1);
Xtrace_all = zeros(ng,1);
for g = 1:ng
    
    dTmp = zeros(nClus,T);
    XTmp = zeros(p*nClus,T);
    for k = 1:nClus
        dTmp(k,:) = THETA{g}(k).d;
        XTmp(id2id(k,p),:) = THETA{g}(k).X;
        dtrace(k,g) = norm(THETA{g}(k).d);
        Xtrace(k,g) = norm(THETA{g}(k).X, 'fro');
    end
    dtrace_all(g) = norm(dTmp, 'fro');
    Xtrace_all(g) = norm(XTmp, 'fro');
end

plot(dtrace_all)
plot(Xtrace(1,:))




