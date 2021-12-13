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
        THETA{p,1}(k) = sample_prior_new(prior(p), N, T, p, false, Inf);
    end
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
    
    % pre-calculation to get approximate posterior
    if(g<burnIn)
        for p = 1:pMax
            THETA{p,g} = THETA{p,g-1};
            for j = 1:nClus
                obsIdx = find(Lab == j);
                
                [THETA{p,g}(j), epsilon(obsIdx), log_pdf] =...
                    update_cluster_new(Y(obsIdx,:),THETA{p,g-1}(j),THETA{p,g}(j),...
                    prior(p), N, T, p, obsIdx, true, false, OPTDC(obsIdx));
            end
        end
    elseif(g== burnIn)
        for p=1:pMax
            for j=1:nClus
                dtmp = zeros
                Xtmp = zeros(T, size(THETA, 2)-1, p);
                Atmp = zeros(p+1,size(THETA, 2)-1);
                btmp = zeros(p+1,size(THETA, 2)-1);
                Ctmp = zeros(N,size(THETA, 2)-1, p);
                
                for gg = 2:size(THETA, 2)
                    for k = 1:p
                        Xtmp(:,gg-1,k) = THETA{p,gg}(j).X(k,:)';
                    end
                end
                
                MXtmp = zeros(T, p);
                VXtmp = zeros(T,T,p);
                for k = 1:p
                    MXtmp(:,k) = mean(Xtmp(:,:,k), 2);
                    VXtmp(:,:,k) = cov(Xtmp(:,:,k)');
                end
                
                
            end
        end
        
    else
        
    end
    
    
    
    
    
    
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
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




