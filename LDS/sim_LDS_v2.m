addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(2)
n = 10;
nClus = 3;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
d = ones(n*nClus,1)*0.1;
C_all = reshape(normrnd(4*1e-3,1e-3,n*nClus*p,1), [], p);
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_all(k,:);
end

X = zeros(p*nClus, T);
x0 = [1 0.2 0.2 1 1 0.2]*10;
Q0 = eye(nClus*p)*1e-3;
X(:,1) = mvnrnd(x0, Q0)';

b1 = ones(p,1)*0.3;
b2 = ones(p,1)*0.2;
b3 = ones(p,1)*0.1;
b = [b1;b2;b3];

Q1 = 1e-3*eye(p);
Q2 = 1e-4*eye(p);
Q3 = 1e-5*eye(p);
Q = blkdiag(Q1, Q2, Q3);

A_in = reshape(normrnd(0,5*1e-3,p*p*nClus,1), [], p, nClus);
A = blkdiag(A_in(:,:,1), A_in(:,:,2), A_in(:,:,3));
A((p+1):2*p, 1:p) = reshape(normrnd(3*1e-3,1e-3,p*p,1), [], p);
A(1:p, (p+1):2*p) = reshape(normrnd(3*1e-3,1e-3,p*p,1), [], p);
A((2*p+1):3*p, 1:p) = reshape(normrnd(-4*1e-3,1e-3,p*p,1), [], p);
A(1:p, (2*p+1):3*p) = reshape(normrnd(-4*1e-3,1e-3,p*p,1), [], p);
Aplot = A;
A(eye(size(A)) > 0) = 1;

figure(1)
imagesc(Aplot)
colorbar()
xlabel('sending')
ylabel('receiving')
% A = eye(nClus*p);

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

figure(2)
imagesc(logLam)
% imagesc(exp(logLam))
colorbar()

Y = poissrnd(exp(logLam));
figure(3)
imagesc(Y)
colorbar()

figure(4)
clusterPlot(Y, Lab)

figure(5)
plot(X')

plot(X(1:p,:)')
plot(X(p+1:2*p,:)')
plot(X(2*p+1:3*p,:)')

%% fitting: pre-setting
% assume now observe Y and Lab
% later Y will be the only observation for clustering problem

ng = 100;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(n*nClus, ng);
C_fit = zeros(n*nClus, p, ng);
x0_fit = zeros(nClus*p, ng);
Q0_fit = zeros(nClus*p, nClus*p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

% initials
% initial for d_fit: 0
C_fit(:,:,1) = reshape(normrnd(0,1e-2,n*nClus*p,1), [], p);
A_fit(:,:,1) = eye(nClus*p);
% initial for b_fit: 0
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

C_trans_tmp = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
end
x0_fit(:,1) = lsqr(C_trans_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
Q0_fit(:,:,1) = eye(nClus*p);
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0_fit(:,:,1),A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

% priors
% place-holder...
mudC0 = zeros(n*(p+1),1);
SigdC0 = eye(n*(p+1))*1e-2;

muA0 = eye(p);
mubA0 = [zeros(p,1); muA0(:); zeros((nClus-1)*p^2, 1)];
SigbA0 = eye(p*(1+p*nClus))*1e-2;

Psi0 = eye(p)*1e-4;
nu0 = p+2;

%% fitting: MCMC

for g = 1:ng
    
    disp(g)
    
    % (1) update X_fit, x0_fit & Q0_fit
    % adaptive smoothing
    C_trans_tmp = zeros(n*nClus, p*nClus);
    for k = 1:length(Lab)
        C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,g-1);
    end
    
    [Xs,Ws,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,g-1),...
        x0_fit(:,g-1),Q0_fit(:,:,g-1),A_fit(:,:,g-1),b_fit(:,g-1),Q_fit(:,:,g-1));
    x0_fit(:,g) = Xs(:,1);
    Q0_fit(:,:,g) = Ws(:,:,1);
    X_fit(:,:,g) = mvnrnd(Xs',Ws)';
    
    % (2) update d_fit & C_fit
    % Laplace approximation
    for l = unique(Lab)
        latentId = ((l-1)*p+1):(l*p);
        C_tmp = C_fit(Lab == l,:,g-1);
        d_tmp = d_fit(Lab == l, g-1);
        Y_tmp = Y(Lab==l,:);
        
        derdC_tmp = @(vecdC) derdC(vecdC,Y_tmp, X_fit(latentId,:,g)) -...
            inv(SigdC0)*(vecdC - mudC0);
        hessdC_tmp = @(vecdC) hessdC(vecdC, X_fit(latentId,:,g)) - inv(SigdC0);
        [mudC,~,niSigdC,~] = newton(derdC_tmp,hessdC_tmp,...
            [d_tmp;C_tmp(:)],1e-8,1000);
        
        SigdC = -inv(niSigdC);
        SigdC = (SigdC + SigdC')/2;
        dC = reshape(mvnrnd(mudC,SigdC)',[],1+p);
        d_fit(Lab == l,g) = dC(:,1);
        C_fit(Lab == l,:,g) = dC(:,2:end);
    end
    
    %     [C_fit(:,:,g) C_all]
    %     [d_fit(:,g) d]
    
    % (3) update b_fit & A_fit
    for l = unique(Lab)
        latentId = ((l-1)*p+1):(l*p);
        Z_tmp = X_fit(latentId,2:T,g);
        Z_tmp2 = Z_tmp(:);
        
        X_tmp = kron([ones(1,T-1); X_fit(:, 1:(T-1), g)]', eye(p));
        SigbA_tmp = inv(inv(SigbA0) + X_tmp'*kron(eye(T-1), inv(Q_fit(latentId,latentId,g-1)))*X_tmp);
        SigbA_tmp = (SigbA_tmp + SigbA_tmp')/2;
        mubA_tmp = SigbA_tmp*(inv(SigbA0)*mubA0 +...
            X_tmp'*kron(eye(T-1), inv(Q_fit(latentId,latentId,g-1)))*Z_tmp2);
        bAtmp = reshape(mvnrnd(mubA_tmp, SigbA_tmp)', [], 1+nClus*p);
        b_fit(latentId,g) = bAtmp(:,1);
        A_fit(latentId,:,g) = bAtmp(:,2:end);
    end
    
    %     [b_fit(:,g) b]
    %     imagesc(abs(A_fit(:,:,g) - A))
    %     colorbar()
    
    % (4) update Q_fit
%     for l = unique(Lab)
%         latentId = ((l-1)*p+1):(l*p);
%         mux = A_fit(latentId,:,g)*X_fit(:,1:(T-1),g) +...
%             repmat(b_fit(latentId, g), 1, T-1);
%         xq = X_fit(latentId,2:T,g) - mux;
%         
%         PsiQ = Psi0 + xq*xq';
%         nuQ = T-1 + nu0;
%         Q_fit(latentId,latentId,g) = iwishrnd(PsiQ,nuQ);
%     end
end

%% diagnose
iter = g;

% for k = 1:iter
%     figure(k)
%     subplot(1,2,1)
%     plot(X_fit(:,:,k)')
%     subplot(1,2,2)
%     plot(X')
% end

subplot(1,2,1)
plot(X_fit(:,:,iter)')
subplot(1,2,2)
plot(X')

[d_fit(:,iter) d]
[C_fit(:,:,iter) C_all]
[x0_fit(:,iter) x0']
abs(Q0_fit(:,:,iter) - Q0)
abs(A_fit(:,:,iter) - A)
[b_fit(:,iter) b]
abs(Q_fit(:,:,iter)- Q)

C_trans_fit = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_fit(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,iter);
end
subplot(1,2,1)
imagesc(C_trans_fit*X_fit(:,:,iter) + d_fit(:,iter))
colorbar()
subplot(1,2,2)
imagesc(C_trans*X + d)
colorbar()




