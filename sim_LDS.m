addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(1)
n = 10;
nClus = 3;
p = 3;
T = 1000;

Z = repelem(1:3, n);
d = ones(n*nClus,1)*0.5;
C_all = reshape(normrnd(6*1e-3,1e-3,n*nClus*p,1), [], p);
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Z)
    C_trans(k, ((Z(k)-1)*p+1):(Z(k)*p)) = C_all(k,:);
end

X1 = zeros(p, T);
X2 = zeros(p, T);
X3 = zeros(p, T);

Q0 = eye(p)*1e-3;
mu1 = [1 0.9 0.8];
mu2 = [0.7 0.6 0.5];
mu3 = [0.4 0.3 0.2];
X1(:,1) = mvnrnd(mu1,Q0);
X2(:,1) = mvnrnd(mu2,Q0);
X3(:,1) = mvnrnd(mu3,Q0);
X = [X1; X2; X3];

b1 = ones(p,1)*0.05;
b2 = -ones(p,1)*0.05;
b3 = -ones(p,1)*0.05;
b = [b1;b2;b3];

Q1 = 1e-3*eye(p);
Q2 = 1e-4*eye(p);
Q3 = 1e-5*eye(p);
Q = blkdiag(Q1, Q2, Q3);

A_in = reshape(normrnd(0,1e-3,p*p*nClus,1), [], p, nClus);
A = blkdiag(A_in(:,:,1), A_in(:,:,2), A_in(:,:,3));
A((p+1):2*p, 1:p) = reshape(normrnd(1e-3,5*1e-4,p*p,1), [], p);
A(1:p, (p+1):2*p) = reshape(normrnd(1e-3,5*1e-4,p*p,1), [], p);
A((2*p+1):3*p, (p+1):2*p) = reshape(normrnd(-2*1e-4,5*1e-4,p*p,1), [], p);
A((p+1):2*p, (2*p+1):3*p) = reshape(normrnd(-2*1e-4,5*1e-4,p*p,1), [], p);
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
for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

figure(2)
imagesc(logLam)
colorbar()

Y = poissrnd(exp(logLam));
figure(3)
imagesc(Y)
colorbar()

figure(4)
clusterPlot(Y, Z)

figure(5)
plot(X')

%% fitting
% assume now observe Y and Z
% later Y will be the only observation

ng = 10;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(n*nClus, ng);
C_fit = zeros(n*nClus, p, ng);
x0_fit = zeros(nClus*p, ng);
Q0_fit = zeros(p, p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

% initials
% initial for d_fit: 0
C_fit(:,:,1) = reshape(normrnd(0,1,n*nClus*p,1), [], p);
A_fit(:,:,1) = eye(nClus*p);
% initial for b_fit: 0
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

C_trans_tmp = zeros(n*nClus, p*nClus);
for k = 1:length(Z)
    C_trans_tmp(k, ((Z(k)-1)*p+1):(Z(k)*p)) = C_fit(k,:,1);
end

x0_fit(:,1) = lsqr(C_trans_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
Q0_fit(:,:,1) = eye(p);
Qc = repmat({Q0_fit(:,:,1)}, 1, nClus);
Q0Filt = blkdiag(Qc{:});

[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0Filt,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

% prior parameters
mud0 = zeros(n*nClus,1);
Sigd0 = eye(n*nClus);

% Please do all the things blockwise,...
% to transfer smoothly to clustering problem
for g = 2:ng
    
    % (1) update latent vectors x_t^{(j)}
    % cannot do blockwise, because of interaction...
    % in the clustering case, need to reorder the observation and matrix...
    % here, I just ignore that for simpicity...
    Qc = repmat({Q0_fit(:,:,g-1)}, 1, nClus);
    Q0Filt = blkdiag(Qc{:});
    C_trans_tmp = zeros(n*nClus, p*nClus);
    for k = 1:length(Z)
        C_trans_tmp(k, ((Z(k)-1)*p+1):(Z(k)*p)) = C_fit(k,:,g-1);
    end
    
    [Xs,Ws,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,g-1),...
        x0_fit(:,g-1),Q0Filt,A_fit(:,:,g-1),b_fit(:,g-1),Q_fit(:,:,g-1));
    X_fit(:,:,g) = mvnrnd(Xs',Ws)';
    
    % (2) update d: do things block-wise/reorder for clustering
    % here just don't do that for simplicity
    lamd = @(d) exp(C_trans_tmp*X_fit(:,:,g) + d);
    
    derd = @(d) sum(Y - lamd(d), 2) - inv(Sigd0)*(d - mud0);
    hesd = @(d) -diag(sum(lamd(d),2)) - inv(Sigd0);
    [mud,~,niSigd,~] = newton(derd,hesd,d_fit(:,g-1),1e-6,1000);
    Sigd = -inv(niSigd);
    d_fit(:,g) = mvnrnd(mud,Sigd)';
    
    % (3) update C: must do things block-wise here...
    
    
end









