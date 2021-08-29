addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(2)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
d = ones(n*nClus,1)*0;
C_all = reshape(normrnd(0.08,1e-3,n*nClus*p,1), [], p);
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_all(k,:);
end

X = zeros(p*nClus, T);
x0 = [1.2 1.2 0.5 0.5 1 1]*10;
Q0 = eye(nClus*p)*1e-2;
X(:,1) = mvnrnd(x0, Q0)';

b1 = ones(p,1)*0.01;
b2 = ones(p,1)*0;
b3 = ones(p,1)*-0.03;
b = [b1;b2;b3];

Q1 = 1e-5*eye(p);
Q2 = 1e-4*eye(p);
Q3 = 1e-3*eye(p);
Q = blkdiag(Q1, Q2, Q3);


A = [1 0 0 0 0.4 -0.4;...
    0 1 0 0 -0.3 0.305;
    0 0 1 0 -0.2 0.19;
    0 0 0 1 0.11 -0.1;
    0 0 0 0 1 0;
    0 0 0 0 0 1];

figure(1)
imagesc(A)
colorbar()
xlabel('sending')
ylabel('receiving')

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

figure(2)
plot(X')

figure(3)
% imagesc(logLam)
imagesc(exp(logLam))
colorbar()

Y = poissrnd(exp(logLam));
figure(4)
imagesc(Y)
colorbar()

figure(5)
clusterPlot(Y, Lab)

% plot(X(1:p,:)')
% plot(X(p+1:2*p,:)')
% plot(X(2*p+1:3*p,:)')

%% fitting: pre-setting
% assume now observe Y and Lab
% later Y will be the only observation for clustering problem
ng = 100;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(n*nClus, ng);
C_fit = zeros(n*nClus, p, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

% priors
% place-holder...
Q0 = eye(nClus*p);

mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p)*1e2;

mudc0 = zeros(p+1,1);
Sigdc0 = eye(p+1)*1e-2;

muBA0_mat = [zeros(nClus*p,1) eye(nClus*p)];
muBA0 = muBA0_mat(:);
SigBA0 = sparse(eye(length(muBA0))*0.25);

Psi0 = eye(nClus*p)*1e-4;
nu0 = nClus*p+2;

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
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

% turn off Q estimation (for debug)
% Q_fit = repmat(Q,1,1,ng); % true
%% fitting: MCMC
for g = 2:ng
    
    disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    C_trans_tmp = zeros(n*nClus, p*nClus);
    for k = 1:length(Lab)
        C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,g-1);
    end
    d_tmp = d_fit(:,g-1);
    x0_tmp = x0_fit(:,g-1);
    A_tmp = A_fit(:,:,g-1);
    b_tmp = b_fit(:,g-1);
    Q_tmp = Q_fit(:,:,g-1);
    
    [muX,~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
    hess_tmp = hessX(muX(:), d_tmp, C_trans_tmp, Q0, Q_tmp, A_tmp, Y);
    % tic;
    % use Cholesky decomposition to sample efficiently
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muX(:)), 1) + R'*muX(:);
    Xsamp = R'\z;
    X_fit(:,:,g) = reshape(Xsamp,[], T);
    % toc;
    
    % (2) update x0_fit
    Sigx0 = inv(inv(Sigx00) + inv(Q0));
    mux0 = Sigx0*(Sigx00\mux00 + Q0\X_fit(:,1,g));
    x0_fit(:,g) = mvnrnd(mux0, Sigx0)';
    % disp(x0_fit(:,g))
    
    % (3) update d_fit & C_fit
    % Laplace approximation
    for i = 1:N
        l = Lab(i);
        latentId = ((l-1)*p+1):(l*p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - inv(Sigdc0)*(dc - mudc0);
        hessdc = @(dc) -X_tmp'*diag(lamdc(dc))*X_tmp - inv(Sigdc0);
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
            [d_fit(i, g-1) C_fit(i,:,g-1)]',1e-8,1000);
        
        % [mudc [d(i) C_all(i,:)]']
        
        Sigdc = -inv(niSigdc);
        Sigdc = (Sigdc + Sigdc')/2;
        dc = mvnrnd(mudc, Sigdc);
        d_fit(i,g) = dc(1);
        C_fit(i,:,g) = dc(2:end);
    end
    
    %     [C_fit(:,:,g) C_all]
    %     [d_fit(:,g) d]
    
    % (4) update b_fit & A_fit
    Z_tmp = X_fit(:,2:T,g);
    Z_tmp2 = Z_tmp(:);
    
    G_tmp = sparse(kron([ones(1,T-1); X_fit(:, 1:(T-1), g)]', eye(nClus*p)));
    invSigBA_tmp = sparse(inv(SigBA0) + G_tmp'*kron(eye(T-1), inv(Q_fit(:,:,g-1)))*G_tmp);
    
    muBA_tmp = invSigBA_tmp\(SigBA0\muBA0 +...
        G_tmp'*kron(eye(T-1), inv(Q_fit(:,:,g-1)))*Z_tmp2);
    
    % use Cholesky decomposition
    R = chol(invSigBA_tmp, 'lower');
    z = randn(length(muBA_tmp), 1) + R'*muBA_tmp;
    BAsamp = reshape(R'\z, nClus*p, []);
    b_fit(:,g) = BAsamp(:,1);
    A_fit(:,:,g) = BAsamp(:,2:end);
    
    %     [b_fit(:,g) b]
    %     imagesc(abs(A_fit(:,:,g) - A))
    %     colorbar()
    
    % (5) update Q_fit
    mux = A_fit(:,:,g)*X_fit(:,1:(T-1),g) + b_fit(:,g);
    xq = X_fit(:,2:T,g) - mux;
    
    PsiQ = Psi0 + xq*xq';
    nuQ = T-1+nu0;
    Q_fit(:,:,g) = iwishrnd(PsiQ,nuQ);
    
end
%% diagnose
% idx = 200:ng;
idx = 50:ng;

% for k = round(linspace(2, 1000, 10))
%     figure(k)
%     subplot(1,2,1)
%     plot(X_fit(:,:,k)')
%     subplot(1,2,2)
%     plot(X')
% end

subplot(1,2,1)
plot(mean(X_fit(:,:,idx), 3)')
subplot(1,2,2)
plot(X')

subplot(3,2,1)
plot(X(1:p,:)')
title('true')
subplot(3,2,2)
plot(mean(X_fit(1:p,:,idx), 3)')
title('fit')
subplot(3,2,3)
plot(X(p+1:2*p,:)')
subplot(3,2,4)
plot(mean(X_fit(p+1:2*p,:,idx), 3)')
subplot(3,2,5)
plot(X(2*p+1:3*p,:)')
subplot(3,2,6)
plot(mean(X_fit(2*p+1:3*p,:,idx), 3)')


[mean(d_fit(:,idx), 2) d]
[mean(C_fit(:,:,idx), 3) C_all]
[mean(x0_fit(:,idx), 2) x0']
abs(mean(A_fit(:,:,idx), 3) - A)
[mean(b_fit(:,idx), 2) b]

subplot(1,2,1)
imagesc(A)
colorbar()
cLim = caxis;
title('true')
subplot(1,2,2)
imagesc(mean(A_fit(:,:,idx), 3))
colorbar()
set(gca,'CLim',cLim)
title('fit')

mean(Q_fit(:,:,idx), 3)

C_fit_mean = mean(C_fit(:,:,idx), 3);
C_trans_fit = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_fit(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit_mean(k,:);
end
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()
subplot(1,2,2)
imagesc(exp(C_trans_fit*mean(X_fit(:,:,idx), 3) + mean(d_fit(:,idx), 2)))
set(gca,'CLim',cLim)
title('fit')
colorbar()

