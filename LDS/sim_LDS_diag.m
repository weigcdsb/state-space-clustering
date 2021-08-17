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
ng = 1000;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
x0_fit = zeros(nClus*p, ng);
d_fit = zeros(n*nClus, ng);
C_fit = zeros(n*nClus, p, ng);
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

muA0 = eye(p);
mubA0 = [zeros(nClus*p,1) eye(nClus*p)];
Sigba0 = eye(nClus*p+1)*0.25;

nu0 = 4;
sig20 = 1e-4;

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
    
    % [C_fit(:,:,g) C_all]
    % [d_fit(:,g) d]
    
    % (4) update b_fit & A_fit: independent process noise
    for k = 1:size(X_fit, 1)
        
        X_tmp = [ones(1, T-1) ;X_fit(:,1:(T-1),g)]';
        Sigba = inv(inv(Sigba0) + X_tmp'*X_tmp./Q_fit(k,k,g-1));
        Sigba = (Sigba + Sigba')/2;
        muba = Sigba*(inv(Sigba0)*mubA0(k,:)' + X_tmp'*X_fit(k,2:T,g)'./Q_fit(k,k,g-1));
        ba = mvnrnd(muba, Sigba);
        b_fit(k,g) = ba(1);
        A_fit(k,:,g) = ba(2:end);
    end
    
    % [b_fit(:,g) b]
    % imagesc(abs(A_fit(:,:,g) - A))
    
    % (5) update Q_fit: independent process noise
    for k = 1:size(X_fit, 1)
        
        mux = b_fit(k,g) + A_fit(k,:,g)*X_fit(:,1:(T-1),g);
        
        alphq = (nu0 + T-1)/2;
        betaq = (nu0*sig20 + sum((X_fit(k,2:T,g) - mux).^2))/2;
        Q_fit(k,k,g) = 1/gamrnd(alphq, 1/betaq);
    end
    
    % Q_fit(:,:,g)
    
end

%% diagnose
idx = 200:ng;

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
subplot(3,2,2)
plot(mean(X_fit(1:p,:,idx), 3)')
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
imagesc(mean(A_fit(:,:,idx), 3))
colorbar()
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


