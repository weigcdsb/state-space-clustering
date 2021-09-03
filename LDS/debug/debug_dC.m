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

d = randn(N,1);
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k))+1;
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
imagesc(exp(logLam))
colorbar()
% clusterPlot(Y, Lab)

ng = 100; % 40
idx = 50:ng; % 20:ng

%% fitting: MCMC, no update prior

d_fit = zeros(N, ng);
C_fit = zeros(N, p, ng);
mudc0 = zeros(p+1,1);
Sigdc0 = eye(p+1);
C_fit(:,:,1) = reshape(normrnd(0,1e-2,N*p,1), [], p);

X_fit = repmat(X, 1, 1, ng); % true
x0_fit = repmat(x0',1,ng); %true
A_fit = repmat(A,1,1,ng); % true
b_fit = repmat(b,1,ng); % true
Q_fit = repmat(Q,1,1,ng); % true

for g = 2:ng
    
    disp(g)
    
    for i = 1:N
        l = Lab(i);
        latentId = ((l-1)*p+1):(l*p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - inv(Sigdc0)*(dc - mudc0);
        hessdc = @(dc) -X_tmp'*diag(lamdc(dc))*X_tmp - inv(Sigdc0);
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
            [d_fit(i, g-1) C_fit(i,:,g-1)]',1e-8,1000);
        
        Sigdc = -inv(niSigdc);
        Sigdc = (Sigdc + Sigdc')/2;
        dc = mvnrnd(mudc, Sigdc);
        
        d_fit(i,g) = dc(1);
        C_fit(i,:,g) = dc(2:end);
    end
end

figure
subplot(1,3,1)
plot(d,mean(d_fit(:,idx), 2),'rx');
ylabel('estimate')
title('d')
C_fit_mean = mean(C_fit(:,:,idx), 3);
subplot(1,3,2)
plot(sum(C_trans(:,1:p:end), 2),C_fit_mean(:,1),'rx');
title('1st column of C')
xlabel('true')
subplot(1,3,3)
plot(sum(C_trans(:,2:p:end), 2),C_fit_mean(:,2),'rx');
title('2nd column of C')
sgtitle('No Update of Prior')

%% fitting: MCMC, update priors
X_fit2 = repmat(X, 1, 1, ng); % true
x0_fit2 = repmat(x0',1,ng); %true
A_fit2 = repmat(A,1,1,ng); % true
b_fit2 = repmat(b,1,ng); % true
Q_fit2 = repmat(Q,1,1,ng); % true

% pre-allocation
d_fit2 = zeros(N, nClus, ng);
C_fit2 = zeros(N, nClus*p, ng);
mudc_fit2 = zeros(p+1, nClus, ng);
Sigdc_fit2 = zeros(p+1, p+1, nClus, ng);

% priors
deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1);

Psidc0 = eye(p+1);
nudc0 = p+1+2;
Sigdc_fit2(:,:,1:nClus,1) = repmat(eye(p+1)*1e-2,1,1,nClus);

for g = 2:ng
    
    disp(g)
    
    for i = 1:N
        l = Lab(i);
        latentId = id2id(l,p);
        X_tmp = [ones(1, T) ;X_fit2(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - Sigdc_fit2(:,:,l,g-1)\(dc - mudc_fit2(:,l,g-1));
        hessdc = @(dc) -X_tmp'*diag(lamdc(dc))*X_tmp - inv(Sigdc_fit2(:,:,l,g-1));
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
            [d_fit2(i,l, g-1) C_fit2(i,latentId,g-1)]',1e-8,1000);
        
        Sigdc = -inv(niSigdc);
        Sigdc = (Sigdc + Sigdc')/2;
        dc = mvnrnd(mudc, Sigdc);
        d_fit2(i,l,g) = dc(1);
        C_fit2(i,latentId,g) = dc(2:end);
    end
    
    % (4) update mudc_fit2 & Sigdc_fit2
    for l = unique(Lab)
        dc_tmp = [d_fit2(Lab == l, l, g) C_fit2(Lab == l, id2id(l,p), g)];
        invTaudc = inv(Taudc0) + sum(Lab == l)*inv(Sigdc_fit2(:,:,l,g-1));
        deltadc = invTaudc\(Taudc0\deltadc0 + Sigdc_fit2(:,:,l,g-1)\sum(dc_tmp,1)');
        mudc_fit2(:,l,g) = mvnrnd(deltadc, inv(invTaudc));
        
        % assume different covariances 
        dcRes = dc_tmp' - mudc_fit2(:,l,g);
        Psidc = Psidc0 + dcRes*dcRes';
        nudc = sum(Lab == l) + nudc0;
        Sigdc_fit2(:,:,l,g) = iwishrnd(Psidc,nudc);
    end
end

figure
subplot(1,3,1)
plot(d,sum(mean(d_fit2(:,:,idx), 3),2),'rx');
ylabel('estimate')
title('d')
C_fit2_mean = mean(C_fit2(:,:,idx), 3);
subplot(1,3,2)
plot(sum(C_trans(:,1:p:end), 2),sum(C_fit2_mean(:,1:p:end), 2),'rx');
xlabel('true')
title('1st column of C')
subplot(1,3,3)
plot(sum(C_trans(:,2:p:end), 2),sum(C_fit2_mean(:,2:p:end), 2),'rx');
title('2nd column of C')
sgtitle('with Update of Prior')

% plot(squeeze(mudc_fit2(1,1,:)))
% plot(squeeze(mudc_fit2(1,2,:)))
% plot(squeeze(mudc_fit2(1,3,:)))
% plot(squeeze(mudc_fit2(2,1,:)))

    






