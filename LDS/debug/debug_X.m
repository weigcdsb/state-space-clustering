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
C_trans = zeros(N, p*nClus);
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
logLam = zeros(N, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));
clusterPlot(Y, Lab)

%%
rng(3)
ng = 1000;

X_fit = zeros(nClus*p, T, ng);
x0_fit = repmat(x0, 1, ng); % true
d_fit = repmat(d,1,ng); % true
C_fit = repmat([sum(C_trans(:,1:p:end), 2) sum(C_trans(:,2:p:end), 2)],...
    1,1,ng); % true
A_fit = repmat(A,1,1,ng); % true
b_fit = repmat(b,1,ng); % true
Q_fit = repmat(Q,1,1,ng); % true

% priors
Q0 = eye(nClus*p)*1e-2;

% initials
% C_trans_tmp = zeros(N, p*nClus);
% for k = 1:length(Lab)
%     C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
% end
% [X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
%     x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

X_fit(:,:,1) = zeros(nClus*p, T);


%% MCMC
x_all = reshape(X_fit(:,:,1),[],1);
tau = sqrt(T/4);
acc = 0;
for g = 2:ng
    
    disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    
    C_trans_tmp = zeros(N, p*nClus);
    for k = 1:N
        C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,g-1);
    end
    d_tmp = d_fit(:,g-1);
    x0_tmp = x0_fit(:,g-1);
    A_tmp = A_fit(:,:,g-1);
    b_tmp = b_fit(:,g-1);
    Q_tmp = Q_fit(:,:,g-1);
    
    X_tmp = X_fit(:,:,g-1);
    gradHess = @(vecX) gradHessX(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0, Q_tmp, A_tmp, b_tmp, Y);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    if(sum(isnan(muXvec)) ~= 0)
        disp('use adaptive smoother initial')
        X_tmp = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
        [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    end
        
%     R = chol(-hess_tmp,'lower'); % sparse
%     z = randn(length(muXvec), 1) + R'*muXvec;
%     x_all = R'\z;
    
    % let's use MH again...
    if g < round(ng/2) % g < round(ng/2)
        R = chol(-hess_tmp,'lower'); % sparse
        z = randn(length(muXvec), 1) + R'*muXvec;
        x_all = R'\z;
    else
        lamX = @(X) exp(C_trans_tmp*X + d_tmp) ;
        
        R = chol(-hess_tmp,'lower'); % sparse
        R = tau*R;
        z = randn(length(x_all), 1) + R'*x_all;
        xStar = R'\z;
        
        XStar = reshape(xStar, [], T);
        X_all = reshape(x_all, [], T);
        
        logNPrior = @(X) -1/2*(X(:,1) - x0_tmp)'*Q0*(X(:,1) - x0_tmp) -...
            1/2*trace((X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp)'*Q_tmp*...
            (X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp));
        
        % lhr
        lhr = sum(log(poisspdf(Y, lamX(XStar))), 'all') -...
            sum(log(poisspdf(Y, lamX(X_all))), 'all') +...
            logNPrior(XStar) - logNPrior(X_all);
        
        if(log(rand(1)) < lhr)
            x_all = xStar;
            acc = acc + 1;
        end
    end
    
    X_fit(:,:,g) = reshape(x_all,[], T);
    
end

%%
acc/(ng) 

x_norm = zeros(g, 1);
for k = 1:g
    x_norm(k) = norm(X_fit(:,:,k), 'fro');
end

plot(x_norm)


%%
idx = round(ng/2):ng;

figure
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






