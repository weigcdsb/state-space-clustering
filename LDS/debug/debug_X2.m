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
ng = 100;

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



C_trans_tmp = zeros(N, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
end
[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

% X_fit(:,:,1) = zeros(nClus*p, T);

%%
opt.M=1;
opt.Madapt=0;
opt.delta = 0.5
x_norm = zeros(ng, 1);
x_norm(1) = norm(X_fit(:,:,1), 'fro');
rej_path = zeros(ng,1);
tic;
for g = 2:50
    
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
    
    logNPrior = @(X) -1/2*(X(:,1) - x0_tmp)'*inv(Q0)*(X(:,1) - x0_tmp) -...
        1/2*trace((X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp)'*inv(Q_tmp)*...
        (X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp));
    
    lamX = @(X) exp(C_trans_tmp*X + d_tmp) ;
    lpdf = @(vecX) sum(log(poisspdf(Y, lamX(reshape(vecX, [], T)))), 'all') +...
        logNPrior(reshape(vecX, [], T));
    glpdf = @(vecX) derX(vecX, d_tmp, C_trans_tmp, x0_tmp,...
        Q0, Q_tmp, A_tmp, b_tmp, Y);
    
    fg=@(vecX_r) deal(lpdf(vecX_r'), glpdf(vecX_r')'); % log density and gradient
    
    [muXvec_NUTS, energies, diagn]=hmc_nuts(fg, reshape(X_fit(:,:,g-1),[],1)',...
        opt);
    rej_path(g) = diagn.rej;
    disp(rej_path(g))
    X_fit(:,:,g) = reshape(muXvec_NUTS(2,:),[], T);
    
    x_norm(g) = norm(X_fit(:,:,g), 'fro');
    figure(1)
    plot(x_norm(1:g))
    
end
toc;




