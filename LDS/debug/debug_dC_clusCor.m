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

%%
rng(3)
ng = 2;
d_fit = zeros(N, ng);
C_fit = zeros(N, p, ng);


% prior
mudc0_sing = zeros(p+1,1);
Sigdc0_sing = eye(p+1);

% initial
C_fit(:,:,1) = reshape(normrnd(0,1e-2,N*p,1), [], p);

X_fit = repmat(X, 1, 1, ng); % true
x0_fit = repmat(x0',1,ng); %true
A_fit = repmat(A,1,1,ng); % true
b_fit = repmat(b,1,ng); % true
Q_fit = repmat(Q,1,1,ng); % true

for g = 2:ng
    
    disp(g)
    
    for l = unique(Lab)
        latentId = id2id(l,p);
        N_clus = sum(Lab == l);
        Y_dc = Y(Lab == l,:)';
        X_dc_blk = [ones(T,1) X_fit(latentId,:,g)'];
        X_dc = sparse(kron(eye(N_clus), X_dc_blk));
        
        
        lamdc = @(dc) exp(X_dc*dc);
        
        Sigdc0 = sparse(kron(eye(N_clus), Sigdc0_sing));
        mudc0 = repmat(mudc0_sing, N_clus, 1);
        
        derdc = @(dc) X_dc'*(Y_dc(:) - lamdc(dc)) - (Sigdc0)\(dc - mudc0);
        hessdc = @(dc) -sparse(X_dc'*diag(lamdc(dc))*X_dc - inv(Sigdc0));
        
        dc_mat = [d_fit(Lab == l, g-1) C_fit(Lab == l,:,g-1)]';
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,dc_mat(:),1e-6,1000);
        
         % use Cholesky decomposition to sample efficiently
        R = chol(-niSigdc,'lower'); % sparse
        z = randn(length(mudc), 1) + R'*mudc;
        dcSamp = reshape(dcSamp,[], N_clus)';
        d_fit(Lab == l,g) = dcSamp(:,1);
        C_fit(Lab == l,:,g) = dcSamp(:,2:end);
    end
end

% worse...

idx = 2;
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
sgtitle('cluster dC')

