addpath(genpath('D:\github\state-space-clustering'));

%% simulation 1: simple example
rng(1)
C = 1;
d = 0;
T = 1000;
X_true = [1*ones(1, round(T/2)) 2*ones(1, T - round(T/2))];
lam_true = exp(X_true);
n = poissrnd(lam_true);

% try different methods
A = 1;
b=0;
x0 = 1;
W0 = 1e-1;
Q = 1e-4;

% filtering
X_filt = ppasmoo_poissexp_v2(n,C,d,x0,W0,A,b,Q);

% direct Newton-Raphson
tic;
der = @(vecX) derX(vecX, d, C, x0, W0, Q, A, b, n);
hess = @(vecX) hessX(vecX, d, C, W0, Q, A, n);
% [X_newton,fx,niSigX,~] = newton(der,hess,ones(T,1)*x0,1e-10,1000);
[X_newton,fx,niSigX,~] = newton(der,hess,X_filt(:),1e-10,1000);
toc;



figure(1)
hold on
plot(X_true, 'k')
plot(X_filt, 'r')
plot(X_newton, 'b')
legend('true', 'filter', 'newton')
hold off

%% simulation 2: ~ LDS
clear all;close all;clc;
rng(2)
nCell = 10;
nClus = 3;
N = nCell*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, nCell);
d1 = ones(nCell,1)*0;
d2 = ones(nCell,1)*1;
d3 = ones(nCell,1)*-5;
d = [d1;d2;d3];
C_all1 = reshape(normrnd(0.08,1e-4,nCell*p,1), [], p);
C_all2 = reshape(normrnd(-0.02,1e-4,nCell*p,1), [], p);
C_all3 = reshape(normrnd(-0.18,1e-4,nCell*p,1), [], p);
C_all = [C_all1; C_all2; C_all3];
C_trans = zeros(nCell*nClus, p*nClus);
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

% let's generate lambda
logLam = zeros(nCell*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));

% try different methods
% filtering
tic;
X_filt = ppasmoo_poissexp_v2(Y,C_trans,d,x0',Q0,A,b,Q);
toc;
% Elapsed time is 0.193763 seconds.

% direct Newton-Raphson
tic;
der = @(vecX) derX(vecX, d, C_trans, x0', Q0, Q, A, b, Y);
hess = @(vecX) hessX(vecX, d, C_trans, Q0, Q, A, Y);
[X_newton,fx,niSigX,~] = newton(der,hess,repmat(x0',T,1),1e-6,1000);
toc;
% Elapsed time is 4.258177 seconds.

% use adaptive smoother as a warm start
tic;
der = @(vecX) derX(vecX, d, C_trans, x0', Q0, Q, A, b, Y);
hess = @(vecX) hessX(vecX, d, C_trans, Q0, Q, A, Y);
[X_newton2,fx,niSigX2,~] = newton(der,hess,X_filt(:),1e-6,1000);
toc;
% Elapsed time is 0.073377 seconds.

figure(1)
subplot(1,2,1)
plot(X', 'k')
title('true')
subplot(1,2,2)
hold on
plot(X_filt', 'r')
plot(reshape(X_newton2, [], T)', 'b')
hold off
legend('filter', 'newton')






