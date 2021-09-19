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
ng = 1000;

%% fitting: MCMC, no update prior
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(N, ng);
C_fit = zeros(N, p, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = repmat(A,1,1,ng); % true
b_fit = repmat(b,1,ng); % true
Q_fit = repmat(Q,1,1,ng); % true

% priors
Q0 = eye(nClus*p)*1e-2;
mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);
mudc0 = zeros(p+1,1);
Sigdc0 = eye(p+1);

% initials
% initial for d_fit: 0
C_fit(:,:,1) = reshape(normrnd(0,1e-2,N*p,1), [], p);
C_trans_tmp = zeros(N, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
end
x0_fit(:,1) = lsqr(C_trans_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
d_tmp = d_fit(:,1);
x0_tmp = x0_fit(:,1);
A_tmp = A_fit(:,:,1);
b_tmp = b_fit(:,1);
Q_tmp = Q_fit(:,:,1);


[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_tmp,...
    x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);

dc_all = [d_fit(:,1) C_fit(:,:,1)]';
acc = zeros(N,1);


% optX.M=0;
% optX.Madapt=1;
% optX.delta = 0.5;
% 
% logNPrior = @(X) -1/2*(X(:,1) - x0_tmp)'*inv(Q0)*(X(:,1) - x0_tmp) -...
%     1/2*trace((X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp)'*inv(Q_tmp)*...
%     (X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp));
% lamX = @(X) exp(C_trans_tmp*X + d_tmp) ;
% lpdf = @(vecX) sum(log(poisspdf(Y, lamX(reshape(vecX, [], T)))), 'all') +...
%     logNPrior(reshape(vecX, [], T));
% glpdf = @(vecX) derX(vecX, d_tmp, C_trans_tmp, x0_tmp,...
%     Q0, Q_tmp, A_tmp, b_tmp, Y);
% fg=@(vecX_r) deal(lpdf(vecX_r'), glpdf(vecX_r')'); % log density and gradient
% 
% [muXvec_NUTS, ~, diagnX]=hmc_nuts(fg, reshape(X_fit(:,:,1),[],1)',optX);
% epsilonX = diagnX.opt.epsilonbar;
% % X_fit(:,:,1) = reshape(muXvec_NUTS(end,:),[], T);


epsilondc = zeros(N,1);
optdc.M=0;
optdc.Madapt=50;
optdc.delta = 0.8;

for i = 1:N
    l = Lab(i);
    latentId = ((l-1)*p+1):(l*p);
    X_tmp = [ones(1, T) ;X_fit(latentId,:,1)]';
    
    lamdc = @(dc) exp(X_tmp*dc);
    
    derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - inv(Sigdc0)*(dc - mudc0);
    lpdf_dc = @(dc) sum(log(poisspdf(Y(i,:)', lamdc(dc)))) +...
        log(mvnpdf(dc, mudc0, Sigdc0));
    
    
    fg_dc=@(dc_r) deal(lpdf_dc(dc_r'), derdc(dc_r')'); % log density and gradient
    [mudc_NUTS, ~, diagndc]=hmc_nuts(fg_dc, [d_fit(i, 1) C_fit(i,:,1)],optdc);
    epsilondc(i) = diagndc.opt.epsilonbar;
    
    d_fit(i,1) = mudc_NUTS(end,1);
    C_fit(i,:,1) = mudc_NUTS(end,2:end);
end


% optX.M=1;
% optX.Madapt=0;
% optX.delta = 0.5;

optdc.M=1;
optdc.Madapt=0;
optdc.delta = 0.8;

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
%     
%     logNPrior = @(X) -1/2*(X(:,1) - x0_tmp)'*inv(Q0)*(X(:,1) - x0_tmp) -...
%         1/2*trace((X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp)'*inv(Q_tmp)*...
%         (X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp));
%     lamX = @(X) exp(C_trans_tmp*X + d_tmp) ;
%     lpdf = @(vecX) sum(log(poisspdf(Y, lamX(reshape(vecX, [], T)))), 'all') +...
%         logNPrior(reshape(vecX, [], T));
%     glpdf = @(vecX) derX(vecX, d_tmp, C_trans_tmp, x0_tmp,...
%         Q0, Q_tmp, A_tmp, b_tmp, Y);
%     fg=@(vecX_r) deal(lpdf(vecX_r'), glpdf(vecX_r')'); % log density and gradient
%     
%     optX.epsilon = epsilonX;
%     [muXvec_NUTS, ~, diagnX]=hmc_nuts(fg, reshape(X_fit(:,:,g-1),[],1)',...
%         optX);
%     X_fit(:,:,g) = reshape(muXvec_NUTS(2,:),[], T);
    
    X_tmp = X_fit(:,:,g-1);
    gradHess = @(vecX) gradHessX(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0, Q_tmp, A_tmp, b_tmp, Y);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    if(sum(isnan(muXvec)) ~= 0)
        disp('use adaptive smoother initial')
        X_tmp = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
        [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    end
    
    % use Cholesky decomposition to sample efficiently
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muXvec), 1) + R'*muXvec;
    Xsamp = R'\z;
    X_fit(:,:,g) = reshape(Xsamp,[], T);
    
    
    % (2) update x0_fit
    Sigx0 = inv(inv(Sigx00) + inv(Q0));
    mux0 = Sigx0*(Sigx00\mux00 + Q0\X_fit(:,1,g));
    x0_fit(:,g) = mvnrnd(mux0, Sigx0)';
    
    % (3) update d_fit & C_fit
    % Laplace approximation
    for i = 1:N
        l = Lab(i);
        latentId = ((l-1)*p+1):(l*p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - inv(Sigdc0)*(dc - mudc0);
        lpdf_dc = @(dc) sum(log(poisspdf(Y(i,:)', lamdc(dc)))) +...
            log(mvnpdf(dc, mudc0, Sigdc0));
        
        optdc.epsilon = epsilondc(i);
        fg_dc=@(dc_r) deal(lpdf_dc(dc_r'), derdc(dc_r')'); % log density and gradient
        [mudc_NUTS, ~, ~]=hmc_nuts(fg_dc, [d_fit(i, g-1) C_fit(i,:,g-1)],optdc);
        
        
        d_fit(i,g) = mudc_NUTS(2,1);
        C_fit(i,:,g) = mudc_NUTS(2,2:end);
    end
    
    figure(1)
    subplot(1,3,1)
    plot(d,d_fit(:,g),'rx');
    ylabel('estimate')
    title('d')
    C_fit_mean = C_fit(:,:,g);
    subplot(1,3,2)
    plot(sum(C_trans(:,1:p:end), 2),C_fit_mean(:,1),'rx');
    title('1st column of C')
    xlabel('true')
    subplot(1,3,3)
    plot(sum(C_trans(:,2:p:end), 2),C_fit_mean(:,2),'rx');
    title('2nd column of C')
    sgtitle('No Update of Prior')
    
end


d_norm = zeros(g, 1);
C_norm_fro = zeros(g, 1);

for k = 1:g
    d_norm(k) = norm(d_fit(:,k), 'fro');
    C_norm_fro(k) = norm(C_fit(:,:,k), 'fro');
end

figure
subplot(1,2,1)
plot(d_norm)
title('norm of d')
subplot(1,2,2)
plot(C_norm_fro)
title('Frobenius norm of C')


idx = round(ng/2):ng;

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


%% fitting: MCMC, update priors

X_fit2 = zeros(nClus*p, T, ng);
d_fit2 = zeros(N, nClus, ng);
C_fit2 = zeros(N, nClus*p, ng);
mudc_fit2 = zeros(p+1, nClus, ng);
Sigdc_fit2 = zeros(p+1, p+1, nClus, ng);
x0_fit2 = zeros(nClus*p, ng);
A_fit2 = repmat(A,1,1,ng); % true
b_fit2 = repmat(b,1,ng); % true
Q_fit2 = repmat(Q,1,1,ng); % true

% priors
% place-holder...
Q0 = eye(nClus*p)*1e-2;

mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1);

Psidc0 = eye(p+1);
nudc0 = p+1+2;

% initials
% initial for d_fit2: 0
C_raw = reshape(normrnd(0,1e-2,N*p,1), [], p);
d_tmp = zeros(N,1);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    C_fit2(Lab == k, ladid_tmp, 1) = C_raw(Lab == k, :);
    d_tmp(Lab == k) = d_fit2(Lab ==k, k);
end

Sigdc_fit2(:,:,1:nClus,1) = repmat(eye(p+1)*1e-2,1,1,nClus);


x0_fit2(:,1) = lsqr(C_fit2(:,:,1),(log(mean(Y(:,1:10),2))-d_tmp));
[X_tmp,~,~] = ppasmoo_poissexp_v2(Y,C_fit2(:,:,1),d_tmp,...
    x0_fit2(:,1),Q0,A_fit2(:,:,1),b_fit2(:,1),Q_fit2(:,:,1));
gradHess = @(vecX) gradHessX(vecX, d_tmp, C_fit2(:,:,1), x0_fit2(:,1), Q0,...
    Q_fit2(:,:,1), A_fit2(:,:,1), b_fit2(:,1), Y);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-10,1000);
X_fit2(:,:,1) = reshape(muXvec, [], T);


dc_all2 = [sum(d_fit2(:,:,1), 2) sum(C_fit2(:,1:p:end), 2) sum(C_fit2(:,2:p:end), 2)]';
acc2 = zeros(N,1);
for g = 2:ng
    
    disp(g)
    
    % (1) update X_fit2
    
    d_raw = d_fit2(:,:,g-1);
    I = (1 : size(d_raw, 1)) .';
    k = sub2ind(size(d_raw), I, Lab');
    d_tmp = d_raw(k);
    C_tmp = C_fit2(:,:,g-1);
    
    x0_tmp = x0_fit2(:,g-1);
    A_tmp = A_fit2(:,:,g-1);
    b_tmp = b_fit2(:,g-1);
    Q_tmp = Q_fit2(:,:,g-1);
    X_tmp = X_fit2(:,:,g-1);
    
    gradHess = @(vecX) gradHessX(vecX, d_tmp, C_tmp, x0_tmp, Q0, Q_tmp, A_tmp, b_tmp, Y);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    if(sum(isnan(muXvec)) ~= 0)
        disp('use adaptive smoother initial')
        X_tmp = ppasmoo_poissexp_v2(Y,C_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
        [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    end
    
    % tic;
    % use Cholesky decomposition to sample efficiently
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muXvec), 1) + R'*muXvec;
    Xsamp = R'\z;
    X_fit2(:,:,g) = reshape(Xsamp,[], T);
    % toc;
    
    % (2) update x0_fit2
    Sigx0 = inv(inv(Sigx00) + inv(Q0));
    mux0 = Sigx0*(Sigx00\mux00 + Q0\X_fit2(:,1,g));
    x0_fit2(:,g) = mvnrnd(mux0, Sigx0)';
    % disp(x0_fit2(:,g))
    
    % (3) update d_fit2 & C_fit2
    % Laplace approximation
    
    for i = 1:N
        l = Lab(i);
        latentId = id2id(l,p);
        X_tmp = [ones(1, T) ;X_fit2(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - Sigdc_fit2(:,:,l,g-1)\(dc - mudc_fit2(:,l,g-1));
        hessdc = @(dc) -X_tmp'*diag(lamdc(dc))*X_tmp - inv(Sigdc_fit2(:,:,l,g-1));
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
            [d_fit2(i,l, g-1) C_fit2(i,latentId,g-1)]',1e-8,1000);
        if(sum(isnan(mudc)) ~= 0)
            [mudc,~,niSigdc,~] = newton(derdc,hessdc,deltadc0,1e-8,1000);
        end
        
        
        if g < 1 % g < round(ng/2)
            Sigdc = -inv(niSigdc);
            Sigdc = (Sigdc + Sigdc')/2;
            dc_all2(:,i) = mvnrnd(mudc, Sigdc);
        else
            R = chol(-niSigdc,'lower'); % sparse
            z = randn(length(dc_all2(:,i)), 1) + R'*dc_all2(:,i);
            dcStar = R'\z;
            
            % lhr
            lhr = sum(log(poisspdf(Y(i,:)', lamdc(dcStar)))) -...
            sum(log(poisspdf(Y(i,:)', lamdc(dc_all2(:,i))))) +...
            log(mvnpdf(dcStar, mudc_fit2(:,l,g-1), Sigdc_fit2(:,:,l,g-1))) -...
            log(mvnpdf(dc_all2(:,i), mudc_fit2(:,l,g-1), Sigdc_fit2(:,:,l,g-1)));
        
            if(log(rand(1)) < lhr)
                dc_all2(:,i) = dcStar;
                acc2(i) = acc2(i)+1;
            end
        end
        
        d_fit2(i,l,g) = dc_all2(1,i);
        C_fit2(i,latentId,g) = dc_all2(2:end,i);
        
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

% acc2/ng


d_norm = zeros(g, 1);
C_norm_fro = zeros(g, 1);

for k = 1:g
    d_norm(k) = norm(d_fit2(:,:,k), 'fro');
    C_norm_fro(k) = norm(C_fit2(:,:,k), 'fro');
end

figure
subplot(1,2,1)
plot(d_norm)
title('norm of d')
subplot(1,2,2)
plot(C_norm_fro)
title('Frobenius norm of C')


idx = round(ng/2):ng;


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

figure
subplot(3,2,1)
plot(X(1:p,:)')
title('true')
subplot(3,2,2)
plot(mean(X_fit2(1:p,:,idx), 3)')
title('fit')
subplot(3,2,3)
plot(X(p+1:2*p,:)')
subplot(3,2,4)
plot(mean(X_fit2(p+1:2*p,:,idx), 3)')
subplot(3,2,5)
plot(X(2*p+1:3*p,:)')
subplot(3,2,6)
plot(mean(X_fit2(2*p+1:3*p,:,idx), 3)')


